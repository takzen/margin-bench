"""dsp.scale_readout: make a weighing scale read right.

A load cell under a 0..50 kg platform feeds an ADC. The raw signal carries the
load plus mains hum (fundamental, 3rd and 5th harmonic, frequency drifting a
little), a strong tone from a nearby switching converter and broadband noise.
The sample rate, the mains frequency (near 50 or 60 Hz, possibly off by a few
tenths of a hertz), the tone and the noise come from the seed; the model is told
the sample rate only.

The model builds the processing chain from blocks (moving average, notch, IIR
lowpass, FIR lowpass, decimation) within a budget of multiply-adds per ADC
sample. It can look at the signal with a spectrum analyzer and an oscilloscope,
put calibration weights on the platform and watch the display. When it submits,
an acceptance test puts weights on and off and grades the display readings:
settling time, noise and accuracy.

Difficulty rule: the limits are the worst result over a family of textbook
designs run through the same test: two moving averages over one mains period,
decimation by the same length, then a second-order Butterworth lowpass with its
cut-off at a tenth of the decimated rate, scaled by 0.7, 1 and 1.4. Settling
gets 15 % slack, noise 20 %. A second-order lowpass alone cannot meet noise and
settling together, and decimating without filtering first folds the hum and
the tone into the reading.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as sps

from margin_bench.core import Score
from margin_bench.grading import scoring
from margin_bench.play.env import ToolError, tool
from margin_bench.tasks.rng import SeededRandom

FS_CHOICES = (3200, 4000, 5000, 6400)
FULL_SCALE = 50.0  # kg
DISPLAY_HZ = 10
BUDGET = 12.0  # multiply-adds per ADC sample
MAX_CALLS = 40
SIGNAL_BUDGET_S = 1200.0
MAX_CAPTURE_S = 300.0
MAX_BLOCKS = 10
SETTLE_BAND = 0.05  # kg
ACCURACY = 0.01  # kg
DRIFT_HZ = 0.05
DRIFT_PERIOD_S = 20.0
SPECTRUM_PEAKS = 12
SCOPE_POINTS = 200
TREND_ROWS = 300
CHUNK = 1 << 16

T_FIRST = 2.0
PLATEAU_S = 4.0
WINDOW_S = 2.0  # noise and accuracy are measured over the last WINDOW_S of each plateau

FAMILY_SCALES = (0.7, 1.0, 1.4)
SETTLE_SLACK = 1.15
NOISE_SLACK = 1.2

SPECTRUM_WINDOWS = {"rectangular": "boxcar", "hann": "hann", "blackman": "blackman", "flattop": "flattop"}
FIR_WINDOWS = {"rectangular": "boxcar", "hann": "hann", "hamming": "hamming", "blackman": "blackman"}
LOWPASS_KINDS = ("butter", "bessel", "cheby1")


def _seed32(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


@dataclass(frozen=True)
class Interference:
    fs: int
    mains_hz: float
    hum_kg: tuple[float, float, float]  # 1st, 3rd, 5th harmonic amplitudes
    hum_phase: tuple[float, float, float]
    tone_hz: float
    tone_kg: float
    tone_phase: float
    noise_kg: float


class Source:
    """The ADC: load + drifting mains hum + switching tone + white noise.

    Noise comes from NumPy's legacy RandomState, whose stream NumPy keeps frozen
    across versions, so an episode replays bit for bit.
    """

    def __init__(self, s: Interference, noise_seed: int) -> None:
        self.s = s
        self.i = 0
        self.load = 0.0
        self.rng = np.random.RandomState(noise_seed)

    def read(self, m: int) -> np.ndarray:
        s = self.s
        t = (self.i + np.arange(m)) / s.fs
        phase = 2 * np.pi * (s.mains_hz * t - DRIFT_HZ * DRIFT_PERIOD_S / (2 * np.pi) * np.cos(2 * np.pi * t / DRIFT_PERIOD_S))
        x = np.full(m, self.load)
        for k, a, p in zip((1, 3, 5), s.hum_kg, s.hum_phase):
            x += a * np.sin(k * phase + p)
        x += s.tone_kg * np.sin(2 * np.pi * s.tone_hz * t + s.tone_phase)
        x += s.noise_kg * self.rng.standard_normal(m)
        self.i += m
        return x


def _int(spec: dict[str, Any], key: str, lo: int, hi: int) -> int:
    v = spec.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        raise ToolError(f"{spec.get('type')}: {key} must be an integer")
    if not lo <= v <= hi:
        raise ToolError(f"{spec.get('type')}: {key} must be between {lo} and {hi}")
    return int(v)


def _float(spec: dict[str, Any], key: str, lo: float, hi: float, *, open_hi: bool = False) -> float:
    v = spec.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"{spec.get('type')}: {key} must be a number")
    if not lo < v < hi if open_hi else not lo < v <= hi:
        raise ToolError(f"{spec.get('type')}: {key} must be above {lo:g} and {'below' if open_hi else 'at most'} {hi:g}")
    return float(v)


def _choice(spec: dict[str, Any], key: str, options: Any) -> str:
    v = spec.get(key)
    if v not in options:
        raise ToolError(f"{spec.get('type')}: {key} must be one of {', '.join(options)}")
    return v


ALLOWED_KEYS = {
    "moving_average": {"n"},
    "notch": {"f0_hz", "q"},
    "lowpass": {"kind", "order", "fc_hz"},
    "fir_lowpass": {"taps", "fc_hz", "window"},
    "decimate": {"factor"},
}


class Chain:
    """The processing chain: a list of blocks, each with its own state and rate."""

    def __init__(self, blocks: list[dict[str, Any]], fs: int) -> None:
        if not isinstance(blocks, list):
            raise ToolError("blocks must be a list")
        if len(blocks) > MAX_BLOCKS:
            raise ToolError(f"at most {MAX_BLOCKS} blocks")
        self.fs = fs
        self.stages: list[dict[str, Any]] = []
        self.spec: list[dict[str, Any]] = []
        rate, decim, cost = float(fs), 1, 0.0
        for i, b in enumerate(blocks):
            if not isinstance(b, dict) or b.get("type") not in ALLOWED_KEYS:
                raise ToolError(f"block {i + 1}: type must be one of {', '.join(ALLOWED_KEYS)}")
            kind = b["type"]
            extra = set(b) - ALLOWED_KEYS[kind] - {"type"}
            if extra:
                raise ToolError(f"block {i + 1} ({kind}): unknown parameters {', '.join(sorted(extra))}")
            nyq = rate / 2
            if kind == "moving_average":
                n = _int(b, "n", 1, 4096)
                st = {"op": "lfilter", "b": np.full(n, 1.0 / n), "a": np.array([1.0])}
                c, norm = 2.0, {"type": kind, "n": n}
            elif kind == "notch":
                f0 = _float(b, "f0_hz", 0.0, nyq, open_hi=True)
                q = _float(b, "q", 0.0, 1000.0)
                bb, aa = sps.iirnotch(f0, q, fs=rate)
                st = {"op": "lfilter", "b": bb, "a": aa}
                c, norm = 5.0, {"type": kind, "f0_hz": f0, "q": q}
            elif kind == "lowpass":
                lk = _choice(b, "kind", LOWPASS_KINDS)
                order = _int(b, "order", 1, 8)
                fc = _float(b, "fc_hz", 0.0, nyq, open_hi=True)
                if lk == "butter":
                    sos = sps.butter(order, fc, fs=rate, output="sos")
                elif lk == "bessel":
                    sos = sps.bessel(order, fc, fs=rate, output="sos", norm="mag")
                else:
                    sos = sps.cheby1(order, 0.5, fc, fs=rate, output="sos")
                st = {"op": "sos", "sos": sos}
                c, norm = 5.0 * math.ceil(order / 2), {"type": kind, "kind": lk, "order": order, "fc_hz": fc}
            elif kind == "fir_lowpass":
                taps = _int(b, "taps", 3, 2001)
                fc = _float(b, "fc_hz", 0.0, nyq, open_hi=True)
                win = _choice(b, "window", FIR_WINDOWS)
                st = {"op": "lfilter", "b": sps.firwin(taps, fc, window=FIR_WINDOWS[win], fs=rate), "a": np.array([1.0])}
                c, norm = float(taps), {"type": kind, "taps": taps, "fc_hz": fc, "window": win}
            else:
                f = _int(b, "factor", 2, 4096)
                st = {"op": "decimate", "factor": f, "phase": 0}
                c, norm = 0.0, {"type": kind, "factor": f}
            cost += c / decim
            norm["rate_in_hz"] = round(rate, 6)
            if kind == "decimate":
                decim *= st["factor"]
                rate = fs / decim
            else:
                if st["op"] == "lfilter":
                    st["zi"] = np.zeros(max(len(st["a"]), len(st["b"])) - 1)
                else:
                    st["zi"] = np.zeros((st["sos"].shape[0], 2))
            self.stages.append(st)
            self.spec.append(norm)
        if rate < DISPLAY_HZ - 1e-9:
            raise ToolError(f"output rate {rate:g} Hz is below the display rate of {DISPLAY_HZ} Hz")
        if cost > BUDGET + 1e-9:
            raise ToolError(f"cost {cost:.2f} multiply-adds per ADC sample is over the budget of {BUDGET:g}")
        self.cost = cost
        self.out_rate = rate

    def blocks(self) -> list[dict[str, Any]]:
        """The chain as set_chain takes it, to build a fresh copy from rest."""
        return [{k: v for k, v in b.items() if k != "rate_in_hz"} for b in self.spec]

    def process(self, x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for st in self.stages:
            if not len(x):
                break
            if st["op"] == "lfilter":
                x, st["zi"] = sps.lfilter(st["b"], st["a"], x, zi=st["zi"])
            elif st["op"] == "sos":
                x, st["zi"] = sps.sosfilt(st["sos"], x, zi=st["zi"])
            else:
                f = st["factor"]
                keep = np.arange((-st["phase"]) % f, len(x), f)
                st["phase"] = (st["phase"] + len(x)) % f
                x, idx = x[keep], idx[keep]
        return x, idx


class Bench:
    """Source, chain and display, advanced together. Every sample goes through the chain."""

    def __init__(self, s: Interference, noise_seed: int) -> None:
        self.src = Source(s, noise_seed)
        self.fs = s.fs
        self.chain = Chain([], s.fs)
        self.tick = s.fs // DISPLAY_HZ
        self.reading = 0.0
        self.frames: list[list[float]] = []

    @property
    def t(self) -> float:
        return self.src.i / self.fs

    def advance(self, m: int, keep: str | None = None) -> np.ndarray | None:
        """Run m ADC samples through; optionally return the raw ("adc") or chain ("output") samples."""
        kept: list[np.ndarray] = []
        done = 0
        while done < m:
            n = min(CHUNK, m - done)
            i0 = self.src.i
            x = self.src.read(n)
            idx = i0 + np.arange(n)
            y, yi = self.chain.process(x, idx)
            ticks = np.arange((i0 // self.tick + 1) * self.tick - 1, i0 + n, self.tick)
            ticks = ticks[ticks >= i0]
            pos = np.searchsorted(yi, ticks, side="right") - 1
            for j, p in zip(ticks, pos):
                if p >= 0:
                    self.reading = float(y[p])
                self.frames.append([round((j + 1) / self.fs, 3), self.src.load, round(self.reading, 5),
                                    round(self.reading - self.src.load, 5)])
            if keep == "adc":
                kept.append(x)
            elif keep == "output":
                kept.append(y)
            done += n
        return np.concatenate(kept) if keep else None


FRAME_COLUMNS = ["t_s", "load_kg", "reading_kg", "error_kg"]


def _settle(ts: np.ndarray, err: np.ndarray, t0: float, t1: float) -> float:
    sel = (ts > t0) & (ts <= t1)
    out = ts[sel][np.abs(err[sel]) > SETTLE_BAND]
    if not len(out):
        return 0.0
    if out[-1] >= t1 - 1e-9:
        return math.inf
    return float(out[-1] - t0 + 1.0 / DISPLAY_HZ)


@dataclass(frozen=True)
class Scenario:
    loads: tuple[float, float]


def acceptance(s: Interference, sc: Scenario, blocks: list[dict[str, Any]], noise_seed: int) -> dict[str, Any]:
    bench = Bench(s, noise_seed)
    bench.chain = Chain(blocks, s.fs)
    plan = [(0.0, T_FIRST), (sc.loads[0], PLATEAU_S), (sc.loads[1], PLATEAU_S), (0.0, PLATEAU_S)]
    for load, dur in plan:
        bench.src.load = load
        bench.advance(round(dur * s.fs))
    f = np.array(bench.frames)
    ts, err = f[:, 0], f[:, 3]
    changes = [T_FIRST + k * PLATEAU_S for k in range(3)]
    settles, stds, means = [], [], []
    for c in changes:
        end = c + PLATEAU_S
        settles.append(_settle(ts, err, c, end))
        w = (ts > end - WINDOW_S) & (ts <= end)
        stds.append(float(np.std(f[w, 2])))
        means.append(float(abs(np.mean(err[w]))))
    return {"frames": bench.frames, "settle_s": settles, "noise_kg": stds, "offset_kg": means}


def reference_chain(fs: int, mains_hz: float, scale: float = 1.0) -> list[dict[str, Any]]:
    n = round(fs / mains_hz)
    return [
        {"type": "moving_average", "n": n},
        {"type": "moving_average", "n": n},
        {"type": "decimate", "factor": n},
        {"type": "lowpass", "kind": "butter", "order": 2, "fc_hz": round(scale * fs / n / 10, 3)},
    ]


def spectrum(x: np.ndarray, rate: float, points: int, window: str, averages: int) -> dict[str, Any]:
    segs = x[: points * averages].reshape(averages, points)
    dc = float(np.mean(segs))
    w = sps.get_window(SPECTRUM_WINDOWS[window], points, fftbins=True)
    X = np.fft.rfft((segs - segs.mean(axis=1, keepdims=True)) * w, axis=1)
    amp = np.sqrt(np.mean(np.abs(X) ** 2, axis=0)) * 2.0 / w.sum()
    amp[0] = 0.0
    floor = float(np.median(amp[1:]))
    k = np.arange(1, len(amp) - 1)
    is_peak = (amp[k] > amp[k - 1]) & (amp[k] >= amp[k + 1]) & (amp[k] > 4.0 * floor)
    peaks = k[is_peak]
    peaks = peaks[np.argsort(amp[peaks])[::-1][:SPECTRUM_PEAKS]]
    peaks = np.sort(peaks)
    return {
        "amp": amp,
        "result": {
            "rate_hz": rate,
            "points": points,
            "window": window,
            "averages": averages,
            "resolution_hz": rate / points,
            "dc_kg": round(dc, 5),
            "noise_floor_kg": float(f"{floor:.3g}"),
            "peaks": [[round(float(p * rate / points), 3), float(f"{amp[p]:.4g}")] for p in peaks],
        },
    }


PROMPT = """\
You are commissioning the readout of a digital weighing scale. The hardware is simulated; treat it as a real bench.

Hardware
- A load cell under a 0..50 kg platform, an amplifier and an ADC sampling at {fs} Hz. The samples arrive already \
scaled to kilograms.
- The installation is electrically noisy. Nothing else about the interference is known: measure it.
- The display updates {display} times per second and shows the newest output sample of your processing chain.

Processing chain. You build it from these blocks, applied in order, in float64 arithmetic. Each block's cost is in \
multiply-adds per sample at its own input rate; the total, referred to the ADC rate (a block after decimation by 8 \
costs an eighth), must stay within {budget:g} per ADC sample.
- moving_average(n): mean of the last n samples. Cost 2 (running sum).
- notch(f0_hz, q): second-order IIR notch. Cost 5.
- lowpass(kind, order, fc_hz): IIR lowpass; kind butter, bessel or cheby1 (0.5 dB ripple); order 1..8; fc_hz is \
the -3 dB point for butter and bessel, the ripple edge for cheby1. Cost 5 per second-order section.
- fir_lowpass(taps, fc_hz, window): windowed-sinc FIR with unity gain at DC; window rectangular, hann, hamming or \
blackman. Cost = taps.
- decimate(factor): keeps every factor-th sample and drops the rest; it does no filtering of its own. Cost 0. Blocks \
after it run at the lower rate. The chain output rate must stay at least {display} Hz.
A new chain starts from rest (all states zero). With no chain the display shows raw ADC samples. At most {max_blocks} blocks.

Instruments: a spectrum analyzer and an oscilloscope, on the ADC signal or on the chain output, and calibration \
weights for the platform. Right now the platform is empty and there is no chain.

Acceptance test. It runs after you call submit, with the chain you leave:
1. The chain starts from rest, the platform empty. At t = 2 s a weight goes on (5..45 kg). 4 s later it is swapped \
for another (5..45 kg). 4 s after that the platform is emptied, and the test ends 4 s later.
2. Requirements, on the display readings:
- after each change, within +/-{band:g} kg of the true load no later than {settle} s, and staying there until the next change;
- over the last {window:g} s before each change and before the end: standard deviation of the readings at most \
{noise} kg, and the mean within +/-{accuracy:g} kg of the true load.

You have {budget_min:g} minutes of signal time (each capture and run uses its own duration) and at most {max_calls} \
tool calls. Work like an engineer: measure, design, check on the bench with known weights, then call submit.
"""


class ScaleReadout:
    id = "dsp.scale_readout"
    domain = "dsp"
    title = "Weighing scale readout"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        fs = rng.choice(FS_CHOICES)
        base = rng.choice((50.0, 60.0))
        hum1 = rng.uniform(2.0, 5.0)
        self.interference = Interference(
            fs=fs,
            mains_hz=round(base + rng.uniform(-0.3, 0.3), 3),
            hum_kg=(hum1, hum1 * rng.uniform(0.2, 0.4), hum1 * rng.uniform(0.05, 0.2)),
            hum_phase=(rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)),
            tone_hz=round(fs * rng.uniform(0.28, 0.46), 2),
            tone_kg=rng.uniform(0.3, 1.5),
            tone_phase=rng.uniform(0, 2 * math.pi),
            noise_kg=rng.uniform(0.03, 0.12),
        )
        l1 = round(rng.uniform(5.0, 45.0), 1)
        l2 = round(rng.uniform(5.0, 40.0), 1)
        if abs(l2 - l1) < 5.0:
            l2 = round(l2 + 5.0, 1)
        self.scenario = Scenario(loads=(l1, l2))
        s = self.interference
        family = [acceptance(s, self.scenario, reference_chain(fs, s.mains_hz, a), self._noise("acceptance"))
                  for a in FAMILY_SCALES]
        self.limits = {
            "settle_s": max(0.3, math.ceil(SETTLE_SLACK * max(max(r["settle_s"]) for r in family) * 20 - 1e-9) / 20),
            "noise_kg": float(f"{NOISE_SLACK * max(max(r['noise_kg']) for r in family):.2g}"),
        }
        self.bench = Bench(s, self._noise("commission"))
        self.calls = 0
        self._submitted = False
        self.result: dict[str, Any] | None = None
        self._last_amp: np.ndarray | None = None

    def _noise(self, what: str) -> int:
        return _seed32(self.id, self.seed, what)

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def sim_time(self) -> float:
        return self.bench.t

    def prompt(self) -> str:
        return PROMPT.format(
            fs=self.interference.fs, display=DISPLAY_HZ, budget=BUDGET, max_blocks=MAX_BLOCKS, band=SETTLE_BAND,
            settle=f"{self.limits['settle_s']:g}", window=WINDOW_S, noise=f"{self.limits['noise_kg']:g}",
            accuracy=ACCURACY, budget_min=SIGNAL_BUDGET_S / 60, max_calls=MAX_CALLS,
        )

    def tools(self) -> list[dict[str, Any]]:
        num, integer = {"type": "number"}, {"type": "integer"}
        block = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": list(ALLOWED_KEYS)},
                "n": integer, "f0_hz": num, "q": num, "kind": {"type": "string", "enum": list(LOWPASS_KINDS)},
                "order": integer, "fc_hz": num, "taps": integer,
                "window": {"type": "string", "enum": list(FIR_WINDOWS)}, "factor": integer,
            },
            "required": ["type"],
        }
        source = {"type": "string", "enum": ["adc", "output"]}
        return [
            tool("set_chain", "Replace the processing chain with these blocks, in order. Returns its cost and output rate.",
                 {"blocks": {"type": "array", "items": block}}, ["blocks"]),
            tool("spectrum", "Spectrum analyzer: averaged amplitude spectrum of the ADC signal or of the chain output. "
                 "Returns resolution, DC, noise floor and the strongest peaks as [frequency Hz, amplitude kg].",
                 {"points": {**integer, "enum": [1024, 2048, 4096, 8192, 16384, 32768, 65536]},
                  "window": {"type": "string", "enum": list(SPECTRUM_WINDOWS)},
                  "averages": {**integer, "minimum": 1, "maximum": 16}, "source": source},
                 ["points", "window", "averages", "source"]),
            tool("scope", f"Oscilloscope: capture `ms` milliseconds; returns statistics and up to {SCOPE_POINTS} samples.",
                 {"ms": {**num, "minimum": 1, "maximum": 2000}, "source": source}, ["ms", "source"]),
            tool("set_load", "Put calibration weights on the platform (total in kg, 0 to empty it).",
                 {"kg": {**num, "minimum": 0, "maximum": FULL_SCALE}}, ["kg"]),
            tool("run", f"Let the bench run and return the display readings ({DISPLAY_HZ} per second, at most "
                 f"{TREND_ROWS} rows) with their statistics.",
                 {"seconds": {**num, "minimum": 0.1, "maximum": MAX_CAPTURE_S}}, ["seconds"]),
            tool("submit", "Finish commissioning. The acceptance test then runs with the current chain."),
        ]

    def _budget(self, seconds: float) -> int:
        left = SIGNAL_BUDGET_S - self.bench.t
        if seconds > MAX_CAPTURE_S:
            raise ToolError(f"one capture or run can take at most {MAX_CAPTURE_S:g} s of signal")
        if seconds > left + 1e-9:
            raise ToolError(f"that needs {seconds:.1f} s of signal but only {max(0.0, left):.1f} s are left")
        return max(1, round(seconds * self.interference.fs))

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise ToolError("already submitted")
        self.calls += 1
        if self.calls > MAX_CALLS and name != "submit":
            raise ToolError(f"tool call limit ({MAX_CALLS}) reached; call submit")
        b, fs = self.bench, self.interference.fs
        if name == "set_chain":
            chain = Chain(args.get("blocks"), fs)
            b.chain = chain
            return {"t_s": round(b.t, 3), "blocks": chain.spec, "cost_per_adc_sample": round(chain.cost, 3),
                    "budget": BUDGET, "output_rate_hz": round(chain.out_rate, 6)}
        if name == "set_load":
            kg = args.get("kg")
            if isinstance(kg, bool) or not isinstance(kg, (int, float)) or not 0 <= kg <= FULL_SCALE:
                raise ToolError(f"kg must be a number from 0 to {FULL_SCALE:g}")
            b.src.load = float(kg)
            return {"t_s": round(b.t, 3), "load_kg": b.src.load}
        if name == "spectrum":
            points = args.get("points")
            if points not in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
                raise ToolError("points must be a power of two from 1024 to 65536")
            window = args.get("window")
            if window not in SPECTRUM_WINDOWS:
                raise ToolError(f"window must be one of {', '.join(SPECTRUM_WINDOWS)}")
            avg = args.get("averages")
            if isinstance(avg, bool) or not isinstance(avg, int) or not 1 <= avg <= 16:
                raise ToolError("averages must be an integer from 1 to 16")
            src = _source(args)
            rate = fs if src == "adc" else b.chain.out_rate
            m = self._budget(points * avg / rate)
            data = b.advance(m, keep=src)
            if len(data) < points * avg:
                raise ToolError("not enough output samples were captured")
            sp = spectrum(data, rate, points, window, avg)
            self._last_amp = sp["amp"]
            return {"t_s": round(b.t, 3), "source": src, **sp["result"]}
        if name == "scope":
            ms = args.get("ms")
            if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not 1 <= ms <= 2000:
                raise ToolError("ms must be a number from 1 to 2000")
            src = _source(args)
            data = b.advance(self._budget(ms / 1000), keep=src)
            rate = fs if src == "adc" else b.chain.out_rate
            if not len(data):
                return {"t_s": round(b.t, 3), "source": src, "samples": 0,
                        "note": "no chain output sample in that window; capture longer"}
            every = max(1, math.ceil(len(data) / SCOPE_POINTS))
            return {
                "t_s": round(b.t, 3), "source": src, "rate_hz": rate, "samples": len(data), "every_nth": every,
                "mean": round(float(np.mean(data)), 5), "rms_ac": round(float(np.std(data)), 5),
                "min": round(float(np.min(data)), 5), "max": round(float(np.max(data)), 5),
                "trace": [round(float(v), 4) for v in data[::every]],
            }
        if name == "run":
            seconds = args.get("seconds")
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0.1 <= seconds <= MAX_CAPTURE_S:
                raise ToolError(f"seconds must be a number from 0.1 to {MAX_CAPTURE_S:g}")
            n0 = len(b.frames)
            b.advance(self._budget(seconds))
            rows = b.frames[n0:]
            r = np.array([f[2] for f in rows]) if rows else np.array([b.reading])
            every = max(1, math.ceil(len(rows) / TREND_ROWS))
            return {
                "t_s": round(b.t, 3), "load_kg": b.src.load,
                "readings": {"columns": ["t_s", "reading_kg"], "rows": [[f[0], round(f[2], 4)] for f in rows[::every]]},
                "mean_kg": round(float(np.mean(r)), 5), "std_kg": round(float(np.std(r)), 5),
                "min_kg": round(float(np.min(r)), 5), "max_kg": round(float(np.max(r)), 5),
                "signal_left_s": round(SIGNAL_BUDGET_S - b.t, 1),
            }
        if name == "submit":
            self._submitted = True
            return {"submitted": True, "t_s": round(b.t, 3)}
        raise ToolError(f"unknown tool {name!r}")

    def finish(self) -> Score:
        if self.result is None:
            self.result = acceptance(self.interference, self.scenario, self.bench.chain.blocks(),
                                     self._noise("acceptance"))
        r, lim = self.result, self.limits
        settle, noise, offset = max(r["settle_s"]), max(r["noise_kg"]), max(r["offset_kg"])
        ratios = {
            "settle_ratio": scoring.at_most(settle, lim["settle_s"]),
            "noise_ratio": scoring.at_most(noise, lim["noise_kg"]),
            "accuracy_ratio": scoring.at_most(offset, ACCURACY),
        }
        metrics = {
            "submitted": self._submitted,
            "tool_calls": self.calls,
            "signal_used_s": round(self.bench.t, 2),
            "settle_s": [None if math.isinf(v) else v for v in r["settle_s"]],
            "noise_kg": r["noise_kg"],
            "offset_kg": r["offset_kg"],
            "final_chain": self.bench.chain.spec,
            "cost_per_adc_sample": round(self.bench.chain.cost, 3),
            **ratios,
        }
        score = scoring.combine(*ratios.values())
        return Score(score=score, passed=score >= 1.0, metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        if name == "set_chain":
            chain = " -> ".join(_block_label(b) for b in result["blocks"]) or "no chain (raw ADC to display)"
            return f"{chain}; cost {result['cost_per_adc_sample']:g}/{BUDGET:g} per ADC sample, output {result['output_rate_hz']:g} Hz", False
        if name == "set_load":
            return f"platform load {result['load_kg']:g} kg", False
        if name == "spectrum":
            peaks = ", ".join(f"{f:g} Hz {a:.3g} kg" for f, a in sorted(result["peaks"], key=lambda p: -p[1])[:4])
            return (f"{result['source']} spectrum, {result['points']} points {result['window']} x{result['averages']}, "
                    f"{result['resolution_hz']:.3g} Hz/bin: {peaks or 'no peaks above the floor'}"), False
        if name == "scope":
            if not result.get("samples"):
                return f"{result['source']} scope: no samples", False
            return (f"{result['source']} scope, {result['samples']} samples: mean {result['mean']:.4f} kg, "
                    f"rms {result['rms_ac']:.4f} kg, {result['min']:.3f}..{result['max']:.3f} kg"), False
        if name == "run":
            return (f"display at load {result['load_kg']:g} kg: mean {result['mean_kg']:.4f}, std {result['std_kg']:.4f}, "
                    f"{result['min_kg']:.3f}..{result['max_kg']:.3f} kg"), False
        if name == "submit":
            return "handed over for the acceptance test", False
        return "", False

    def visual(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        if name == "spectrum" and self._last_amp is not None:
            amp, rate, n = self._last_amp, result["rate_hz"], result["points"]
            f = np.arange(len(amp)) * rate / n
            # log-spaced max-hold down to about 400 points, so narrow peaks survive
            edges = np.unique(np.round(np.geomspace(1, len(amp) - 1, 420)).astype(int))
            fs_, db = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = lo + int(np.argmax(amp[lo:hi]))
                fs_.append(round(float(f[k]), 3))
                db.append(round(float(20 * np.log10(max(amp[k], 1e-9))), 1))
            self._last_amp = None
            return {"spectrum": {"source": result["source"], "f_hz": fs_, "db": db}}
        if name == "scope" and result.get("samples"):
            dt = 1000.0 * result["every_nth"] / result["rate_hz"]
            return {"scope": {"source": result["source"], "dt_ms": dt, "y": result["trace"]}}
        return None

    def view(self) -> dict[str, Any]:
        phases: list[dict[str, Any]] = [
            {"key": "commissioning", "name": "Commissioning", "rows": self.bench.frames, "events": [], "marks": []}
        ]
        lim = self.limits
        if self.result is not None:
            l1, l2 = self.scenario.loads
            changes = [T_FIRST + k * PLATEAU_S for k in range(3)]
            phases.append({
                "key": "acceptance",
                "name": "Acceptance test",
                "rows": self.result["frames"],
                "events": [{"t_s": changes[0], "label": f"{l1:g} kg on"}, {"t_s": changes[1], "label": f"{l2:g} kg"},
                           {"t_s": changes[2], "label": "emptied"}],
                "marks": [{"t_s": c + lim["settle_s"], "label": "settle limit"} for c in changes],
            })
        out: dict[str, Any] = {
            "kind": "scale",
            "columns": FRAME_COLUMNS,
            "phases": phases,
            "scene": {"fs": self.interference.fs, "full_scale_kg": FULL_SCALE, "band_kg": SETTLE_BAND,
                      "budget": BUDGET, "final_chain": self.bench.chain.spec,
                      "final_cost": round(self.bench.chain.cost, 3)},
            "panels": [
                {
                    "title": "Display reading and true load", "unit": "kg", "range": [0.0, FULL_SCALE],
                    "ticks": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0], "height": 1.4,
                    "series": [{"col": 1, "label": "true load", "color": "ink", "dash": True},
                               {"col": 2, "label": "display reading", "color": "curve", "width": 2}],
                },
                {
                    "title": "Reading error (clipped to ±0.25 kg)", "unit": "kg", "range": [-0.25, 0.25],
                    "ticks": [-0.2, -0.1, 0.0, 0.1, 0.2], "height": 1.2, "clip": True,
                    "series": [{"col": 3, "label": "error", "color": "output", "width": 1.5}],
                    "bands": {"*": [{"lo": -SETTLE_BAND, "hi": SETTLE_BAND, "label": "±0.05 kg settle band"}]},
                    "hlines": [{"y": 0.0, "color": "muted"}],
                },
            ],
        }
        if self.result is not None:
            out["result"] = self._result_view()
        return out

    def _result_view(self) -> dict[str, Any]:
        r, lim, s = self.result, self.limits, self.interference
        settle, noise, offset = max(r["settle_s"]), max(r["noise_kg"]), max(r["offset_kg"])
        reqs = [
            ("Settled within ±0.05 kg after each change (worst)", "never" if math.isinf(settle) else f"{settle:.2f} s",
             f"{lim['settle_s']:g} s", settle <= lim["settle_s"]),
            ("Reading noise, standard deviation (worst plateau)", f"{noise:.4f} kg", f"{lim['noise_kg']:g} kg",
             noise <= lim["noise_kg"]),
            ("Mean error (worst plateau)", f"{offset:.4f} kg", f"{ACCURACY:g} kg", offset <= ACCURACY),
        ]
        ref = reference_chain(s.fs, s.mains_hz)
        return {
            "requirements": [{"label": a, "measured": b, "limit": c, "ok": bool(d)} for a, b, c, d in reqs],
            "hidden": [
                ["Mains frequency", f"{s.mains_hz:.2f} Hz (drifting ±{DRIFT_HZ:g} Hz)"],
                ["Hum, 1st / 3rd / 5th", " / ".join(f"{a:.2f}" for a in s.hum_kg) + " kg"],
                ["Switching tone", f"{s.tone_hz:.0f} Hz, {s.tone_kg:.2f} kg"],
                ["White noise", f"{s.noise_kg:.3f} kg rms"],
                ["Sample rate", f"{s.fs} Hz"],
            ],
            "final": {
                "title": "Final chain",
                "columns": ["", "agent", "textbook reference"],
                "rows": [
                    ["Blocks", " -> ".join(_block_label(b) for b in self.bench.chain.spec) or "none",
                     " -> ".join(_block_label(b) for b in ref)],
                    ["Cost per ADC sample", f"{self.bench.chain.cost:.2f}", f"{Chain(ref, s.fs).cost:.2f}"],
                    ["Output rate", f"{self.bench.chain.out_rate:g} Hz", f"{Chain(ref, s.fs).out_rate:g} Hz"],
                ],
            },
        }


def _source(args: dict[str, Any]) -> str:
    src = args.get("source")
    if src not in ("adc", "output"):
        raise ToolError("source must be adc or output")
    return src


def _block_label(b: dict[str, Any]) -> str:
    k = b["type"]
    if k == "moving_average":
        return f"MA({b['n']})"
    if k == "notch":
        return f"notch({b['f0_hz']:g} Hz, Q {b['q']:g})"
    if k == "lowpass":
        return f"{b['kind']}{b['order']}({b['fc_hz']:g} Hz)"
    if k == "fir_lowpass":
        return f"FIR{b['taps']}({b['fc_hz']:g} Hz, {b['window']})"
    return f"decimate({b['factor']})"


# Scripted agents. They work only through the tools, like a model, and narrate.


def _mains_from(peaks: list[list[float]]) -> float:
    low = [p for p in peaks if 30.0 <= p[0] <= 100.0]
    return max(low, key=lambda p: p[1])[0]


def reference_agent(ep: Any) -> None:
    env = ep.env
    fs = env.interference.fs
    ep.model_turn("A quick look at the raw ADC signal on the scope first, then a long Hann-windowed spectrum to see what "
                  "interference there is and where.")
    ep.tool_call("scope", {"ms": 100, "source": "adc"})
    sp = ep.tool_call("spectrum", {"points": 16384, "window": "hann", "averages": 2, "source": "adc"})
    mains = _mains_from(sp["peaks"])
    top = max(sp["peaks"], key=lambda p: p[1])
    n = round(fs / mains)
    blocks = reference_chain(fs, mains)
    fc = blocks[-1]["fc_hz"]
    ep.model_turn(
        f"Mains hum at about {mains:g} Hz with odd harmonics, a strong tone at {top[0]:g} Hz and a white noise floor. "
        f"Two moving averages over one mains period (n = {n}) put nulls on the hum and all its harmonics and act as "
        f"the anti-aliasing filter, so decimating by {n} after them is safe. At the {fs / n:.1f} Hz output a "
        f"second-order Butterworth at {fc:g} Hz trims the remaining noise without making the reading slow. "
        "The moving averages cost 2 each and the lowpass runs at the low rate, so the chain is far under budget."
    )
    ep.tool_call("set_chain", {"blocks": blocks})
    ep.model_turn("Check on the bench: 20 kg on, watch the display, then empty the platform.")
    ep.tool_call("set_load", {"kg": 20})
    ep.tool_call("run", {"seconds": 4})
    ep.tool_call("scope", {"ms": 1000, "source": "output"})
    ep.tool_call("set_load", {"kg": 0})
    ep.tool_call("run", {"seconds": 3})
    ep.model_turn("The reading settles quickly and is quiet. A spectrum of the chain output confirms nothing is left of the hum or the tone.")
    ep.tool_call("spectrum", {"points": 1024, "window": "hann", "averages": 1, "source": "output"})
    ep.tool_call("submit", {})


def no_chain_agent(ep: Any) -> None:
    ep.model_turn("The scale already shows a reading. I submit without a processing chain.")
    ep.tool_call("submit", {})


def lowpass_agent(ep: Any) -> None:
    ep.model_turn("The display is jumpy. A second-order Butterworth lowpass at 5 Hz should smooth it and still respond fast.")
    ep.tool_call("set_chain", {"blocks": [{"type": "lowpass", "kind": "butter", "order": 2, "fc_hz": 5.0}]})
    ep.tool_call("set_load", {"kg": 20})
    ep.tool_call("run", {"seconds": 4})
    ep.model_turn("Much calmer than before. Submitting.")
    ep.tool_call("submit", {})


def decimate_first_agent(ep: Any) -> None:
    fs = ep.env.interference.fs
    f = fs // 50
    ep.model_turn(f"The display only needs 10 readings a second, so I drop to 50 Hz first (decimate by {f}) and filter "
                  "there, which costs almost nothing.")
    ep.tool_call("set_chain", {"blocks": [{"type": "decimate", "factor": f},
                                          {"type": "lowpass", "kind": "butter", "order": 2, "fc_hz": 5.0}]})
    ep.tool_call("set_load", {"kg": 20})
    ep.tool_call("run", {"seconds": 4})
    ep.model_turn("Cheap and smooth enough. Submitting.")
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "no_chain": no_chain_agent, "lowpass": lowpass_agent,
          "decimate_first": decimate_first_agent}
NEGATIVE_AGENTS = ("no_chain", "lowpass", "decimate_first")
