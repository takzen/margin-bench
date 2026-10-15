"""dsp.fir_lowpass_mask: design FIR lowpass taps that fit a magnitude mask.

The model gets a sampling rate, passband and stopband edges, a passband ripple
limit, a stopband attenuation requirement and a maximum number of taps, and
answers with the taps. The grader evaluates the magnitude response on a dense
grid over both bands.

Difficulty rule: the tap limit is Kaiser's length estimate for the tighter of
the two ripple requirements, times TAP_SLACK. With TAP_SLACK = 1.1 a Kaiser
window design with some engineering margin (beta for a few dB more than
required, cutoff placed inside the transition band) fits every instance,
a Kaiser design at exactly the required attenuation fits about two thirds of
them, and an equiripple design fits all of them with room to spare.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from margin_bench.core import Score, Task
from margin_bench.grading import scoring
from margin_bench.tasks.rng import SeededRandom

SAMPLE_RATES_HZ = (1000, 8000, 16000, 44100, 48000)
PASSBAND_RIPPLES_DB = (0.05, 0.1, 0.2, 0.5, 1.0)
STOPBAND_ATTENS_DB = (40, 50, 60, 70, 80)
TAP_SLACK = 1.1
MIN_TAPS, MAX_TAPS = 15, 101

# Points per band on the grading grid. Fixed, so grading is deterministic.
GRID_POINTS = 4096
# Floor for |H| before taking dB, so exact zeros do not produce -inf.
MAG_FLOOR = 1e-12
PLOT_POINTS = 1024
PLOT_DB_FLOOR = -150.0
MAX_PLOTTED_VIOLATIONS = 256

PROMPT = """\
Design a digital FIR lowpass filter that meets the specification below.

Sampling rate: {fs:g} Hz
Passband: 0 to {fp:g} Hz. The magnitude response must stay within {ap:g} dB of 0 dB (between -{ap:g} dB and +{ap:g} dB).
Stopband: {fst:g} to {nyq:g} Hz. The magnitude response must not exceed -{as_:g} dB.
Number of taps: at most {nmax}.

The transition band between {fp:g} Hz and {fst:g} Hz is unconstrained. Linear phase is not required. \
Coefficients must be real numbers and are used exactly as written, in double precision. \
The magnitude response is checked on a dense frequency grid covering both bands, including the band edges.

Give the impulse response h[0], h[1], ..., h[N-1] as a JSON object at the end of your answer, in a ```json code block:

```json
{{"taps": [h0, h1, ...]}}
```
"""


def _nice_step(x: float) -> float:
    """Smallest value of the form 1, 2 or 5 times a power of ten that is >= x."""
    e = 10.0 ** math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return m * e
    raise AssertionError("unreachable")


def _deltas(ap_db: float, as_db: float) -> tuple[float, float]:
    """Linear ripple limits. The lower edge of the +/-ap mask is the tighter one."""
    return 1.0 - 10.0 ** (-ap_db / 20.0), 10.0 ** (-as_db / 20.0)


def _kaiser_numtaps(atten_db: float, transition: float) -> int:
    """Kaiser's length estimate; `transition` is normalized to the sampling rate."""
    return math.ceil((atten_db - 7.95) / (14.36 * transition)) + 1


class FirLowpassMask:
    id = "dsp.fir_lowpass_mask"
    domain = "dsp"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(1000):
            fs = rng.choice(SAMPLE_RATES_HZ)
            ap = rng.choice(PASSBAND_RIPPLES_DB)
            as_ = rng.choice(STOPBAND_ATTENS_DB)
            target_taps = rng.integer(21, 81)
            fp_frac = rng.uniform(0.1, 0.5)  # passband edge as a fraction of Nyquist

            dp, ds = _deltas(ap, as_)
            atten = -20.0 * math.log10(min(dp, ds))
            transition = (atten - 7.95) / (14.36 * (target_taps - 1))
            step = _nice_step(fs / 1000)
            fp = round(fp_frac * fs / 2 / step) * step
            fst = round((fp_frac * fs / 2 + transition * fs) / step) * step
            if fp <= 0 or fst <= fp or fst > 0.9 * fs / 2:
                continue
            nmax = math.ceil(TAP_SLACK * _kaiser_numtaps(atten, (fst - fp) / fs))
            if not MIN_TAPS <= nmax <= MAX_TAPS:
                continue

            spec = {
                "sample_rate_hz": float(fs),
                "passband_edge_hz": float(fp),
                "stopband_edge_hz": float(fst),
                "passband_ripple_db": float(ap),
                "stopband_atten_db": float(as_),
                "max_taps": nmax,
            }
            prompt = PROMPT.format(fs=fs, fp=fp, fst=fst, nyq=fs / 2, ap=ap, as_=as_, nmax=nmax)
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            # Every instance must be provably solvable.
            if self.grade(task, self.reference(task)).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        """Parks-McClellan equiripple design at the tap limit."""
        s = task.spec
        dp, ds = _deltas(s["passband_ripple_db"], s["stopband_atten_db"])
        fs = s["sample_rate_hz"]
        taps = signal.remez(
            s["max_taps"],
            [0.0, s["passband_edge_hz"], s["stopband_edge_hz"], fs / 2],
            [1.0, 0.0],
            weight=[1.0 / dp, 1.0 / ds],
            fs=fs,
        )
        return {"taps": taps.tolist()}

    def negative(self, task: Task) -> dict[str, Any]:
        """Truncated sinc (rectangular window): stopband stuck near -21 dB."""
        s = task.spec
        cutoff = (s["passband_edge_hz"] + s["stopband_edge_hz"]) / 2
        taps = signal.firwin(s["max_taps"], cutoff, window="boxcar", fs=s["sample_rate_hz"])
        return {"taps": taps.tolist()}

    def grade(self, task: Task, answer: dict[str, Any]) -> Score:
        taps = answer.get("taps") if isinstance(answer, dict) else None
        if not isinstance(taps, list) or not taps:
            return Score.format_failure('expected "taps": a non-empty list of numbers')
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in taps):
            return Score.format_failure('"taps" must contain only numbers')
        h = np.asarray(taps, dtype=float)
        if not np.all(np.isfinite(h)):
            return Score.format_failure('"taps" must be finite')

        s = task.spec
        fs = s["sample_rate_hz"]
        fp, fst = s["passband_edge_hz"], s["stopband_edge_hz"]
        ap, as_ = s["passband_ripple_db"], s["stopband_atten_db"]
        nyq = fs / 2

        f_pass = np.linspace(0.0, fp, GRID_POINTS)
        f_stop = np.linspace(fst, nyq, GRID_POINTS)
        db_pass = _mag_db(h, f_pass, fs)
        db_stop = _mag_db(h, f_stop, fs)

        dev = float(np.max(np.abs(db_pass)))
        atten = float(-np.max(db_stop))
        too_many = len(h) > s["max_taps"]
        r_pass = scoring.at_most(dev, ap)
        r_stop = scoring.at_least(atten, as_)

        metrics = {
            "numtaps": len(h),
            "max_taps": s["max_taps"],
            "passband_dev_db": dev,
            "passband_margin_db": ap - dev,
            "stopband_atten_db": atten,
            "stopband_margin_db": atten - as_,
            "passband_ratio": r_pass,
            "stopband_ratio": r_stop,
        }

        bad_pass = np.abs(db_pass) > ap
        bad_stop = db_stop > -as_
        violations = np.concatenate(
            [
                np.column_stack([f_pass[bad_pass], db_pass[bad_pass]]),
                np.column_stack([f_stop[bad_stop], db_stop[bad_stop]]),
            ]
        )
        if len(violations) > MAX_PLOTTED_VIOLATIONS:
            idx = np.linspace(0, len(violations) - 1, MAX_PLOTTED_VIOLATIONS).round().astype(int)
            violations = violations[idx]

        f_plot = np.unique(np.concatenate([np.linspace(0.0, nyq, PLOT_POINTS + 1), [fp, fst]]))
        db_plot = np.maximum(_mag_db(h, f_plot, fs), PLOT_DB_FLOOR)
        plot_data = {
            "kind": "magnitude_mask",
            "x_label": "Frequency (Hz)",
            "y_label": "Magnitude (dB)",
            "curve": {"x": _rounded(f_plot), "y": _rounded(db_plot)},
            "mask": [
                {"label": "passband", "x": [0.0, fp], "lower": -ap, "upper": ap},
                {"label": "stopband", "x": [fst, nyq], "lower": None, "upper": -as_},
            ],
            "violations": [[round(float(x), 4), round(float(y), 4)] for x, y in violations],
        }

        if too_many:
            return Score(
                score=0.0,
                passed=False,
                error=f"{len(h)} taps exceeds the limit of {s['max_taps']}",
                metrics=metrics,
                plot_data=plot_data,
            )
        passed = dev <= ap and atten >= as_
        return Score(
            score=scoring.combine(r_pass, r_stop),
            passed=passed,
            metrics=metrics,
            plot_data=plot_data,
        )


def _mag_db(h: np.ndarray, f: np.ndarray, fs: float) -> np.ndarray:
    _, resp = signal.freqz(h, worN=f, fs=fs)
    return 20.0 * np.log10(np.maximum(np.abs(resp), MAG_FLOOR))


def _rounded(a: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in a]
