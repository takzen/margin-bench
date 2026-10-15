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
from margin_bench.grading import plot, scoring
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
            if self.grade(task, self.reference(task), with_plot=False).passed:
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

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
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
        plot_data = _plot(h, s, dev, db_pass, db_stop, f_pass, f_stop, bad_pass, bad_stop) if with_plot else {}

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


def _plot(h, s, dev, db_pass, db_stop, f_pass, f_stop, bad_pass, bad_stop) -> dict[str, Any]:
    fs = s["sample_rate_hz"]
    fp, fst = s["passband_edge_hz"], s["stopband_edge_hz"]
    ap, as_ = s["passband_ripple_db"], s["stopband_atten_db"]
    nyq = fs / 2

    f_full = np.unique(np.concatenate([np.linspace(0.0, nyq, PLOT_POINTS + 1), [fp, fst]]))
    db_full = _mag_db(h, f_full, fs)
    y_bot = math.floor((-as_ - 40) / 20) * 20
    y_top = max(10.0, math.ceil((float(np.max(db_full)) + 3) / 10) * 10)
    full = plot.panel(
        "Magnitude response",
        "Frequency (Hz)",
        "Magnitude (dB)",
        [plot.series("Answer", f_full, np.maximum(db_full, y_bot))],
        x_range=(0.0, nyq),
        y_range=(y_bot, y_top),
        forbidden_list=[plot.forbidden(fst, nyq, -as_, None, "stopband")],
        markers=[plot.marker("x", fp, "passband edge"), plot.marker("x", fst, "stopband edge")],
        violation_points=plot.violations(
            np.concatenate([f_pass[bad_pass], f_stop[bad_stop]]),
            np.maximum(np.concatenate([db_pass[bad_pass], db_stop[bad_stop]]), y_bot),
        ),
        note="The passband limits are too thin to see at this scale; see the passband detail.",
    )

    x1 = fp + 0.12 * (fst - fp)
    span = max(2 * ap, min(1.15 * dev, 40.0))
    f_det = np.linspace(0.0, x1, 801)
    detail = plot.panel(
        "Passband detail",
        "Frequency (Hz)",
        "Magnitude (dB)",
        [plot.series("Answer", f_det, np.clip(_mag_db(h, f_det, fs), -span, span))],
        x_range=(0.0, x1),
        y_range=(-span, span),
        forbidden_list=[
            plot.forbidden(0.0, fp, ap, None, "above ripple limit"),
            plot.forbidden(0.0, fp, None, -ap, "below ripple limit"),
        ],
        markers=[plot.marker("x", fp, "passband edge")],
        violation_points=plot.violations(f_pass[bad_pass], np.clip(db_pass[bad_pass], -span, span)),
    )
    return plot.plot(full, detail)
