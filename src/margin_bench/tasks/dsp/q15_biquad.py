"""dsp.q15_biquad: a fixed-point biquad lowpass with exact integer semantics.

The model gets a target (RBJ cookbook lowpass: f0, Q, sampling rate), an input
bound, and the exact integer arithmetic of a Direct Form I biquad with int16
coefficients, an int32 accumulator and a rounding right shift. It answers with
the integer coefficients and the shift.

The grader checks three things: the magnitude response of the integer filter
against the target, the DC gain, and overflow, by running the integer
recurrence on test inputs that include the worst case for the output.

Difficulty rule: f0 is low (0.8 % to 8 % of the sampling rate), where a1 is
below -1 and does not fit Q15, and where plain rounding of the coefficients
can leave a DC gain error. The reference uses Q14 (shift 14), rounds, and
corrects b1 so the integer DC gain is exactly 1.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.tasks.rng import SeededRandom

SAMPLE_RATES_HZ = (8000, 16000, 44100, 48000)
QS = (0.5, 0.7071, 1.0, 1.5)
MAG_TOL = 0.02
DC_TOL_DB = 0.02
INPUT_HEADROOM = 0.9
SIM_LEN = 1024
GRID_POINTS = 2048
INT16_MIN, INT16_MAX = -32768, 32767
INT32_MAX = 2**31 - 1
KEYS = ("b0", "b1", "b2", "a1", "a2")

PROMPT = """\
Implement a second-order lowpass filter in fixed point.

Target: the RBJ Audio EQ Cookbook lowpass with f0 = {f0:g} Hz and Q = {q:g}, at a sampling rate of {fs:g} Hz.

Arithmetic (Direct Form I), executed exactly as written for every sample n:
  acc  = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
  y[n] = (acc + (1 << (shift - 1))) >> shift      (for shift = 0: y[n] = acc)
- x and y are int16. b0, b1, b2, a1, a2 are int16 integers (-32768 to 32767). shift is an integer from 0 to 15.
- acc is int32: every partial sum, accumulated left to right, must stay within the int32 range.
- >> is an arithmetic right shift (rounds toward minus infinity).
- y[n] must fit in int16 without saturation or wraparound.
- Inputs are bounded: |x[n]| <= {a_in}. Initial state is zero.

Requirements:
- the magnitude response of the integer filter, H(z) = (b0 + b1 z^-1 + b2 z^-2) / (2^shift + a1 z^-1 + a2 z^-2) \
(rounding ignored), must stay within {mag_tol:g} (absolute, linear scale) of the target magnitude \
at every frequency from 0 to {nyq:g} Hz;
- the DC gain must be within {dc_tol:g} dB of 0 dB;
- no overflow for any input within the bound;
- the filter must be stable.

Give the integers as a JSON object at the end of your answer, in a ```json code block:

```json
{{"b0": ..., "b1": ..., "b2": ..., "a1": ..., "a2": ..., "shift": ...}}
```
"""


def rbj_lowpass(f0: float, fs: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * q)
    c = math.cos(w0)
    b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


def _target(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return rbj_lowpass(spec["f0_hz"], spec["sample_rate_hz"], spec["q"])


def _ba(c: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    scale = float(1 << c["shift"])
    return np.array([c["b0"], c["b1"], c["b2"]]) / scale, np.array([1.0, c["a1"] / scale, c["a2"] / scale])


def _run(c: dict[str, int], x: list[int]) -> tuple[list[int], int, int]:
    """Exact integer recurrence with unbounded ints. Returns y, peak |y|, peak |partial acc|."""
    b0, b1, b2, a1, a2, sh = (c[k] for k in (*KEYS, "shift"))
    rnd = (1 << (sh - 1)) if sh > 0 else 0
    x1 = x2 = y1 = y2 = 0
    peak_y = peak_acc = 0
    ys = []
    for xn in x:
        acc = b0 * xn
        p = abs(acc)
        acc += b1 * x1
        p = max(p, abs(acc))
        acc += b2 * x2
        p = max(p, abs(acc))
        acc -= a1 * y1
        p = max(p, abs(acc))
        acc -= a2 * y2
        p = max(p, abs(acc))
        yn = (acc + rnd) >> sh
        ys.append(yn)
        peak_acc = max(peak_acc, p)
        peak_y = max(peak_y, abs(yn))
        if peak_y > 2**40:  # diverging; no need to go on
            break
        x2, x1, y2, y1 = x1, xn, y1, yn
    return ys, peak_y, peak_acc


def _test_inputs(c: dict[str, int], a_in: int, fs: float) -> dict[str, list[int]]:
    b, a = _ba(c)
    h = signal.lfilter(b, a, np.r_[1.0, np.zeros(SIM_LEN - 1)])
    worst = [a_in if v >= 0 else -a_in for v in h[::-1]]  # maximizes y[SIM_LEN - 1]
    f, resp = signal.freqz(b, a, worN=512, fs=fs)
    f_peak = max(float(f[int(np.argmax(np.abs(resp)))]), fs / SIM_LEN)
    period = max(2, int(round(fs / f_peak)))
    square = [a_in if (n // (period // 2 or 1)) % 2 == 0 else -a_in for n in range(SIM_LEN)]
    return {
        "worst case": worst,
        "positive step": [a_in] * SIM_LEN,
        "negative step": [-a_in] * SIM_LEN,
        "square at peak gain": square,
    }


class Q15Biquad:
    id = "dsp.q15_biquad"
    domain = "dsp"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(500):
            fs = rng.choice(SAMPLE_RATES_HZ)
            q = rng.choice(QS)
            f0 = round(fs * rng.uniform(0.008, 0.08))
            b, a = rbj_lowpass(f0, fs, q)
            h = signal.lfilter(b, a, np.r_[1.0, np.zeros(8191)])
            a_in = int(INPUT_HEADROOM * INT16_MAX / np.sum(np.abs(h))) // 256 * 256
            spec = {"sample_rate_hz": float(fs), "f0_hz": float(f0), "q": q, "input_bound": a_in}
            prompt = PROMPT.format(
                f0=f0, q=q, fs=fs, a_in=a_in, mag_tol=MAG_TOL, nyq=fs / 2, dc_tol=DC_TOL_DB
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            if self.grade(task, self.reference(task), with_plot=False).passed and not self.grade(
                task, self.negative(task), with_plot=False
            ).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        """Q14, rounded, with b1 adjusted so the integer DC gain is exactly 1."""
        b, a = _target(task.spec)
        scale = 1 << 14
        b0, b1, b2 = (round(v * scale) for v in b)
        a1, a2 = round(a[1] * scale), round(a[2] * scale)
        b1 = scale + a1 + a2 - b0 - b2
        return {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2, "shift": 14}

    def negative(self, task: Task) -> dict[str, Any]:
        """Q15 with saturation: a1 is below -1 and gets clipped to -32768."""
        b, a = _target(task.spec)
        b0, b1, b2, a1, a2 = (_q15_saturated(v) for v in (*b, a[1], a[2]))
        return {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2, "shift": 15}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            c = {k: answers.integer(answer, k) for k in (*KEYS, "shift")}
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        s = task.spec
        fs = s["sample_rate_hz"]
        bad_range = [k for k in KEYS if not INT16_MIN <= c[k] <= INT16_MAX]
        if bad_range or not 0 <= c["shift"] <= 15:
            what = ", ".join(bad_range) if bad_range else "shift"
            return Score(score=0.0, passed=False, error=f"out of range: {what}", metrics=dict(c))

        b, a = _ba(c)
        stable = bool(np.all(np.abs(np.roots(a)) < 1.0))
        f = np.linspace(0.0, fs / 2, GRID_POINTS)
        bt, at = _target(s)
        mag_t = np.abs(signal.freqz(bt, at, worN=f, fs=fs)[1])
        mag_q = np.abs(signal.freqz(b, a, worN=f, fs=fs)[1])
        if not stable:
            return Score(
                score=0.0,
                passed=False,
                error="the integer filter is unstable",
                metrics={"stable": False},
                plot_data=_plot(s, f, mag_t, mag_q, None) if with_plot else {},
            )

        mag_err = float(np.max(np.abs(mag_q - mag_t)))
        dc_gain = abs(float(np.sum(b) / np.sum(a)))
        dc_db = 20 * math.log10(dc_gain) if dc_gain > 0 else -math.inf
        runs = {name: _run(c, x) for name, x in _test_inputs(c, s["input_bound"], fs).items()}
        peak_y = max(r[1] for r in runs.values())
        peak_acc = max(r[2] for r in runs.values())
        ratios = {
            "magnitude_ratio": scoring.at_most(mag_err, MAG_TOL),
            "dc_ratio": scoring.at_most(abs(dc_db), DC_TOL_DB) if math.isfinite(dc_db) else 0.0,
            "output_headroom_ratio": min(1.0, INT16_MAX / peak_y) if peak_y else 1.0,
            "accumulator_headroom_ratio": min(1.0, INT32_MAX / peak_acc) if peak_acc else 1.0,
        }
        metrics = {
            "stable": True,
            "max_magnitude_error": mag_err,
            "dc_gain_db": dc_db if math.isfinite(dc_db) else None,
            "peak_output": peak_y,
            "peak_accumulator": peak_acc,
            **ratios,
        }
        passed = all(v >= 1.0 for v in ratios.values())
        worst = runs["worst case"][0]
        plot_data = _plot(s, f, mag_t, mag_q, worst) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _q15_saturated(v: float) -> int:
    return max(INT16_MIN, min(INT16_MAX, round(v * 32768)))


def _plot(s, f, mag_t, mag_q, worst) -> dict[str, Any]:
    bad = np.abs(mag_q - mag_t) > MAG_TOL
    top = max(float(np.max(mag_t)), float(np.max(np.minimum(mag_q, 4.0)))) * 1.15
    panels = [
        plot.panel(
            "Magnitude response",
            "Frequency (Hz)",
            "|H| (linear)",
            [plot.series("Target", f, mag_t, role="target"), plot.series("Answer", f, np.minimum(mag_q, top))],
            x_range=(0.0, s["sample_rate_hz"] / 2),
            y_range=(0.0, top),
            envelope_obj=plot.envelope(f, mag_t - MAG_TOL, mag_t + MAG_TOL, f"target ± {MAG_TOL:g}"),
            markers=[plot.marker("x", s["f0_hz"], "f0")],
            violation_points=plot.violations(f[bad], np.minimum(mag_q[bad], top)),
        )
    ]
    if worst is not None:
        n = np.arange(len(worst))
        y = np.asarray(worst, dtype=float)
        bad_y = np.abs(y) > INT16_MAX
        lim = max(40000.0, float(np.max(np.abs(y))) * 1.1)
        panels.append(
            plot.panel(
                "Output for the worst-case input",
                "Sample n",
                "y[n]",
                [plot.series("Answer", n, np.clip(y, -lim, lim))],
                y_range=(-lim, lim),
                forbidden_list=[
                    plot.forbidden(0, len(worst) - 1, INT16_MAX, None, "int16 overflow"),
                    plot.forbidden(0, len(worst) - 1, None, INT16_MIN, "int16 overflow"),
                ],
                violation_points=plot.violations(n[bad_y], np.clip(y[bad_y], -lim, lim)),
            )
        )
    return plot.plot(*panels)
