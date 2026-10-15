"""dsp.ema_shift: an exponential moving average on bit shifts, as written in firmware.

    acc += ((x << F) - acc) >> k;   y = (acc + R) >> F

The model picks the shift k for a cutoff frequency, the number of fractional
bits F and whether the output is rounded, so that the steady-state error for a
constant input stays within a limit and nothing overflows int32 for the given
ADC resolution.

The classic mistake is F = 0: (x - acc) >> k is zero whenever 0 <= x - acc < 2^k,
so the output can stop up to 2^k - 1 LSB short of a constant input (a dead
band). Fractional bits fix it, but they eat into the int32 headroom.

Difficulty rule: the cutoff target is within 10 % of the cutoff of one shift
k, with a 20 % tolerance, so exactly one k qualifies. The reference uses that
k, rounds the output, and takes the smallest F that meets the error limit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.tasks.rng import SeededRandom

SAMPLE_RATES_HZ = (100, 1000, 8000, 50000)
ADC_BITS = (12, 16, 20, 24)
ERROR_LIMITS_LSB = (0, 1, 2, 4)
CUTOFF_TOL = 0.2
INT32_MAX = 2**31 - 1
MAX_K, MAX_F = 20, 24

PROMPT = """\
Configure an exponential moving average filter that runs on a microcontroller.

Input: signed {bits}-bit ADC samples, x[n] from {xmin} to {xmax}, at {fs:g} samples per second.
Filter, executed for every sample with a 32-bit signed accumulator acc (initially any value the filter can reach):
  acc  = acc + (((x[n] << F) - acc) >> k)
  y[n] = (acc + R) >> F,   where R = 1 << (F - 1) if the output is rounded and F > 0, otherwise R = 0
>> is an arithmetic right shift (rounds toward minus infinity). Every intermediate value \
(x[n] << F, the difference, acc, acc + R) must fit in int32.

Requirements:
- cutoff: the -3 dB frequency of the linearized filter H(z) = 2^-k / (1 - (1 - 2^-k) z^-1) \
must be within {tol_pct:g} % of {fc:g} Hz;
- steady-state error: for every constant input value, once the output stops changing, \
|x - y| must be at most {err} LSB, from any starting state;
- no int32 overflow for any input sequence within the ADC range.

Give your choice as a JSON object at the end of your answer, in a ```json code block:

```json
{{"k": ..., "frac_bits": ..., "round_output": true}}
```
"""


def cutoff_hz(k: int, fs: float) -> float:
    """-3 dB frequency of alpha / (1 - (1 - alpha) z^-1), alpha = 2^-k."""
    alpha = 2.0**-k
    c = 1 - alpha**2 / (2 * (1 - alpha))
    return math.acos(max(-1.0, c)) * fs / (2 * math.pi)


def _measured_cutoff(k: int, fs: float) -> float:
    """Measure the -3 dB point of the linearized filter on a frequency grid."""
    alpha = 2.0**-k
    f = np.geomspace(fs * 1e-7, fs / 2, 20000)
    _, h = signal.freqz([alpha], [1, -(1 - alpha)], worN=f, fs=fs)
    below = np.nonzero(np.abs(h) < 1 / math.sqrt(2))[0]
    return float(f[below[0]]) if len(below) else fs / 2


def _test_values(bits: int) -> list[int]:
    xmin, xmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    picks = [xmin, xmin + 1, -2, -1, 0, 1, 2, xmax - 1, xmax]
    step = (xmax - xmin) // 13
    picks += [xmin + 3 + i * step for i in range(13)]
    return sorted(set(picks))


def _steady_errors(k: int, f: int, rnd: bool, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Settle every test value from both rails; return values and worst |x - y| per value."""
    values = np.array(_test_values(bits), dtype=np.int64)
    xmin, xmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    x = np.concatenate([values, values])
    acc = np.concatenate([np.full(len(values), xmin << f), np.full(len(values), xmax << f)]).astype(np.int64)
    target = x << f
    for _ in range(64 * (2**k) + 64):
        step = (target - acc) >> k
        if not step.any():
            break
        acc = acc + step
    r = (1 << (f - 1)) if (rnd and f > 0) else 0
    y = (acc + r) >> f
    err = np.abs(x - y)
    return values, np.maximum(err[: len(values)], err[len(values) :])


def _peak_intermediate(k: int, f: int, rnd: bool, bits: int) -> int:
    """Rail-to-rail steps, held until settled: the largest |intermediate| seen."""
    xmin, xmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    r = (1 << (f - 1)) if (rnd and f > 0) else 0
    acc, peak = 0, 0
    hold = 40 * 2**k
    for xv in (xmin, xmax, xmin, xmax):
        xs = xv << f
        for _ in range(hold):
            diff = xs - acc
            acc += diff >> k
            peak = max(peak, abs(xs), abs(diff), abs(acc), abs(acc + r))
            if diff >> k == 0:
                break
    return peak


class EmaShift:
    id = "dsp.ema_shift"
    domain = "dsp"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(500):
            fs = rng.choice(SAMPLE_RATES_HZ)
            bits = rng.choice(ADC_BITS)
            k = rng.integer(3, 10)
            err = rng.choice(ERROR_LIMITS_LSB)
            fc = float(f"{cutoff_hz(k, fs) * rng.uniform(0.9, 1.1):.2g}")
            if sum(abs(cutoff_hz(j, fs) - fc) <= CUTOFF_TOL * fc for j in range(1, MAX_K + 1)) != 1:
                continue
            spec = {"sample_rate_hz": float(fs), "adc_bits": bits, "cutoff_hz": fc, "max_error_lsb": err, "k": k}
            spec["reference_frac_bits"] = self._min_frac_bits(spec)
            if spec["reference_frac_bits"] is None:
                continue
            prompt = PROMPT.format(
                bits=bits, xmin=-(2 ** (bits - 1)), xmax=2 ** (bits - 1) - 1, fs=fs,
                tol_pct=100 * CUTOFF_TOL, fc=fc, err=err,
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            if self.grade(task, self.reference(task), with_plot=False).passed and not self.grade(
                task, self.negative(task), with_plot=False
            ).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def _min_frac_bits(self, spec: dict[str, Any]) -> int | None:
        for f in range(0, 32 - spec["adc_bits"]):
            _, errs = _steady_errors(spec["k"], f, True, spec["adc_bits"])
            if errs.max() <= spec["max_error_lsb"]:
                return f
        return None

    def reference(self, task: Task) -> dict[str, Any]:
        return {"k": task.spec["k"], "frac_bits": task.spec["reference_frac_bits"], "round_output": True}

    def negative(self, task: Task) -> dict[str, Any]:
        """No fractional bits: the textbook dead band."""
        return {"k": task.spec["k"], "frac_bits": 0, "round_output": False}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            k = answers.integer(answer, "k")
            f = answers.integer(answer, "frac_bits")
            rnd = answers.boolean(answer, "round_output")
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        if not (1 <= k <= MAX_K and 0 <= f <= MAX_F):
            return Score(score=0.0, passed=False, error=f"k must be 1..{MAX_K} and frac_bits 0..{MAX_F}",
                         metrics={"k": k, "frac_bits": f})
        s = task.spec
        bits, fs = s["adc_bits"], s["sample_rate_hz"]
        fc = _measured_cutoff(k, fs)
        rel = abs(fc - s["cutoff_hz"]) / s["cutoff_hz"]
        values, errs = _steady_errors(k, f, rnd, bits)
        worst_err = int(errs.max())
        peak = _peak_intermediate(k, f, rnd, bits)
        ratios = {
            "cutoff_ratio": scoring.at_most(rel, CUTOFF_TOL),
            "error_ratio": scoring.at_most(worst_err + 1, s["max_error_lsb"] + 1),
            "headroom_ratio": min(1.0, INT32_MAX / peak) if peak else 1.0,
        }
        metrics = {
            "cutoff_hz": fc,
            "cutoff_error_rel": rel,
            "max_steady_error_lsb": worst_err,
            "peak_intermediate": peak,
            **ratios,
        }
        passed = all(v >= 1.0 for v in ratios.values())
        plot_data = _plot(s, k, f, rnd, values, errs) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _plot(s, k, f, rnd, values, errs) -> dict[str, Any]:
    fs, bits = s["sample_rate_hz"], s["adc_bits"]
    # Step response in seconds, from 0 to a quarter of full scale, through the integer filter.
    x_step = (2 ** (bits - 1)) // 4
    n = int(min(8 * 2**k, 20000))
    r = (1 << (f - 1)) if (rnd and f > 0) else 0
    acc, ys = 0, []
    for _ in range(n):
        acc += ((x_step << f) - acc) >> k
        ys.append((acc + r) >> f)
    t = np.arange(n) / fs
    tau = 1 / (2 * math.pi * s["cutoff_hz"])
    target = x_step * (1 - np.exp(-t / tau))
    step = plot.panel(
        "Step response",
        "Time (s)",
        "y (LSB)",
        [plot.series("First-order target", t, target, role="target"), plot.series("Answer", t, np.array(ys, float))],
        note=f"Step from 0 to {x_step} LSB.",
    )
    idx = np.arange(len(values))
    bad = errs > s["max_error_lsb"]
    top = max(float(errs.max()), s["max_error_lsb"]) * 1.2 + 1
    ss = plot.panel(
        "Steady-state error by input value",
        "Test input (index, low to high)",
        "|x - y| (LSB)",
        [plot.series("Answer", idx, errs.astype(float))],
        y_range=(0.0, top),
        forbidden_list=[plot.forbidden(0, len(values) - 1, s["max_error_lsb"], None, "error limit")],
        violation_points=plot.violations(idx[bad], errs[bad].astype(float)),
    )
    return plot.plot(step, ss)
