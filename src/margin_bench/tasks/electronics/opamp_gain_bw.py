"""electronics.opamp_gain_bw: reach a gain and a bandwidth with a finite-GBW op-amp.

The model cascades up to three non-inverting stages built from a single-pole
op-amp model and E96 resistors. ngspice runs an AC sweep; the grader measures
the low-frequency gain and the -3 dB bandwidth.

Difficulty rule: the bandwidth requirement is 85 % of what n equal stages
reach (n = 2 or 3) and more than 10 % above what n - 1 stages reach, so one
textbook stage with the whole gain is never enough. The gain tolerance (1 %)
also makes the finite open-loop gain matter at high stage gains.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from margin_bench.core import Score, Task
from margin_bench.grading import plot, scoring
from margin_bench.sims.ngspice import Ngspice, fmt
from margin_bench.tasks.electronics.common import eng, in_series, values
from margin_bench.tasks.rng import SeededRandom

A0 = 1e5
GBWS_HZ = (1e6, 3e6, 10e6)
GAIN_TOL = 0.01
BW_FACTOR = 0.85
MAX_STAGES = 3
R_RANGE = (100.0, 1e6)

PROMPT = """\
Design a voltage amplifier from non-inverting op-amp stages.

Op-amp model: single pole, open-loop DC gain {a0:g} (100 dB), gain-bandwidth product {gbw}, \
otherwise ideal (infinite input impedance, zero output impedance, no offset, no slew-rate limit).
Each stage is a non-inverting amplifier: Rf from the output to the inverting input, Rg from the inverting input \
to ground, ideal gain 1 + Rf/Rg. Stages are cascaded directly, the first stage driven by the input.

Requirements, measured by AC simulation of the whole chain:
- gain at 1 Hz within {tol:g} % of {g:g} V/V;
- -3 dB bandwidth (where the gain falls to 1/sqrt(2) of its 1 Hz value) at least {bw};
- 1 to {max_stages} stages; every resistor from the E96 series, between 100 Ω and 1 MΩ.

Give the stages in order as a JSON object at the end of your answer, in a ```json code block:

```json
{{"stages": [{{"rf": 10000, "rg": 1000}}]}}
```
(that example shows the format only)
"""


def _stage_gain(rf: float, rg: float) -> float:
    g = 1 + rf / rg
    return g / (1 + g / A0)


def _bw_equal_stages(gain: float, gbw: float, n: int) -> float:
    return gbw / gain ** (1 / n) * math.sqrt(2 ** (1 / n) - 1)


def simulate(stages: list[tuple[float, float]], gbw: float) -> tuple[np.ndarray, np.ndarray]:
    fp = gbw / A0
    netlist = [
        "op-amp chain",
        "V1 in 0 DC 0 AC 1",
        ".subckt opamp inp inn out",
        "G1 0 n1 inp inn 1",
        f"R1 n1 0 {fmt(A0)}",
        f"C1 n1 0 {fmt(1 / (2 * math.pi * fp * A0))}",
        "E1 out 0 n1 0 1",
        ".ends",
    ]
    node = "in"
    for i, (rf, rg) in enumerate(stages, 1):
        netlist += [
            f"X{i} {node} m{i} o{i} opamp",
            f"Rf{i} o{i} m{i} {fmt(rf)}",
            f"Rg{i} m{i} 0 {fmt(rg)}",
        ]
        node = f"o{i}"
    netlist += [f".ac dec 200 1 {fmt(10 * gbw)}", ".end"]
    r = Ngspice.get().run(netlist, ["frequency", f"v({node})"])
    return r["frequency"].real, np.abs(r[f"v({node})"])


def measure(f: np.ndarray, mag: np.ndarray) -> tuple[float, float | None]:
    g_dc = float(mag[0])
    below = np.nonzero(mag < g_dc / math.sqrt(2))[0]
    if len(below) == 0:
        return g_dc, None
    i = int(below[0])
    lo, hi = math.log(mag[i - 1]), math.log(mag[i])
    frac = (lo - math.log(g_dc / math.sqrt(2))) / (lo - hi)
    return g_dc, float(math.exp(math.log(f[i - 1]) + frac * (math.log(f[i]) - math.log(f[i - 1]))))


def _gain_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rg = values("E96", 1e3, 9.9e3)
    rf = values("E96", *R_RANGE)
    rf_g, rg_g = np.meshgrid(rf, rg)
    g = 1 + rf_g / rg_g
    return (g / (1 + g / A0)).ravel(), rf_g.ravel(), rg_g.ravel()


_TABLE = None


def _best_pair(target: float) -> tuple[float, float]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _gain_table()
    gains, rfs, rgs = _TABLE
    i = int(np.argmin(np.abs(np.log(gains / target))))
    return float(rfs[i]), float(rgs[i])


class OpampGainBw:
    id = "electronics.opamp_gain_bw"
    domain = "electronics"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(300):
            gain = float(f"{10 ** rng.uniform(math.log10(50), math.log10(2000)):.2g}")
            gbw = rng.choice(GBWS_HZ)
            n = rng.choice((2, 3))
            bw_min = float(f"{BW_FACTOR * _bw_equal_stages(gain, gbw, n):.2g}")
            if bw_min > BW_FACTOR * _bw_equal_stages(gain, gbw, n) * 1.001:
                bw_min = float(f"{bw_min * 0.95:.2g}")  # rounding went up; step down
            if _bw_equal_stages(gain, gbw, n - 1) * 1.1 > bw_min:
                continue
            spec = {"gain": gain, "gbw_hz": gbw, "bandwidth_min_hz": bw_min, "reference_stages": n}
            prompt = PROMPT.format(
                a0=A0, gbw=eng(gbw, "Hz"), tol=100 * GAIN_TOL, g=gain, bw=eng(bw_min, "Hz"), max_stages=MAX_STAGES
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            if self.grade(task, self.reference(task), with_plot=False).passed and not self.grade(
                task, self.negative(task), with_plot=False
            ).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        """n nearly equal stages; the last one corrects the accumulated gain error."""
        s = task.spec
        n = s["reference_stages"]
        stages, achieved = [], 1.0
        for i in range(n):
            target = s["gain"] ** (1 / n) if i < n - 1 else s["gain"] / achieved
            rf, rg = _best_pair(target)
            stages.append({"rf": rf, "rg": rg})
            achieved *= _stage_gain(rf, rg)
        return {"stages": stages}

    def negative(self, task: Task) -> dict[str, Any]:
        """One stage with the whole gain: the textbook circuit, short of bandwidth."""
        rf, rg = _best_pair(task.spec["gain"])
        return {"stages": [{"rf": rf, "rg": rg}]}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        stages = answer.get("stages") if isinstance(answer, dict) else None
        if not isinstance(stages, list) or not stages:
            return Score.format_failure('expected "stages": a non-empty list of {"rf": ..., "rg": ...}')
        parsed = []
        for st in stages:
            if not isinstance(st, dict) or not all(
                isinstance(st.get(k), (int, float)) and not isinstance(st.get(k), bool) and math.isfinite(st.get(k))
                for k in ("rf", "rg")
            ):
                return Score.format_failure('each stage must be {"rf": number, "rg": number}')
            parsed.append((float(st["rf"]), float(st["rg"])))
        problems = []
        if len(parsed) > MAX_STAGES:
            problems.append(f"{len(parsed)} stages; at most {MAX_STAGES} allowed")
        for i, (rf, rg) in enumerate(parsed, 1):
            for name, v in (("rf", rf), ("rg", rg)):
                if not in_series(v, "E96") or not R_RANGE[0] * (1 - 1e-9) <= v <= R_RANGE[1] * (1 + 1e-9):
                    problems.append(f"stage {i} {name} = {v:g} is not an E96 value in range")
        if problems:
            return Score(score=0.0, passed=False, error="; ".join(problems), metrics={"stages": len(parsed)})
        s = task.spec
        f, mag = simulate(parsed, s["gbw_hz"])
        g_dc, bw = measure(f, mag)
        gain_err = abs(g_dc - s["gain"]) / s["gain"]
        ratios = {
            "gain_ratio": scoring.at_most(gain_err, GAIN_TOL),
            "bandwidth_ratio": scoring.at_least(bw, s["bandwidth_min_hz"]) if bw is not None else 1.0,
        }
        metrics = {
            "stages": len(parsed),
            "gain": g_dc,
            "gain_error_rel": gain_err,
            "bandwidth_hz": bw,
            **ratios,
        }
        passed = all(v >= 1.0 for v in ratios.values())
        plot_data = _plot(s, f, mag, bw) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _plot(s: dict[str, Any], f: np.ndarray, mag: np.ndarray, bw: float | None) -> dict[str, Any]:
    db = 20 * np.log10(mag)
    g_db = 20 * math.log10(s["gain"])
    floor = g_db - 10 * math.log10(2)
    bad = (f <= s["bandwidth_min_hz"]) & (db < floor)
    markers = [plot.marker("x", s["bandwidth_min_hz"], "required bandwidth")]
    if bw is not None:
        markers.append(plot.marker("x", bw, "measured bandwidth"))
    return plot.plot(
        plot.panel(
            "Gain of the chain",
            "Frequency (Hz)",
            "Gain (dB)",
            [plot.series("Answer (ngspice)", f, db)],
            x_scale="log",
            y_range=(min(float(db.min()), g_db - 40), g_db + 6),
            forbidden_list=[plot.forbidden(float(f[0]), s["bandwidth_min_hz"], None, floor, "more than 3 dB down")],
            markers=markers + [plot.marker("y", g_db, "required gain")],
            note="The gain at 1 Hz must be within 1 % of the required gain; see the measurements.",
            violation_points=plot.violations(f[bad], db[bad]),
        )
    )
