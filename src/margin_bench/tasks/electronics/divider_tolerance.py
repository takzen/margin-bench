"""electronics.divider_tolerance: a loaded divider that holds its window at every tolerance corner.

The model picks R1 and R2 from a given series. The supply has a tolerance, the
resistors have a tolerance, and the output drives a load (an ADC input, say)
that has a tolerance of its own. ngspice runs an operating point at each of the
sixteen corners (supply, R1, R2 and load each at either limit); every corner
must land inside the output window, and the supply current must stay under a
limit.

Difficulty rule: the window is 1.3 times the smallest worst-case spread any
pair in the series can reach under the current limit, rounded up. The current
limit keeps the divider's total resistance comparable to the load, and the
load's own spread keeps R2 from simply being made large, so the two pull
against each other. A design that ignores the load misses the window.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.sims.ngspice import Ngspice, fmt
from margin_bench.tasks.electronics.common import eng, in_series, values
from margin_bench.tasks.rng import SeededRandom

SUPPLIES_V = (3.3, 5.0, 12.0, 24.0)
SUPPLY_TOLS = (0.02, 0.05)
SERIES_TOL = {"E96": 0.01, "E24": 0.05}
LOADS_OHM = (10e3, 47e3, 100e3, 1e6)
LOAD_TOLS = (0.1, 0.2, 0.3)
R_RANGE = (100.0, 10e6)
WINDOW_SLACK = 1.3

PROMPT = """\
Design a resistive divider that holds its output window over all tolerances.

Circuit: supply -> R1 -> output; R2 from the output to ground; a load of {rl} ± {rl_tol:g} % from the output to ground.
Supply: {vin:g} V ± {vin_tol:g} %. Resistors: {series} series, ± {r_tol:g} %, between 100 Ω and 10 MΩ.

Requirements, checked by simulation at all sixteen corners (supply, R1, R2 and load each at either \
tolerance limit):
- output voltage between {vlo:g} V and {vhi:g} V at every corner;
- current drawn from the supply at most {imax} at every corner.

Give R1 and R2 in ohms as a JSON object at the end of your answer, in a ```json code block:

```json
{{"r1": ..., "r2": ...}}
```
"""


def _corners(s: dict[str, Any], r1: float, r2: float) -> list[tuple[float, float, float, float]]:
    vt, rt, lt = s["supply_tol"], SERIES_TOL[s["series"]], s["load_tol"]
    vin, rl = s["supply_v"], s["load_ohm"]
    return [
        (vin * (1 + a * vt), r1 * (1 + b * rt), r2 * (1 + c * rt), rl * (1 + d * lt))
        for a, b, c, d in product((-1, 1), repeat=4)
    ]


def simulate(s: dict[str, Any], r1: float, r2: float) -> tuple[np.ndarray, np.ndarray]:
    vouts, currents = [], []
    for vin, r1c, r2c, rlc in _corners(s, r1, r2):
        netlist = [
            "divider corner",
            f"V1 in 0 DC {fmt(vin)}",
            "Vs in a DC 0",
            f"R1 a out {fmt(r1c)}",
            f"R2 out 0 {fmt(r2c)}",
            f"RL out 0 {fmt(rlc)}",
            ".op",
            ".end",
        ]
        r = Ngspice.get().run(netlist, ["v(out)", "vs#branch"])
        vouts.append(float(r["v(out)"][0]))
        currents.append(abs(float(r["vs#branch"][0])))
    return np.array(vouts), np.array(currents)


def _analytic_search(s: dict[str, Any]) -> tuple[float, float, float] | None:
    """Pair with the smallest worst-case deviation from the window center, under the current limit."""
    rs = values(s["series"], *R_RANGE)
    r1, r2 = np.meshgrid(rs, rs, indexing="ij")
    r1, r2 = r1.ravel(), r2.ravel()
    vt, rt, lt, vin = s["supply_tol"], SERIES_TOL[s["series"]], s["load_tol"], s["supply_v"]
    vmin = np.full(r1.shape, np.inf)
    vmax = np.full(r1.shape, -np.inf)
    imax = np.zeros(r1.shape)
    for a, b, c, d in product((-1, 1), repeat=4):
        v = vin * (1 + a * vt)
        r1c, r2c, rl = r1 * (1 + b * rt), r2 * (1 + c * rt), s["load_ohm"] * (1 + d * lt)
        lower = r2c * rl / (r2c + rl)
        vo = v * lower / (r1c + lower)
        vmin, vmax = np.minimum(vmin, vo), np.maximum(vmax, vo)
        imax = np.maximum(imax, v / (r1c + lower))
    center = s["center_v"]
    dev = np.maximum(vmax - center, center - vmin)
    dev[imax > s["current_max_a"] * (1 - 1e-6)] = np.inf  # keep clear of float ties with ngspice
    i = int(np.argmin(dev))
    if not np.isfinite(dev[i]):
        return None
    return float(r1[i]), float(r2[i]), float(dev[i])


class DividerTolerance:
    id = "electronics.divider_tolerance"
    domain = "electronics"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(300):
            vin = rng.choice(SUPPLIES_V)
            vt = rng.choice(SUPPLY_TOLS)
            series = rng.choice(("E96", "E24"))
            rl = rng.choice(LOADS_OHM)
            lt = rng.choice(LOAD_TOLS)
            center = float(f"{vin * rng.uniform(0.15, 0.6):.2g}")
            r_min_total = rl * rng.uniform(0.3, 2.0)
            imax = float(f"{vin * (1 + vt) / r_min_total:.2g}")
            s: dict[str, Any] = {
                "supply_v": vin, "supply_tol": vt, "series": series, "load_ohm": rl, "load_tol": lt,
                "center_v": center, "current_max_a": imax,
            }  # fmt: skip
            found = _analytic_search(s)
            if found is None:
                continue
            r1, r2, dev = found
            half = _round_up(WINDOW_SLACK * dev)
            s["window_v"] = [round(center - half, 6), round(center + half, 6)]
            if s["window_v"][0] <= 0:
                continue
            s["reference"] = {"r1": r1, "r2": r2}
            prompt = PROMPT.format(
                rl=eng(rl, "Ω"), rl_tol=100 * lt, vin=vin, vin_tol=100 * vt, series=series, r_tol=100 * SERIES_TOL[series],
                vlo=s["window_v"][0], vhi=s["window_v"][1], imax=eng(imax, "A"),
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=s)
            if self.grade(task, self.reference(task), with_plot=False).passed and not self.grade(
                task, self.negative(task), with_plot=False
            ).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        return dict(task.spec["reference"])

    def negative(self, task: Task) -> dict[str, Any]:
        """Ignores the load: right unloaded ratio, total resistance just above the current limit."""
        s = task.spec
        rs = values(s["series"], *R_RANGE)
        ratio = s["center_v"] / s["supply_v"]
        total_min = s["supply_v"] * (1 + s["supply_tol"]) / s["current_max_a"] * (1 + SERIES_TOL[s["series"]])
        best = None
        for r2 in rs:
            r1 = r2 * (1 - ratio) / ratio
            r1 = float(rs[np.argmin(np.abs(np.log(rs / r1)))])
            if r1 + r2 < total_min:
                continue
            key = (abs(r2 / (r1 + r2) - ratio) > 0.01 * ratio, r1 + r2)
            if best is None or key < best[0]:
                best = (key, r1, float(r2))
        return {"r1": best[1], "r2": best[2]}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            r1, r2 = answers.number(answer, "r1"), answers.number(answer, "r2")
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        s = task.spec
        problems = [
            f"{name} = {v:g} is not an {s['series']} value in range"
            for name, v in (("r1", r1), ("r2", r2))
            if not in_series(v, s["series"]) or not R_RANGE[0] * (1 - 1e-9) <= v <= R_RANGE[1] * (1 + 1e-9)
        ]
        if problems:
            return Score(score=0.0, passed=False, error="; ".join(problems), metrics={})
        vouts, currents = simulate(s, r1, r2)
        lo, hi = s["window_v"]
        excess = max(0.0, lo - float(vouts.min())) + max(0.0, float(vouts.max()) - hi)
        ratios = {
            "window_ratio": max(0.0, 1.0 - excess / (hi - lo)),
            "current_ratio": scoring.at_most(float(currents.max()), s["current_max_a"]),
        }
        metrics = {
            "vout_min": float(vouts.min()),
            "vout_max": float(vouts.max()),
            "current_max": float(currents.max()),
            **ratios,
        }
        passed = excess == 0.0 and float(currents.max()) <= s["current_max_a"]
        plot_data = _plot(s, vouts, currents) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _round_up(x: float) -> float:
    """Round up to two significant digits."""
    scale = 10.0 ** (math.floor(math.log10(x)) - 1)
    return float(f"{math.ceil(x / scale - 1e-9) * scale:.2g}")


def _plot(s: dict[str, Any], vouts: np.ndarray, currents: np.ndarray) -> dict[str, Any]:
    idx = np.arange(1, len(vouts) + 1)
    x_end = len(vouts) + 0.5
    lo, hi = s["window_v"]
    pad = (hi - lo) * 0.6
    bad_v = (vouts < lo) | (vouts > hi)
    v_panel = plot.panel(
        "Output voltage at each corner",
        "Tolerance corner",
        "Vout (V)",
        [plot.series("Answer (ngspice)", idx, vouts)],
        x_range=(0.5, x_end),
        y_range=(min(lo - pad, float(vouts.min()) - pad / 3), max(hi + pad, float(vouts.max()) + pad / 3)),
        forbidden_list=[plot.forbidden(0.5, x_end, hi, None, "above window"), plot.forbidden(0.5, x_end, None, lo, "below window")],
        violation_points=plot.violations(idx[bad_v], vouts[bad_v]),
        note="Each corner puts the supply, R1, R2 and the load at one of their tolerance limits.",
    )
    imax = s["current_max_a"]
    bad_i = currents > imax
    i_panel = plot.panel(
        "Supply current at each corner",
        "Tolerance corner",
        "Current (A)",
        [plot.series("Answer (ngspice)", idx, currents)],
        x_range=(0.5, x_end),
        y_range=(0.0, max(imax, float(currents.max())) * 1.3),
        forbidden_list=[plot.forbidden(0.5, x_end, imax, None, "current limit")],
        violation_points=plot.violations(idx[bad_i], currents[bad_i]),
    )
    return plot.plot(v_panel, i_panel)
