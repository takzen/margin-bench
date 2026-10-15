"""control.pid_step_spec: tune a PID for overshoot, settling time and control effort.

The plant is a stable third-order lag. The model answers with parallel PID
gains; the derivative filter time constant is fixed by the spec. The grader
closes the loop and simulates a unit reference step.

Difficulty rule: a reference PID is built by textbook pole cancellation
(series form, Ti = tau1, Td = tau2) with its gain set by bisection for half the
allowed overshoot. The settling-time limit is the reference's settling time
times 1.25 and the control-effort limit its peak control signal times 1.5,
both rounded up to two significant digits.
"""

from __future__ import annotations

import math
from typing import Any

import control as ct
import numpy as np

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.tasks.control.common import (
    is_stable,
    lag_plant,
    lag_plant_text,
    pid,
    settling_time,
    sig,
    sig_up,
)
from margin_bench.tasks.rng import SeededRandom

OVERSHOOT_LIMITS_PCT = (5.0, 10.0, 15.0, 20.0)
SETTLING_BAND = 0.02
MAX_SS_ERROR = 0.005
TS_SLACK = 1.25
U_SLACK = 1.5
SIM_POINTS = 6001

PROMPT = """\
Tune a PID controller for the plant below.

Plant: G(s) = {plant}   (time in seconds)
Controller: C(s) = Kp + Ki/s + Kd*s/(Tf*s + 1), with Tf = {tf:g} s fixed.
Loop: unity negative feedback, e = r - y, u = C(s) e, y = G(s) u.

For a unit step in the reference r at t = 0, with zero initial conditions, simulated from 0 to {t_end:g} s:
- overshoot of y at most {os_max:g} % (overshoot = (max y - 1) * 100 %),
- settling time at most {ts_max:g} s (the time after which |y - 1| stays within 2 %),
- peak control signal |u(t)| at most {u_max:g} (including the jump at t = 0),
- steady-state error |1 - y(inf)| at most {ess_pct:g} %,
- the closed loop must be stable.

Give the gains as a JSON object at the end of your answer, in a ```json code block:

```json
{{"kp": ..., "ki": ..., "kd": ...}}
```
"""


def _loops(spec: dict[str, Any], kp: float, ki: float, kd: float) -> tuple[ct.LTI, ct.LTI, ct.LTI]:
    g = lag_plant(spec["plant_gain"], spec["plant_taus"])
    c = pid(kp, ki, kd, spec["tf"])
    return ct.feedback(c * g, 1), ct.feedback(c, g), c * g


def _simulate(spec: dict[str, Any], kp: float, ki: float, kd: float, t_end: float, n: int = SIM_POINTS):
    y_sys, u_sys, _ = _loops(spec, kp, ki, kd)
    t = np.linspace(0.0, t_end, n)
    stable = is_stable(y_sys)
    if not stable:
        return t, None, None, False, y_sys
    y = ct.step_response(y_sys, t).outputs
    u = ct.step_response(u_sys, t).outputs
    return t, np.asarray(y), np.asarray(u), True, y_sys


def _series_to_parallel(kc: float, ti: float, td: float) -> tuple[float, float, float]:
    return kc * (1 + td / ti), kc / ti, kc * td


class PidStepSpec:
    id = "control.pid_step_spec"
    domain = "control"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(200):
            k = sig(rng.uniform(0.5, 5.0))
            tau1 = sig(10 ** rng.uniform(0.0, 1.3))
            tau2 = sig(tau1 * rng.uniform(0.1, 0.5))
            tau3 = sig(tau1 * rng.uniform(0.02, 0.08))
            if not tau1 > tau2 > tau3 > 0:
                continue
            tf = sig(tau3 / 5)
            os_max = rng.choice(OVERSHOOT_LIMITS_PCT)
            spec: dict[str, Any] = {
                "plant_gain": k,
                "plant_taus": [tau1, tau2, tau3],
                "tf": tf,
                "overshoot_max_pct": os_max,
                "ss_error_max": MAX_SS_ERROR,
            }
            gains = _tune_reference(spec, os_max / 2)
            if gains is None:
                continue
            t_probe = 60 * tau1
            t, y, u, stable, _ = _simulate(spec, *gains[:3], t_probe)
            ts = settling_time(t, y, 1.0, SETTLING_BAND)
            if not stable or not math.isfinite(ts):
                continue
            spec["settling_time_max"] = sig_up(TS_SLACK * ts)
            spec["u_max"] = sig_up(U_SLACK * float(np.max(np.abs(u))))
            spec["t_end"] = sig_up(3 * spec["settling_time_max"])
            spec["reference_series"] = {"kc": gains[3], "ti": tau1, "td": tau2}
            task = Task(family_id=self.id, seed=seed, prompt=self._prompt(spec), spec=spec)
            if self.grade(task, self.reference(task), with_plot=False).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def _prompt(self, spec: dict[str, Any]) -> str:
        return PROMPT.format(
            plant=lag_plant_text(spec["plant_gain"], spec["plant_taus"]),
            tf=spec["tf"],
            t_end=spec["t_end"],
            os_max=spec["overshoot_max_pct"],
            ts_max=spec["settling_time_max"],
            u_max=spec["u_max"],
            ess_pct=100 * spec["ss_error_max"],
        )

    def reference(self, task: Task) -> dict[str, Any]:
        r = task.spec["reference_series"]
        kp, ki, kd = _series_to_parallel(r["kc"], r["ti"], r["td"])
        return {"kp": kp, "ki": ki, "kd": kd}

    def negative(self, task: Task) -> dict[str, Any]:
        """Proportional only: no integral action, so a steady-state error remains."""
        return {"kp": self.reference(task)["kp"], "ki": 0.0, "kd": 0.0}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            kp, ki, kd = (answers.number(answer, key) for key in ("kp", "ki", "kd"))
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        s = task.spec
        t, y, u, stable, y_sys = _simulate(s, kp, ki, kd, s["t_end"])
        if not stable:
            return Score(
                score=0.0,
                passed=False,
                error="closed loop is unstable",
                metrics={"stable": False},
                plot_data={},
            )
        overshoot = max(0.0, (float(np.max(y)) - 1.0) * 100)
        ts = settling_time(t, y, 1.0, SETTLING_BAND)
        u_peak = float(np.max(np.abs(u)))
        ess = abs(1.0 - float(np.real(y_sys.dcgain())))
        ratios = {
            "overshoot_ratio": scoring.at_most(overshoot, s["overshoot_max_pct"]),
            "settling_ratio": scoring.at_most(ts, s["settling_time_max"]) if math.isfinite(ts) else 0.0,
            "effort_ratio": scoring.at_most(u_peak, s["u_max"]),
            "ss_error_ratio": scoring.at_most(ess, s["ss_error_max"]),
        }
        passed = (
            overshoot <= s["overshoot_max_pct"]
            and ts <= s["settling_time_max"]
            and u_peak <= s["u_max"]
            and ess <= s["ss_error_max"]
        )
        metrics = {
            "stable": True,
            "overshoot_pct": overshoot,
            "settling_time": ts if math.isfinite(ts) else None,
            "u_peak": u_peak,
            "ss_error": ess,
            **ratios,
        }
        plot_data = _plot(s, t, y, u) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _tune_reference(spec: dict[str, Any], os_target: float) -> tuple[float, float, float, float] | None:
    """Series PID with Ti = tau1, Td = tau2; bisect the gain for the target overshoot."""
    tau1, tau2, _ = spec["plant_taus"]
    t_end = 4 * tau1  # the overshoot peak comes well before this

    def overshoot(kc: float) -> float:
        _, y, _, stable, _ = _simulate(spec, *_series_to_parallel(kc, tau1, tau2), t_end, 1601)
        return math.inf if not stable else (float(np.max(y)) - 1.0) * 100

    lo, hi = 1e-4 / spec["plant_gain"], 1e4 / spec["plant_gain"]
    if overshoot(lo) > os_target:
        return None
    for _ in range(24):
        mid = math.sqrt(lo * hi)
        if overshoot(mid) <= os_target:
            lo = mid
        else:
            hi = mid
    kc = sig(lo, 4)
    return (*_series_to_parallel(kc, tau1, tau2), kc)


def _plot(s: dict[str, Any], t: np.ndarray, y: np.ndarray, u: np.ndarray) -> dict[str, Any]:
    os_line = 1 + s["overshoot_max_pct"] / 100
    ts_max, t_end = s["settling_time_max"], s["t_end"]
    bad_y = (y > os_line) | ((t >= ts_max) & (np.abs(y - 1) > SETTLING_BAND))
    y_top = max(os_line, float(np.max(y))) * 1.1
    step = plot.panel(
        "Step response",
        "Time (s)",
        "Output y",
        [plot.series("Answer", t, y)],
        x_range=(0.0, t_end),
        y_range=(min(0.0, float(np.min(y))), y_top),
        forbidden_list=[
            plot.forbidden(0.0, ts_max, os_line, None, "overshoot limit"),
            plot.forbidden(ts_max, t_end, 1 + SETTLING_BAND, None, "outside 2 % band"),
            plot.forbidden(ts_max, t_end, None, 1 - SETTLING_BAND, "outside 2 % band"),
        ],
        markers=[plot.marker("x", ts_max, "settling time limit")],
        violation_points=plot.violations(t[bad_y], y[bad_y]),
    )
    u_lim = s["u_max"]
    bad_u = np.abs(u) > u_lim
    u_abs = max(u_lim, float(np.max(np.abs(u)))) * 1.1
    effort = plot.panel(
        "Control signal",
        "Time (s)",
        "Control u",
        [plot.series("Answer", t, u)],
        x_range=(0.0, t_end),
        y_range=(min(-0.1 * u_abs, float(np.min(u)) * 1.1), u_abs),
        forbidden_list=[
            plot.forbidden(0.0, t_end, u_lim, None, "effort limit"),
            plot.forbidden(0.0, t_end, None, -u_lim, "effort limit"),
        ],
        violation_points=plot.violations(t[bad_u], u[bad_u]),
    )
    return plot.plot(step, effort)
