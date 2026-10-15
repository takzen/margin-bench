"""control.discrete_pid: discretize a continuous PID at a given sample time.

The model gets a plant, a continuous PID with a filtered derivative and a
sample time, and answers with the coefficients of the discrete controller.
The grader closes the loop around the plant sampled with a zero-order hold
and compares the sampled step response with the continuous loop's response.

Difficulty rule: the tolerance is 1.25 times the worst error of three standard
discretizations (Tustin, backward Euler, forward Euler for the integral with
backward Euler for the derivative), rounded up to two significant digits, and
at least 0.01. Any of those methods passes; forgetting the sample time in the
integral term does not.
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

ZETA = 0.7
TOL_SLACK = 1.25
MIN_TOL = 0.01

PROMPT = """\
Discretize the continuous PID controller below for a digital implementation.

Plant: G(s) = {plant}   (time in seconds)
Continuous controller: C(s) = Kp + Ki/s + Kd*s/(Tf*s + 1), with
  Kp = {kp:g}, Ki = {ki:g}, Kd = {kd:g}, Tf = {tf:g} s.
Sample time: Ts = {ts:g} s.

The discrete controller runs once per sample as
  u[k] = -a1*u[k-1] - a2*u[k-2] + b0*e[k] + b1*e[k-1] + b2*e[k-2],
that is C(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2), with e[k] = r[k] - y[k].
The plant is driven through a zero-order hold and y is sampled at the same instants.

Requirement: for a unit step in r at t = 0, the sampled closed-loop output y[k] must stay within \
{tol:g} of the continuous closed loop's step response at t = k*Ts, for every sample from 0 to {t_end:g} s, \
and the discrete closed loop must be stable. Use 0 for coefficients you do not need.

Give the coefficients as a JSON object at the end of your answer, in a ```json code block:

```json
{{"b": [b0, b1, b2], "a": [a1, a2]}}
```
"""


def _discretize(spec: dict[str, Any], method: str, ki_scale: float = 1.0) -> dict[str, Any]:
    kp, ki, kd, tf, ts = spec["kp"], spec["ki"] * ki_scale, spec["kd"], spec["tf"], spec["ts"]
    if method == "tustin":
        cz = ct.sample_system(pid(kp, ki, kd, tf), ts, method="tustin")
        num, den = (np.atleast_1d(np.squeeze(v)) for v in ct.tfdata(cz))
    else:
        # Integral: backward Euler Ts z/(z-1) or forward Euler Ts/(z-1), both written in z^-1.
        if method == "backward":
            i_num, i_den = [ki * ts, 0.0], [1.0, -1.0]
        elif method == "forward_i":
            i_num, i_den = [0.0, ki * ts], [1.0, -1.0]
        else:
            raise ValueError(method)
        # Derivative with filter, backward Euler: Kd (1 - z^-1) / ((Tf + Ts) - Tf z^-1).
        d_num, d_den = [kd, -kd], [tf + ts, -tf]
        num = np.polyadd(np.polyadd(np.polymul([kp], np.polymul(i_den, d_den)), np.polymul(i_num, d_den)), np.polymul(d_num, i_den))
        den = np.polymul(i_den, d_den)
    num, den = np.asarray(num, float), np.asarray(den, float)
    num, den = num / den[0], den / den[0]
    num = np.pad(num, (3 - len(num), 0)) if len(num) < 3 else num
    den = np.pad(den, (0, 3 - len(den))) if len(den) < 3 else den
    return {"b": num.tolist(), "a": den[1:3].tolist()}


def _responses(spec: dict[str, Any], b: list[float], a: list[float]):
    g = lag_plant(spec["plant_gain"], spec["plant_taus"])
    ts, t_end = spec["ts"], spec["t_end"]
    n = int(round(t_end / ts)) + 1
    tk = np.arange(n) * ts
    cont = ct.feedback(pid(spec["kp"], spec["ki"], spec["kd"], spec["tf"]) * g, 1)
    y_c = np.asarray(ct.step_response(cont, tk).outputs)
    cz = ct.tf(b, [1.0, *a], ts)
    loop_d = ct.feedback(cz * ct.sample_system(g, ts, method="zoh"), 1)
    if not is_stable(loop_d):
        return tk, y_c, None, cont
    y_d = np.asarray(ct.step_response(loop_d, tk).outputs)
    return tk, y_c, y_d, cont


class DiscretePid:
    id = "control.discrete_pid"
    domain = "control"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(200):
            k = sig(rng.uniform(0.5, 5.0))
            tau1 = sig(10 ** rng.uniform(-0.5, 1.0))
            tau2 = sig(tau1 * rng.uniform(0.1, 0.4))
            if not tau1 > tau2 > 0:
                continue
            tf = sig(tau2 / 5)
            kc = tau1 / (k * 4 * ZETA**2 * tf)
            # Series PID with Ti = tau1, Td = tau2, written in parallel form.
            kp, ki, kd = sig(kc * (1 + tau2 / tau1), 3), sig(kc / tau1, 3), sig(kc * tau2, 3)
            ts = sig(tf * rng.uniform(0.2, 1.0))
            spec: dict[str, Any] = {
                "plant_gain": k,
                "plant_taus": [tau1, tau2],
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "tf": tf,
                "ts": ts,
            }
            g = lag_plant(k, [tau1, tau2])
            cont = ct.feedback(pid(kp, ki, kd, tf) * g, 1)
            if not is_stable(cont):
                continue
            t_probe = np.linspace(0, 40 * tau1, 8001)
            ts_c = settling_time(t_probe, np.asarray(ct.step_response(cont, t_probe).outputs), 1.0, 0.02)
            if not math.isfinite(ts_c):
                continue
            spec["t_end"] = sig_up(3 * ts_c)
            errors = []
            for method in ("tustin", "backward", "forward_i"):
                d = _discretize(spec, method)
                _, y_c, y_d, _ = _responses(spec, d["b"], d["a"])
                errors.append(math.inf if y_d is None else float(np.max(np.abs(y_d - y_c))))
            if not all(math.isfinite(e) for e in errors):
                continue
            spec["tolerance"] = max(MIN_TOL, sig_up(TOL_SLACK * max(errors)))
            prompt = PROMPT.format(
                plant=lag_plant_text(k, [tau1, tau2]), kp=kp, ki=ki, kd=kd, tf=tf, ts=ts,
                tol=spec["tolerance"], t_end=spec["t_end"],
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            if not self.grade(task, self.negative(task), with_plot=False).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        return _discretize(task.spec, "tustin")

    def negative(self, task: Task) -> dict[str, Any]:
        """Tustin, but the integral gain is not multiplied by the sample time."""
        return _discretize(task.spec, "tustin", ki_scale=1.0 / task.spec["ts"])

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            b = answers.number_list(answer, "b", 3, 3)
            a = answers.number_list(answer, "a", 2, 2)
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        s = task.spec
        tk, y_c, y_d, cont = _responses(s, b, a)
        if y_d is None:
            return Score(score=0.0, passed=False, error="discrete closed loop is unstable", metrics={"stable": False})
        err = float(np.max(np.abs(y_d - y_c)))
        ratio = scoring.at_most(err, s["tolerance"])
        metrics = {"stable": True, "max_error": err, "tolerance": s["tolerance"], "error_ratio": ratio}
        plot_data = _plot(s, tk, y_c, y_d, cont) if with_plot else {}
        return Score(score=ratio, passed=err <= s["tolerance"], metrics=metrics, plot_data=plot_data)


def _plot(s: dict[str, Any], tk: np.ndarray, y_c: np.ndarray, y_d: np.ndarray, cont: ct.LTI) -> dict[str, Any]:
    tol = s["tolerance"]
    t_dense = np.linspace(0, s["t_end"], 1200)
    y_dense = np.asarray(ct.step_response(cont, t_dense).outputs)
    bad = np.abs(y_d - y_c) > tol
    return plot.plot(
        plot.panel(
            "Sampled step response",
            "Time (s)",
            "Output y",
            [plot.series("Continuous loop", t_dense, y_dense, role="target"), plot.series("Answer", tk, y_d)],
            x_range=(0.0, s["t_end"]),
            envelope_obj=plot.envelope(tk, y_c - tol, y_c + tol, f"within {tol:g} of the continuous loop"),
            violation_points=plot.violations(tk[bad], y_d[bad]),
        )
    )
