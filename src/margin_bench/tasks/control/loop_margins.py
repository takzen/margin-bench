"""control.loop_margins: design a lead/lag compensator for phase margin, gain margin and bandwidth.

The plant is an integrator with two lags. The model answers with a
first-order compensator C(s) = K (s/z + 1)/(s/p + 1). The grader computes the
margins of L = C G and checks closed-loop stability.

Difficulty rule: the minimum crossover frequency is set so that, at 1.2 times
that frequency, the plant needs between 20 and 50 degrees of phase lead on top
of the required phase margin plus 5 degrees. A textbook single-lead design
(maximum phase lead placed at the crossover) then meets every requirement; a
pure gain does not.
"""

from __future__ import annotations

import math
from typing import Any

import control as ct
import numpy as np

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.tasks.control.common import is_stable, lag_plant, lag_plant_text, sig
from margin_bench.tasks.rng import SeededRandom

PHASE_MARGINS_DEG = (40.0, 45.0, 50.0, 55.0, 60.0)
GAIN_MARGINS_DB = (6.0, 8.0, 10.0)
CROSSOVER_FACTOR = 1.2
EXTRA_PHASE_DEG = 5.0

PROMPT = """\
Design a compensator for the plant below.

Plant: G(s) = {plant}   (time in seconds)
Compensator: C(s) = K (s/z + 1) / (s/p + 1), with z > 0 and p > 0 in rad/s.
Loop: unity negative feedback, loop transfer function L(s) = C(s) G(s).

Requirements on L:
- phase margin at least {pm:g} degrees,
- gain margin at least {gm:g} dB,
- gain crossover frequency (where |L(jw)| = 1) at least {wc:g} rad/s,
- the closed loop must be stable.

Give K, z and p as a JSON object at the end of your answer, in a ```json code block:

```json
{{"k": ..., "z": ..., "p": ...}}
```
"""


def _lag_phase(g: ct.LTI, w: float) -> float:
    resp = complex(g(1j * w))
    phase = math.degrees(math.atan2(resp.imag, resp.real))
    # For these plants the true phase lies in (-360, -90]; atan2 returns (-180, 180].
    while phase > -90.0 + 1e-9:
        phase -= 360.0
    return phase


def _lead(g: ct.LTI, wc: float, pm_req: float) -> tuple[float, float, float]:
    """Single lead with its maximum phase at wc and |C G(j wc)| = 1."""
    needed = pm_req + EXTRA_PHASE_DEG - (180.0 + _lag_phase(g, wc))
    if needed <= 0:
        z = p = wc
    else:
        sin_phi = math.sin(math.radians(needed))
        alpha = (1 - sin_phi) / (1 + sin_phi)
        z, p = wc * math.sqrt(alpha), wc / math.sqrt(alpha)
    c_shape = complex((1j * wc / z + 1) / (1j * wc / p + 1))
    k = 1.0 / abs(c_shape * complex(g(1j * wc)))
    return k, z, p


class LoopMargins:
    id = "control.loop_margins"
    domain = "control"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(200):
            gain = sig(rng.uniform(0.5, 20.0))
            tau1 = sig(10 ** rng.uniform(-1.0, 0.7))
            tau2 = sig(tau1 * rng.uniform(0.05, 0.4))
            if not tau1 > tau2 > 0:
                continue
            pm = rng.choice(PHASE_MARGINS_DEG)
            gm = rng.choice(GAIN_MARGINS_DB)
            lead_deg = rng.uniform(20.0, 50.0)
            g = lag_plant(gain, [tau1, tau2], integrator=True)
            # Plant phase wanted at the design crossover: -180 + pm + 5 - lead.
            target = -180.0 + pm + EXTRA_PHASE_DEG - lead_deg
            lo, hi = 1e-4 / tau1, 1e4 / tau2
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                if _lag_phase(g, mid) > target:
                    lo = mid
                else:
                    hi = mid
            wc_min = sig(lo / CROSSOVER_FACTOR)
            spec = {
                "plant_gain": gain,
                "plant_taus": [tau1, tau2],
                "phase_margin_min_deg": pm,
                "gain_margin_min_db": gm,
                "crossover_min_rad_s": wc_min,
            }
            prompt = PROMPT.format(
                plant=lag_plant_text(gain, [tau1, tau2], integrator=True), pm=pm, gm=gm, wc=wc_min
            )
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            ok = self.grade(task, self.reference(task), with_plot=False).passed
            if ok and not self.grade(task, self.negative(task), with_plot=False).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def _plant(self, spec: dict[str, Any]) -> ct.LTI:
        return lag_plant(spec["plant_gain"], spec["plant_taus"], integrator=True)

    def reference(self, task: Task) -> dict[str, Any]:
        s = task.spec
        k, z, p = _lead(self._plant(s), CROSSOVER_FACTOR * s["crossover_min_rad_s"], s["phase_margin_min_deg"])
        return {"k": k, "z": z, "p": p}

    def negative(self, task: Task) -> dict[str, Any]:
        """Pure gain that puts the crossover at the same frequency: not enough phase."""
        s = task.spec
        wc = CROSSOVER_FACTOR * s["crossover_min_rad_s"]
        k = 1.0 / abs(complex(self._plant(s)(1j * wc)))
        return {"k": k, "z": 1.0, "p": 1.0}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            k, z, p = (answers.number(answer, key) for key in ("k", "z", "p"))
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        s = task.spec
        if z <= 0 or p <= 0:
            return Score(score=0.0, passed=False, error="z and p must be positive", metrics={"k": k, "z": z, "p": p})
        g = self._plant(s)
        c = ct.tf([k / z, k], [1 / p, 1])
        loop = c * g
        stable = is_stable(ct.feedback(loop, 1))
        gm, pm, _, wpc, wgc, _ = ct.stability_margins(loop)
        gm_db = math.inf if not np.isfinite(gm) or gm <= 0 else 20 * math.log10(gm)
        pm = float(pm) if np.isfinite(pm) else -math.inf
        wgc = float(wgc) if np.isfinite(wgc) else 0.0
        metrics = {
            "stable": stable,
            "phase_margin_deg": pm if math.isfinite(pm) else None,
            "gain_margin_db": gm_db if math.isfinite(gm_db) else None,
            "crossover_rad_s": wgc,
            "phase_crossover_rad_s": float(wpc) if np.isfinite(wpc) else None,
        }
        plot_data = _plot(s, loop, wgc, float(wpc) if np.isfinite(wpc) else None) if with_plot else {}
        if not stable:
            return Score(score=0.0, passed=False, error="closed loop is unstable", metrics=metrics, plot_data=plot_data)
        ratios = {
            "phase_margin_ratio": scoring.at_least(pm, s["phase_margin_min_deg"]) if math.isfinite(pm) else 0.0,
            "gain_margin_ratio": scoring.at_least(gm_db, s["gain_margin_min_db"]) if math.isfinite(gm_db) else 1.0,
            "crossover_ratio": scoring.at_least(wgc, s["crossover_min_rad_s"]),
        }
        metrics.update(ratios)
        passed = all(r >= 1.0 for r in ratios.values())
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _plot(s: dict[str, Any], loop: ct.LTI, wgc: float, wpc: float | None) -> dict[str, Any]:
    w_min = s["crossover_min_rad_s"]
    w = np.logspace(math.log10(w_min) - 2, math.log10(w_min) + 2.5, 900)
    resp = np.array([complex(loop(1j * x)) for x in w])
    mag_db = 20 * np.log10(np.abs(resp))
    phase = np.degrees(np.unwrap(np.angle(resp)))  # starts near -90 deg: the integrator dominates
    markers = [plot.marker("x", w_min, "min crossover")]
    if wgc > 0:
        markers.append(plot.marker("x", wgc, "crossover"))
    if wpc is not None:
        markers.append(plot.marker("x", wpc, "phase crossover"))
    low = w <= w_min
    bad = low & (mag_db < 0)
    mag = plot.panel(
        "Loop gain |L(jw)|",
        "Frequency (rad/s)",
        "Magnitude (dB)",
        [plot.series("Answer", w, mag_db)],
        x_scale="log",
        forbidden_list=[plot.forbidden(float(w[0]), w_min, None, 0.0, "crossover below the minimum")],
        markers=markers,
        violation_points=plot.violations(w[bad], mag_db[bad]),
        note="Crossover must not happen below the minimum frequency.",
    )
    pm_line = -180.0 + s["phase_margin_min_deg"]
    ph = plot.panel(
        "Loop phase",
        "Frequency (rad/s)",
        "Phase (deg)",
        [plot.series("Answer", w, phase)],
        x_scale="log",
        markers=markers + [plot.marker("y", pm_line, "phase needed at crossover"), plot.marker("y", -180.0, "-180 deg")],
    )
    return plot.plot(mag, ph)
