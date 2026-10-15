"""electronics.sallen_key_e24: a unity-gain Sallen-Key lowpass from standard values.

The model picks R1 and R2 from the E24 series and C1 and C2 from E12 for a
target natural frequency f0 and quality factor Q. ngspice runs an AC sweep of
the circuit; the grader measures f0 where the phase crosses -90 degrees and Q
as the gain there (for this circuit, |H(j w0)| = Q).

Difficulty rule: the reference is the best E24/E12 combination found by
exhaustive search over the capacitor pairs; instances are kept only when it
lands within half of each tolerance. The negative is the equal-component
design (R1 = R2, C1 = C2), which always gives Q = 0.5.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from margin_bench.core import Score, Task
from margin_bench.grading import answers, plot, scoring
from margin_bench.sims.ngspice import Ngspice, fmt
from margin_bench.tasks.electronics.common import eng, in_series, nearest, values
from margin_bench.tasks.rng import SeededRandom

Q_TARGETS = (0.5412, 0.5774, 0.7071, 1.0, 1.3066)
F0_TOL = 0.02
Q_TOL = 0.03
R_RANGE = (1e3, 1e6)
C_RANGE = (100e-12, 1e-6)
OPAMP_GAIN = 1e6

PROMPT = """\
Design a unity-gain Sallen-Key lowpass filter from standard component values.

Circuit (ideal op-amp wired as a voltage follower):
  input -> R1 -> node A;  node A -> R2 -> node B;  C1 from node A to the output;  C2 from node B to ground;
  op-amp non-inverting input at node B, output fed back to its inverting input.

Target: natural frequency f0 = {f0} and quality factor Q = {q:g}.
Measured by AC simulation: f0 is where the phase of Vout/Vin crosses -90 degrees, and Q is |Vout/Vin| at that frequency.

Requirements:
- f0 within {f0_tol:g} % of the target, Q within {q_tol:g} % of the target;
- R1 and R2 from the E24 series, between 1 kΩ and 1 MΩ;
- C1 and C2 from the E12 series, between 100 pF and 1 µF.

Give the values in ohms and farads as a JSON object at the end of your answer, in a ```json code block:

```json
{{"r1": 10000, "r2": 22000, "c1": 2.2e-8, "c2": 4.7e-9}}
```
(those numbers only show the format)
"""


def analytic(r1: float, r2: float, c1: float, c2: float) -> tuple[float, float]:
    w0 = 1 / math.sqrt(r1 * r2 * c1 * c2)
    return w0 / (2 * math.pi), math.sqrt(r1 * r2 * c1 * c2) / (c2 * (r1 + r2))


def simulate(r1: float, r2: float, c1: float, c2: float, f_lo: float, f_hi: float) -> tuple[np.ndarray, np.ndarray]:
    netlist = [
        "sallen-key lowpass",
        "V1 in 0 DC 0 AC 1",
        f"R1 in a {fmt(r1)}",
        f"R2 a b {fmt(r2)}",
        f"C1 a out {fmt(c1)}",
        f"C2 b 0 {fmt(c2)}",
        f"E1 out 0 b out {fmt(OPAMP_GAIN)}",
        f".ac dec 400 {fmt(f_lo)} {fmt(f_hi)}",
        ".end",
    ]
    r = Ngspice.get().run(netlist, ["frequency", "v(out)"])
    return r["frequency"].real, r["v(out)"]


def measure(f: np.ndarray, h: np.ndarray) -> tuple[float | None, float | None]:
    """f0 at the -90 degree phase crossing (log-interpolated), Q = |H| there."""
    phase = np.degrees(np.unwrap(np.angle(h)))
    idx = np.nonzero((phase[:-1] > -90) & (phase[1:] <= -90))[0]
    if len(idx) == 0:
        return None, None
    i = int(idx[0])
    frac = (phase[i] + 90) / (phase[i] - phase[i + 1])
    f0 = float(math.exp(math.log(f[i]) + frac * (math.log(f[i + 1]) - math.log(f[i]))))
    q = float(np.abs(h[i]) + frac * (np.abs(h[i + 1]) - np.abs(h[i])))
    return f0, q


class SallenKeyE24:
    id = "electronics.sallen_key_e24"
    domain = "electronics"

    def generate(self, seed: int) -> Task:
        rng = SeededRandom(seed)
        for _ in range(200):
            f0 = float(f"{10 ** rng.uniform(math.log10(50), math.log10(20000)):.2g}")
            q = rng.choice(Q_TARGETS)
            ref = _search(f0, q)
            if ref is None or ref[0] > 0.5:
                continue
            spec = {"f0_hz": f0, "q": q, "reference": ref[1]}
            prompt = PROMPT.format(f0=eng(f0, "Hz"), q=q, f0_tol=100 * F0_TOL, q_tol=100 * Q_TOL)
            task = Task(family_id=self.id, seed=seed, prompt=prompt, spec=spec)
            if self.grade(task, self.reference(task), with_plot=False).passed and not self.grade(
                task, self.negative(task), with_plot=False
            ).passed:
                return task
        raise RuntimeError(f"no feasible instance for seed {seed}")

    def reference(self, task: Task) -> dict[str, Any]:
        return dict(task.spec["reference"])

    def negative(self, task: Task) -> dict[str, Any]:
        """Equal components: R1 = R2, C1 = C2. Textbook simple, but Q is stuck at 0.5."""
        f0 = task.spec["f0_hz"]
        c = nearest(10e-9 * 1000 / f0 if f0 > 100 else 100e-9, "E12")
        c = min(max(c, C_RANGE[0]), C_RANGE[1])
        r = nearest(1 / (2 * math.pi * f0 * c), "E24")
        return {"r1": r, "r2": r, "c1": c, "c2": c}

    def grade(self, task: Task, answer: dict[str, Any], with_plot: bool = True) -> Score:
        try:
            r1, r2, c1, c2 = (answers.number(answer, k) for k in ("r1", "r2", "c1", "c2"))
        except answers.AnswerError as e:
            return Score.format_failure(str(e))
        problems = []
        for name, v, series, (lo, hi) in (
            ("r1", r1, "E24", R_RANGE), ("r2", r2, "E24", R_RANGE),
            ("c1", c1, "E12", C_RANGE), ("c2", c2, "E12", C_RANGE),
        ):  # fmt: skip
            if not in_series(v, series) or not lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                problems.append(f"{name} = {v:g} is not an {series} value in range")
        if problems:
            return Score(score=0.0, passed=False, error="; ".join(problems), metrics={})
        s = task.spec
        f, h = simulate(r1, r2, c1, c2, s["f0_hz"] / 100, s["f0_hz"] * 100)
        f0, q = measure(f, h)
        if f0 is None:
            return Score(score=0.0, passed=False, error="no -90 degree phase crossing in the sweep", metrics={})
        f0_err = abs(f0 - s["f0_hz"]) / s["f0_hz"]
        q_err = abs(q - s["q"]) / s["q"]
        ratios = {"f0_ratio": scoring.at_most(f0_err, F0_TOL), "q_ratio": scoring.at_most(q_err, Q_TOL)}
        metrics = {"f0_hz": f0, "q": q, "f0_error_rel": f0_err, "q_error_rel": q_err, **ratios}
        passed = f0_err <= F0_TOL and q_err <= Q_TOL
        plot_data = _plot(s, f, h, f0) if with_plot else {}
        return Score(score=scoring.combine(*ratios.values()), passed=passed, metrics=metrics, plot_data=plot_data)


def _search(f0: float, q: float) -> tuple[float, dict[str, float]] | None:
    """Best E24/E12 design: for every capacitor pair, solve for R1, R2 and try nearby E24 values."""
    w0 = 2 * math.pi * f0
    caps = values("E12", *C_RANGE)
    rs = values("E24", *R_RANGE)
    best = None
    for c1 in caps:
        for c2 in caps:
            if c1 < 4 * q * q * c2:  # R1, R2 would be complex
                continue
            s_sum = 1 / (w0 * q * c2)
            prod = 1 / (w0 * w0 * c1 * c2)
            disc = s_sum * s_sum - 4 * prod
            root = math.sqrt(max(disc, 0.0))
            ideal = ((s_sum + root) / 2, (s_sum - root) / 2)
            for r1_ideal, r2_ideal in (ideal, ideal[::-1]):
                if r2_ideal <= 0:
                    continue
                for r1 in _neighbors(rs, r1_ideal):
                    for r2 in _neighbors(rs, r2_ideal):
                        fa, qa = analytic(r1, r2, c1, c2)
                        cost = max(abs(fa - f0) / f0 / F0_TOL, abs(qa - q) / q / Q_TOL)
                        if best is None or cost < best[0]:
                            best = (cost, {"r1": float(r1), "r2": float(r2), "c1": float(c1), "c2": float(c2)})
    return best


def _neighbors(series: np.ndarray, x: float) -> list[float]:
    i = int(np.searchsorted(series, x))
    return [float(series[j]) for j in (i - 1, i) if 0 <= j < len(series)]


def _plot(s: dict[str, Any], f: np.ndarray, h: np.ndarray, f0: float) -> dict[str, Any]:
    w = f / s["f0_hz"]
    target = 1 / (1 - w**2 + 1j * w / s["q"])
    mag = plot.panel(
        "Magnitude response",
        "Frequency (Hz)",
        "Gain (dB)",
        [
            plot.series("Target", f, 20 * np.log10(np.abs(target)), role="target"),
            plot.series("Answer (ngspice)", f, 20 * np.log10(np.abs(h))),
        ],
        x_scale="log",
        markers=[plot.marker("x", s["f0_hz"], "target f0"), plot.marker("x", f0, "measured f0")],
    )
    phase = plot.panel(
        "Phase",
        "Frequency (Hz)",
        "Phase (deg)",
        [
            plot.series("Target", f, np.degrees(np.unwrap(np.angle(target))), role="target"),
            plot.series("Answer (ngspice)", f, np.degrees(np.unwrap(np.angle(h)))),
        ],
        x_scale="log",
        markers=[plot.marker("y", -90.0, "-90 deg"), plot.marker("x", f0, "measured f0")],
    )
    return plot.plot(mag, phase)
