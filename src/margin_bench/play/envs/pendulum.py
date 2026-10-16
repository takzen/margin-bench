"""control.cart_pendulum: balance an inverted pendulum on a cart.

A cart on a 1 m track is pushed by a force-controlled drive; a uniform rod
hangs from a pivot on the cart. Rod mass and length are given (anyone can
weigh and measure a rod); the cart's mass and friction and the pivot friction
are hidden. Encoders measure the cart position and the rod angle; a state
feedback controller at 100 Hz computes the force from them, estimating the
velocities by differencing with an optional filter, and its output reaches
the drive one sample later.

The model commissions the balancing controller: it can push the cart by hand,
let the rod hang or set it upright and let go, tap the rod, set the four
gains and the reference position, and record the motion. The acceptance test
lets the rod go 3 degrees off upright, moves the cart 20 cm and back, then taps
the rod.

Difficulty rule: the limits are the worst result over a family of LQR designs
on the true linearized plant (Q weighting cart position and rod angle, R the
force), with R and the position weight scaled by 0.7 and 1.4, and a 5 ms
velocity filter; times get 20 % slack. With no gains the rod falls; balancing
the angle alone lets the cart wander off its reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import linalg

from margin_bench import scoring
from margin_bench.core import Score
from margin_bench.play.env import ToolError, tool
from margin_bench.rng import SeededRandom

G = 9.81
DT = 0.002  # s, integration step (RK4)
CTRL_EVERY = 5  # controller at 100 Hz
TS = DT * CTRL_EVERY
FRAME_EVERY = 5
TRACK = 0.5  # m, end stops at +/- TRACK
FALL_DEG = 30.0
X_COUNT = 5e-5  # m per cart encoder count
TH_COUNT = 2 * math.pi / 4096  # rad per rod encoder count
REF_LIMIT = 0.3
COMMISSION_BUDGET_S = 120.0
MAX_RUN_MS = 10000.0
MAX_ROWS = 300
MAX_CALLS = 40

T_REF1, T_REF2, T_TAP, T_END = 3.0, 7.0, 11.0, 17.0
MOVE = 0.2  # m
SETTLE_X = 0.01  # m
SETTLE_TH = math.radians(1.0)
FAMILY = ((1.0, 1.0), (0.7, 1.0), (1.4, 1.0), (1.0, 0.7), (1.0, 1.4))  # (R scale, position weight scale)
SLACK = 1.2
_LIMITS: dict[int, dict[str, float]] = {}


@dataclass(frozen=True)
class Rig:
    cart_kg: float
    rod_kg: float
    rod_m: float  # full rod length; the centre of mass sits at half of it
    cart_friction: float  # N per m/s
    pivot_friction: float  # N m per rad/s
    f_max: float  # N

    @property
    def lc(self) -> float:
        return self.rod_m / 2

    @property
    def i_pivot(self) -> float:
        return self.rod_kg * self.rod_m**2 / 3

    def linear(self) -> tuple[np.ndarray, np.ndarray]:
        """Linearized upright model, state [x, v, theta, omega], input force."""
        M, m, l, ip, b, c = self.cart_kg, self.rod_kg, self.lc, self.i_pivot, self.cart_friction, self.pivot_friction
        A2 = np.array([[M + m, m * l], [m * l, ip]])
        inv = np.linalg.inv(A2)
        # [xdd, thdd] = inv @ ([F - b v, m g l th - c w])
        a = np.zeros((4, 4))
        bvec = np.zeros(4)
        a[0, 1] = a[2, 3] = 1.0
        a[1, 1], a[3, 1] = inv[0, 0] * -b, inv[1, 0] * -b
        a[1, 2], a[3, 2] = inv[0, 1] * m * G * l, inv[1, 1] * m * G * l
        a[1, 3], a[3, 3] = inv[0, 1] * -c, inv[1, 1] * -c
        bvec[1], bvec[3] = inv[0, 0], inv[1, 0]
        return a, bvec.reshape(4, 1)


@dataclass
class Gains:
    k_x: float = 0.0
    k_v: float = 0.0
    k_theta: float = 0.0
    k_omega: float = 0.0
    filter_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"k_x": self.k_x, "k_v": self.k_v, "k_theta": self.k_theta, "k_omega": self.k_omega,
                "filter_ms": self.filter_ms}


class Sim:
    def __init__(self, rig: Rig, gains: Gains, noise_seed: str, *, upright: bool = False) -> None:
        self.r = rig
        self.g = replace(gains)
        self.k = 0
        self.s = np.array([0.0, 0.0, 0.0 if upright else math.pi, 0.0])  # x, v, theta (0 = up), omega
        self.enabled = False
        self.ref = 0.0
        self.manual = 0.0  # open-loop force while the controller is off
        self.f_next = 0.0
        self.force = 0.0
        self.x_prev = self.th_prev = None
        self.v_hat = self.w_hat = 0.0
        self.rng = SeededRandom(noise_seed)
        self.frames: list[list[Any]] = []
        self.incidents: list[dict[str, Any]] = []
        self._fallen = False
        self._stopped = False
        self._record()

    @property
    def t(self) -> float:
        return self.k * DT

    def theta_wrapped(self) -> float:
        return (self.s[2] + math.pi) % (2 * math.pi) - math.pi

    def _deriv(self, s: np.ndarray, f: float) -> np.ndarray:
        r = self.r
        x, v, th, w = s
        m, l, ip = r.rod_kg, r.lc, r.i_pivot
        a11, a12 = r.cart_kg + m, m * l * math.cos(th)
        a22 = ip
        r1 = f - r.cart_friction * v + m * l * w * w * math.sin(th)
        r2 = m * G * l * math.sin(th) - r.pivot_friction * w
        det = a11 * a22 - a12 * a12
        xdd = (r1 * a22 - a12 * r2) / det
        thdd = (a11 * r2 - a12 * r1) / det
        return np.array([v, xdd, w, thdd])

    def _measure(self) -> tuple[float, float]:
        x = math.floor(self.s[0] / X_COUNT) * X_COUNT
        th = math.floor(self.theta_wrapped() / TH_COUNT) * TH_COUNT
        return x, th

    def _control(self) -> None:
        x, th = self._measure()
        if self.x_prev is None:
            self.x_prev, self.th_prev = x, th
        v_raw, w_raw = (x - self.x_prev) / TS, (th - self.th_prev) / TS
        self.x_prev, self.th_prev = x, th
        tf = self.g.filter_ms / 1000
        a = 1.0 if tf <= 0 else TS / (tf + TS)
        self.v_hat += a * (v_raw - self.v_hat)
        self.w_hat += a * (w_raw - self.w_hat)
        if self.enabled:
            g = self.g
            u = -(g.k_x * (x - self.ref) + g.k_v * self.v_hat + g.k_theta * th + g.k_omega * self.w_hat)
        else:
            u = self.manual
        self.force = self.f_next  # one sample of delay
        self.f_next = max(-self.r.f_max, min(self.r.f_max, u))

    def _record(self) -> None:
        self.frames.append([round(self.t, 3), round(float(self.s[0]), 5), round(math.degrees(self.theta_wrapped()), 3),
                            round(self.force, 3), round(self.ref, 4) if self.enabled else None])

    def advance(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            if self.k % CTRL_EVERY == 0:
                self._control()
            f, s = self.force, self.s
            k1 = self._deriv(s, f)
            k2 = self._deriv(s + DT / 2 * k1, f)
            k3 = self._deriv(s + DT / 2 * k2, f)
            k4 = self._deriv(s + DT * k3, f)
            self.s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if abs(self.s[0]) >= TRACK:  # end stop: the cart stops dead
                self.s[0] = math.copysign(TRACK, self.s[0])
                if not self._stopped:
                    self.incidents.append({"t_s": round(self.t, 3), "alarm": "end stop"})
                self._stopped = True
                self.s[1] = 0.0
            else:
                self._stopped = False
            self.k += 1
            fallen = abs(self.theta_wrapped()) > math.radians(FALL_DEG) and self.enabled
            if fallen and not self._fallen:
                self.incidents.append({"t_s": round(self.t, 3), "alarm": "rod fell"})
                self.enabled = False  # the controller gives up, as a real one would
                self.manual = 0.0
            self._fallen = fallen or (self._fallen and abs(self.theta_wrapped()) > math.radians(FALL_DEG))
            if self.k % FRAME_EVERY == 0:
                self._record()

    def place(self, theta: float) -> None:
        """Hold the rod at `theta` (0 = up) with the cart at rest, then let go."""
        self.s = np.array([self.s[0], 0.0, theta, 0.0])
        self.x_prev = self.th_prev = None
        self.v_hat = self.w_hat = 0.0

    def tap(self, impulse: float) -> None:
        """Horizontal impulse at the top of the rod (N s), applied to the rod's angular momentum."""
        r = self.r
        self.s[3] += impulse * r.rod_m * math.cos(self.s[2]) / r.i_pivot


FRAME_COLUMNS = ["t_s", "x_m", "theta_deg", "force_n", "reference_m"]


@dataclass(frozen=True)
class Scenario:
    tilt_deg: float
    tap_ns: float


def lqr(rig: Rig, r_scale: float = 1.0, q_scale: float = 1.0) -> Gains:
    a, b = rig.linear()
    q = np.diag([100.0 * q_scale, 0.0, 100.0, 0.0])
    r = np.array([[0.05 * r_scale]])
    p = linalg.solve_continuous_are(a, b, q, r)
    k = (np.linalg.inv(r) @ b.T @ p).ravel()
    return Gains(k_x=float(k[0]), k_v=float(k[1]), k_theta=float(k[2]), k_omega=float(k[3]), filter_ms=5.0)


def _settled_after(frames: list[list[Any]], t0: float, t1: float, ref: float) -> float:
    last_out = None
    for f in frames:
        if t0 < f[0] <= t1 and (abs(f[1] - ref) > SETTLE_X or abs(math.radians(f[2])) > SETTLE_TH):
            last_out = f[0]
    if last_out is None:
        return 0.0
    if last_out >= t1 - TS:
        return math.inf
    return last_out + TS - t0


def acceptance(rig: Rig, sc: Scenario, gains: Gains, noise_seed: str) -> dict[str, Any]:
    sim = Sim(rig, gains, noise_seed, upright=True)
    sim.enabled = True
    sim.place(math.radians(sc.tilt_deg))
    sim.advance(T_REF1)
    sim.ref = MOVE
    sim.advance(T_REF2 - T_REF1)
    sim.ref = 0.0
    sim.advance(T_TAP - T_REF2)
    if sim.enabled:
        sim.tap(sc.tap_ns)
    sim.advance(T_END - T_TAP)
    f = sim.frames
    return {
        "frames": f,
        "fell": any(i["alarm"] == "rod fell" for i in sim.incidents),
        "end_stop": any(i["alarm"] == "end stop" for i in sim.incidents),
        "catch_s": _settled_after(f, 0.0, T_REF1, 0.0),
        "move_s": max(_settled_after(f, T_REF1, T_REF2, MOVE), _settled_after(f, T_REF2, T_TAP, 0.0)),
        "tap_s": _settled_after(f, T_TAP, T_END, 0.0),
        "max_x_m": max(abs(r[1]) for r in f),
        "max_theta_deg": max(abs(r[2]) for r in f),
    }


PROMPT = """\
You are commissioning the balancing controller of an inverted pendulum on a cart. The rig is simulated; treat it \
as a real one.

Rig
- A cart on a straight track with end stops at x = -0.5 m and +0.5 m. A drive pushes the cart with the force you \
command, limited to +/-{fmax:g} N. Positive force pushes toward +x.
- On the cart, a pivot carries a uniform rod: {rod_kg:g} kg, {rod_m:g} m long (centre of mass at {lc:g} m from the \
pivot). theta is the rod angle from upright, positive when the top leans toward +x; it is measured from -180 to \
180 degrees.
- The cart's mass, its friction and the pivot friction are not given.
- Encoders: cart position in steps of 0.05 mm, rod angle in steps of 360/4096 degrees.
- Controller: at 100 Hz, force = -(k_x * (x - x_ref) + k_v * v + k_theta * theta + k_omega * omega), in SI units \
(m, m/s, rad, rad/s). v and omega are estimated as the change of the measured position and angle over each 10 ms \
sample, through an optional first-order filter (filter_ms). The force it computes is applied one sample later. If \
|theta| ever exceeds {fall:g} degrees while it is on, the controller switches itself off.
- With the controller off you can push the cart with a constant force.

Now: the rod hangs straight down, the cart rests at x = 0, the controller is off and all gains are zero.

Acceptance test. It runs after you call submit, with the gains and filter you leave:
1. The cart is at x = 0 with the reference at 0 and the controller on. The rod is let go at {tilt:g} degrees from upright.
2. At t = 3 s the reference steps to +0.2 m; at t = 7 s back to 0. At t = 11 s the top of the rod is tapped with \
{tap:g} N s toward +x. The test ends at t = 17 s.
Requirements:
- the rod never falls past {fall:g} degrees and the cart never reaches an end stop;
- after the release: |x| < 1 cm and |theta| < 1 degree within {catch} s, and staying so until t = 3 s;
- after each reference step: within 1 cm of the reference and |theta| < 1 degree within {move} s, and staying so;
- after the tap: back within 1 cm of 0 and 1 degree within {tap_s} s, and staying so.

You have {budget:g} s of rig time and at most {calls} tool calls. Work like a controls engineer: identify what you \
need, design the gains, try them on the rig, then call submit.
"""


class CartPendulum:
    id = "control.cart_pendulum"
    domain = "control"
    title = "Inverted pendulum on a cart"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        rod_m = round(rng.uniform(0.3, 0.8), 2)
        rod_kg = round(rng.uniform(0.1, 0.3), 3)
        cart = rng.uniform(0.6, 2.0)
        self.rig = Rig(cart_kg=cart, rod_kg=rod_kg, rod_m=rod_m, cart_friction=rng.uniform(0.5, 3.0),
                       pivot_friction=rng.uniform(1e-4, 2e-3), f_max=float(round(rng.uniform(10.0, 16.0))))
        # the tap kicks the rod to about 0.6 rad/s, whatever its size
        self.scenario = Scenario(tilt_deg=3.0, tap_ns=round(0.6 * rod_kg * rod_m / 3, 4))
        if seed not in _LIMITS:
            family = [acceptance(self.rig, self.scenario, lqr(self.rig, a, b), self._noise("acceptance")) for a, b in FAMILY]
            _LIMITS[seed] = {k: math.ceil(SLACK * 10 * max(f[k] for f in family)) / 10 for k in ("catch_s", "move_s", "tap_s")}
        self.limits = dict(_LIMITS[seed])
        self.sim = Sim(self.rig, Gains(), self._noise("commission"))
        self.calls = 0
        self._submitted = False
        self.result: dict[str, Any] | None = None

    def _noise(self, what: str) -> str:
        return f"{self.id}:{self.seed}:{what}"

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def sim_time(self) -> float:
        return self.sim.t

    def prompt(self) -> str:
        r, sc, lim = self.rig, self.scenario, self.limits
        return PROMPT.format(fmax=r.f_max, rod_kg=r.rod_kg, rod_m=r.rod_m, lc=r.lc, fall=FALL_DEG, tilt=sc.tilt_deg,
                             tap=sc.tap_ns, catch=f"{lim['catch_s']:g}", move=f"{lim['move_s']:g}",
                             tap_s=f"{lim['tap_s']:g}", budget=COMMISSION_BUDGET_S, calls=MAX_CALLS)

    def tools(self) -> list[dict[str, Any]]:
        num = {"type": "number"}
        return [
            tool("set_gains", "Set the state feedback gains and the velocity filter. Omitted ones keep their value.",
                 {"k_x": num, "k_v": num, "k_theta": num, "k_omega": num, "filter_ms": {**num, "minimum": 0, "maximum": 100}}),
            tool("set_controller", "Switch the balancing controller on or off.", {"on": {"type": "boolean"}}, ["on"]),
            tool("set_reference", "Cart reference position in m, within +/-0.3.", {"x_m": num}, ["x_m"]),
            tool("set_force", "Constant force on the cart in N while the controller is off.", {"newtons": num}, ["newtons"]),
            tool("hang_rod", "Bring the rod to hang straight down at rest and stop the cart where it is."),
            tool("release_upright", "Hold the rod upright at this angle in degrees (at most 5 off), cart at rest, and "
                 "let go at once.", {"angle_deg": {**num, "minimum": -5, "maximum": 5}}, ["angle_deg"]),
            tool("tap", "Tap the top of the rod with a horizontal impulse in N s (toward +x if positive).",
                 {"impulse_ns": {**num, "minimum": -0.5, "maximum": 0.5}}, ["impulse_ns"]),
            tool("run", f"Let the rig run for `ms` milliseconds and return measured position, angle and force, at most "
                 f"{MAX_ROWS} rows (every 10 ms at best).", {"ms": {**num, "minimum": 10, "maximum": MAX_RUN_MS}}, ["ms"]),
            tool("submit", "Finish commissioning. The acceptance test then runs with the current gains."),
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise ToolError("already submitted")
        self.calls += 1
        if self.calls > MAX_CALLS and name != "submit":
            raise ToolError(f"tool call limit ({MAX_CALLS}) reached; call submit")
        s = self.sim
        if name == "set_gains":
            unknown = set(args) - {"k_x", "k_v", "k_theta", "k_omega", "filter_ms"}
            if unknown:
                raise ToolError(f"unknown parameters: {', '.join(sorted(unknown))}")
            for key in ("k_x", "k_v", "k_theta", "k_omega"):
                if key in args:
                    setattr(s.g, key, _num(args, key, -1e5, 1e5))
            if "filter_ms" in args:
                s.g.filter_ms = _num(args, "filter_ms", 0.0, 100.0)
            return self._status()
        if name == "set_controller":
            on = args.get("on")
            if not isinstance(on, bool):
                raise ToolError("on must be true or false")
            s.enabled = on
            s.manual = 0.0
            return self._status()
        if name == "set_reference":
            s.ref = _num(args, "x_m", -REF_LIMIT, REF_LIMIT)
            return self._status()
        if name == "set_force":
            if s.enabled:
                raise ToolError("switch the controller off first")
            s.manual = _num(args, "newtons", -s.r.f_max, s.r.f_max)
            return self._status()
        if name == "hang_rod":
            s.enabled, s.manual = False, 0.0
            s.place(math.pi)
            return self._status()
        if name == "release_upright":
            s.place(math.radians(_num(args, "angle_deg", -5.0, 5.0)))
            return self._status()
        if name == "tap":
            s.tap(_num(args, "impulse_ns", -0.5, 0.5))
            return self._status()
        if name == "run":
            ms = _num(args, "ms", 10.0, MAX_RUN_MS)
            left = COMMISSION_BUDGET_S - s.t
            if left <= 1e-9:
                raise ToolError("rig time is used up; call submit")
            n_total = max(1, round(min(ms / 1000, left) / TS))
            n_sample = max(1, math.ceil(n_total / MAX_ROWS))
            n_inc = len(s.incidents)
            rows, done = [], 0
            while done < n_total:
                n = min(n_sample, n_total - done)
                s.advance(n * TS)
                done += n
                x, th = s._measure()
                rows.append([round(1000 * s.t), round(x, 5), round(math.degrees(th), 3), round(s.force, 3)])
            out = self._status()
            out["trend"] = {"columns": ["t_ms", "x_m", "theta_deg", "force_n"], "rows": rows}
            if len(s.incidents) > n_inc:
                out["alarms"] = s.incidents[n_inc:]
            return out
        if name == "submit":
            self._submitted = True
            return {"submitted": True, "t_ms": round(1000 * s.t)}
        raise ToolError(f"unknown tool {name!r}")

    def _status(self) -> dict[str, Any]:
        s = self.sim
        x, th = s._measure()
        return {"t_ms": round(1000 * s.t), "controller_on": s.enabled, "x_m": round(x, 5),
                "theta_deg": round(math.degrees(th), 3), "force_n": round(s.force, 3), "reference_m": s.ref,
                "manual_force_n": s.manual, "gains": s.g.to_dict(),
                "time_left_ms": round(1000 * (COMMISSION_BUDGET_S - s.t))}

    def finish(self) -> Score:
        if self.result is None:
            self.result = acceptance(self.rig, self.scenario, self.sim.g, self._noise("acceptance"))
        r, lim = self.result, self.limits
        ratios = {k.replace("_s", "_ratio"): scoring.at_most(r[k], lim[k]) for k in ("catch_s", "move_s", "tap_s")}
        metrics = {
            "submitted": self._submitted, "tool_calls": self.calls, "rig_time_ms": round(1000 * self.sim.t),
            "commissioning_events": [i["alarm"] for i in self.sim.incidents],
            **{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items() if k != "frames"},
            "final_gains": self.sim.g.to_dict(), **ratios,
        }
        if r["fell"] or r["end_stop"]:
            why = "the rod fell" if r["fell"] else "the cart hit an end stop"
            return Score(score=0.0, passed=False, error=f"{why} in the acceptance test", metrics=metrics)
        score = scoring.combine(*ratios.values())
        return Score(score=score, passed=score >= 1.0, metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        g = result.get("gains") or {}
        if name == "set_gains":
            return (f"k_x {g['k_x']:.4g}, k_v {g['k_v']:.4g}, k_theta {g['k_theta']:.4g}, k_omega {g['k_omega']:.4g}, "
                    f"filter {g['filter_ms']:g} ms"), False
        if name == "set_controller":
            return "controller on" if result["controller_on"] else "controller off", False
        if name == "set_reference":
            return f"reference {result['reference_m']:g} m", False
        if name == "set_force":
            return f"force {result['manual_force_n']:g} N", False
        if name == "hang_rod":
            return "rod hanging at rest", False
        if name == "release_upright":
            return f"rod let go at {args.get('angle_deg')} degrees from upright", False
        if name == "tap":
            return f"rod tapped with {args.get('impulse_ns')} N s", False
        if name == "run":
            rows = result["trend"]["rows"]
            xs, ths = [r_[1] for r_ in rows], [r_[2] for r_ in rows]
            text = (f"ran to {result['t_ms'] / 1000:.2f} s: x {min(xs):+.3f}..{max(xs):+.3f} m, theta {min(ths):+.1f}.."
                    f"{max(ths):+.1f} deg, {len(rows)} rows")
            alarms = result.get("alarms") or []
            for a in alarms:
                text += f"; {a['alarm'].upper()} at {a['t_s']:.2f} s"
            return text, any(a["alarm"] == "rod fell" or a["alarm"] == "end stop" for a in alarms)
        if name == "submit":
            return "handed over for the acceptance test", False
        return "", False

    def visual(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def view(self) -> dict[str, Any]:
        lim = self.limits
        phases: list[dict[str, Any]] = [
            {"key": "commissioning", "name": "Commissioning", "rows": self.sim.frames, "events": [], "marks": []}
        ]
        if self.result is not None:
            phases.append({
                "key": "acceptance", "name": "Acceptance test", "rows": self.result["frames"],
                "events": [{"t_s": 0.0, "label": "let go"}, {"t_s": T_REF1, "label": "x_ref 0.2 m"},
                           {"t_s": T_REF2, "label": "x_ref 0"}, {"t_s": T_TAP, "label": "tap"}],
                "marks": [{"t_s": lim["catch_s"], "label": "catch limit"},
                          {"t_s": T_REF1 + lim["move_s"], "label": "move limit"},
                          {"t_s": T_REF2 + lim["move_s"], "label": "move limit"},
                          {"t_s": T_TAP + lim["tap_s"], "label": "tap limit"}],
            })
        r = self.rig
        out: dict[str, Any] = {
            "kind": "pendulum",
            "columns": FRAME_COLUMNS,
            "phases": phases,
            "scene": {"track_m": TRACK, "rod_m": r.rod_m, "f_max": r.f_max, "fall_deg": FALL_DEG,
                      "final_gains": self.sim.g.to_dict()},
            "panels": [
                {"title": "Cart position", "unit": "m", "range": [-TRACK, TRACK], "ticks": [-0.5, -0.25, 0, 0.25, 0.5],
                 "height": 1.2,
                 "series": [{"col": 4, "label": "reference", "color": "ink", "dash": True},
                            {"col": 1, "label": "cart", "color": "curve", "width": 2}],
                 "hlines": [{"y": TRACK, "label": "end stops", "color": "critical"}, {"y": -TRACK, "color": "critical"}]},
                {"title": "Rod angle from upright (clipped to ±40°)", "unit": "°", "range": [-40.0, 40.0],
                 "ticks": [-30, -10, 0, 10, 30], "height": 1.2, "clip": True,
                 "series": [{"col": 2, "label": "rod angle", "color": "output", "width": 1.5}],
                 "bands": {"*": [{"lo": -1.0, "hi": 1.0, "label": "±1° band"}]},
                 "hlines": [{"y": FALL_DEG, "label": "fall limit", "color": "critical"}, {"y": -FALL_DEG, "color": "critical"}]},
                {"title": "Force", "unit": "N", "range": [-r.f_max, r.f_max], "ticks": [-r.f_max, 0, r.f_max], "height": 0.8,
                 "series": [{"col": 3, "label": "force", "color": "demand", "width": 1}]},
            ],
        }
        if self.result is not None:
            out["result"] = self._result_view()
        return out

    def _result_view(self) -> dict[str, Any]:
        res, lim, r = self.result, self.limits, self.rig
        sec = lambda v: "never" if math.isinf(v) else f"{v:.2f} s"  # noqa: E731
        events = [i["alarm"] for i in self.sim.incidents]
        reqs = [
            ("Rod stays up, cart clear of the end stops",
             "fell" if res["fell"] else "end stop" if res["end_stop"] else f"max {res['max_theta_deg']:.1f}°, {res['max_x_m']:.2f} m",
             f"{FALL_DEG:g}°, 0.5 m", not res["fell"] and not res["end_stop"]),
            ("Caught after the release", sec(res["catch_s"]), f"{lim['catch_s']:g} s", res["catch_s"] <= lim["catch_s"]),
            ("At the reference after each move (worst)", sec(res["move_s"]), f"{lim['move_s']:g} s", res["move_s"] <= lim["move_s"]),
            ("Recovered after the tap", sec(res["tap_s"]), f"{lim['tap_s']:g} s", res["tap_s"] <= lim["tap_s"]),
        ]
        g, ref = self.sim.g, lqr(r)
        return {
            "requirements": [{"label": a, "measured": b, "limit": c, "ok": bool(d)} for a, b, c, d in reqs],
            "hidden": [
                ["Cart mass", f"{r.cart_kg:.2f} kg"],
                ["Cart friction", f"{r.cart_friction:.2f} N/(m/s)"],
                ["Pivot friction", f"{r.pivot_friction:.1e} N m/(rad/s)"],
                ["Events while commissioning", ", ".join(events) if events else "none"],
            ],
            "final": {
                "title": "Final gains", "columns": ["", "agent", "LQR reference"],
                "rows": [[k, f"{getattr(g, k):.4g}", f"{getattr(ref, k):.4g}"]
                         for k in ("k_x", "k_v", "k_theta", "k_omega", "filter_ms")],
            },
        }


def _num(args: dict[str, Any], key: str, lo: float, hi: float) -> float:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise ToolError(f"{key} must be between {lo:g} and {hi:g}")
    return float(v)


# Scripted agents. They work only through the tools, like a model, and narrate.


def reference_agent(ep: Any) -> None:
    env = ep.env
    rod_kg, rod_m, sc = env.rig.rod_kg, env.rig.rod_m, env.scenario
    f_test = 2.0
    ep.model_turn("The rod's mass and length are given; the cart's mass is not. With the rod hanging and the controller "
                  f"off I push the cart with {f_test:g} N for 0.3 s and fit the motion of a mass with viscous friction.")
    ep.tool_call("set_force", {"newtons": f_test})
    rows = ep.tool_call("run", {"ms": 300})["trend"]["rows"]
    t0, x0 = rows[0][0] / 1000 - 0.01, rows[0][1]
    ts = np.array([r[0] / 1000 - t0 - 0.01 for r in rows])  # the force arrives one sample late
    xs = np.array([r[1] - x0 for r in rows])

    def push(t: np.ndarray, m_tot: float, b: float) -> np.ndarray:
        t = np.maximum(t, 0.0)
        return f_test / b * (t - m_tot / b * (1 - np.exp(-b * t / m_tot)))

    best = min(((float(np.sum((push(ts, mt, b) - xs) ** 2)), mt, b)
                for mt in np.linspace(0.3, 3.0, 136) for b in np.linspace(0.05, 5.0, 100)))
    m_tot, friction = best[1], best[2]
    a = f_test / m_tot
    cart = max(0.1, m_tot - rod_kg)
    ep.tool_call("set_force", {"newtons": -f_test})
    ep.tool_call("run", {"ms": 300})
    ep.tool_call("set_force", {"newtons": 0})
    ep.tool_call("hang_rod", {})
    est = Rig(cart_kg=cart, rod_kg=rod_kg, rod_m=rod_m, cart_friction=friction, pivot_friction=0.0, f_max=env.rig.f_max)
    g = lqr(est)
    ep.model_turn(
        f"Fitting x(t) for a mass with viscous friction: initial acceleration {a:.2f} m/s^2, so the cart is about "
        f"{cart:.2f} kg, friction about {friction:.2f} N/(m/s). Linearizing about upright (uniform rod, pivot "
        f"inertia m L^2 / 3) and solving LQR with Q = diag(100, 0, 100, 0), R = 0.05 gives K = [{g.k_x:.2f}, "
        f"{g.k_v:.2f}, {g.k_theta:.2f}, {g.k_omega:.2f}]. A 5 ms filter tames the encoder steps in the velocity estimates."
    )
    ep.tool_call("set_gains", {"k_x": round(g.k_x, 4), "k_v": round(g.k_v, 4), "k_theta": round(g.k_theta, 4),
                               "k_omega": round(g.k_omega, 4), "filter_ms": 5})
    ep.tool_call("set_reference", {"x_m": 0})
    ep.tool_call("set_controller", {"on": True})
    ep.tool_call("release_upright", {"angle_deg": 2})
    ep.tool_call("run", {"ms": 3000})
    ep.model_turn("Caught and centred. Now the same moves as the test: 20 cm out and back, then a tap.")
    ep.tool_call("set_reference", {"x_m": 0.2})
    ep.tool_call("run", {"ms": 4000})
    ep.tool_call("set_reference", {"x_m": 0})
    ep.tool_call("run", {"ms": 4000})
    ep.tool_call("tap", {"impulse_ns": sc.tap_ns})
    ep.tool_call("run", {"ms": 4000})
    ep.model_turn("It rides out the tap and comes back to the centre. Submitting.")
    ep.tool_call("submit", {})


def factory_agent(ep: Any) -> None:
    ep.model_turn("The controller is installed; I submit it as it is.")
    ep.tool_call("submit", {})


def angle_only_agent(ep: Any) -> None:
    g = lqr(ep.env.rig)
    ep.model_turn("Balancing is about the angle: a PD on the rod angle is enough. I leave the cart position out.")
    ep.tool_call("set_gains", {"k_x": 0, "k_v": 0, "k_theta": round(g.k_theta, 4), "k_omega": round(g.k_omega, 4),
                               "filter_ms": 5})
    ep.tool_call("submit", {})


def wrong_sign_agent(ep: Any) -> None:
    g = lqr(ep.env.rig)
    ep.model_turn("To bring the cart back to the reference it should push toward it, so the position gains get the "
                  "opposite sign of the angle gains... I flip k_x and k_v.")
    ep.tool_call("set_gains", {"k_x": round(-g.k_x, 4), "k_v": round(-g.k_v, 4), "k_theta": round(g.k_theta, 4),
                               "k_omega": round(g.k_omega, 4), "filter_ms": 5})
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "factory": factory_agent, "angle_only": angle_only_agent,
          "wrong_sign": wrong_sign_agent}
NEGATIVE_AGENTS = ("factory", "angle_only", "wrong_sign")
