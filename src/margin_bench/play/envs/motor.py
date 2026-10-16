"""control.dc_motor: commission the cascade speed drive of a DC motor.

A permanent-magnet DC motor drives a dynamometer. A bridge fed from a fixed
supply makes the armature voltage; an inner PI loop at 10 kHz controls the
current (with one sample of computation delay), an outer PI loop at 1 kHz
controls the speed from a 1024-line encoder and hands the inner loop a current
reference limited to the drive's rated current. Above 1.5 times that current
the drive trips. Resistance, inductance, torque constant and inertia come from
the seed and are hidden.

The model commissions the drive: it can apply a voltage or a current directly,
tune both loops, run in speed control, put a load torque on the dynamometer
and record the response at up to 10 kHz. A careless voltage on a motor at rest
draws the locked-rotor current and trips the drive.

The acceptance test starts from rest, steps the speed reference to 60 % of the
no-load speed (the start-up hits the current limit), then puts a load torque
on and takes it off again.

Difficulty rule: the limits are the worst result over a family of textbook
cascade tunings on the true motor: current PI by pole cancellation at a
crossover of 1/20 of the current loop's sample rate, speed PI at a tenth of
that with its integral zero a factor 4 lower and a speed filter, anti-windup
on, the speed gains at the corners and centre of 0.7..1.4. Times and speed
errors get 15 % slack (at least 5 ms and 5 rpm). The factory gains are far too
soft; a careless voltage on a motor at rest trips the drive while commissioning.
Without anti-windup the start-up of most motors overshoots badly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from margin_bench import scoring
from margin_bench.core import Score
from margin_bench.play.env import ToolError, tool
from margin_bench.rng import SeededRandom

TS = 1e-4  # s, current loop sample time and integration step
SPEED_EVERY = 10  # speed loop runs every 10th step (1 kHz)
FRAME_EVERY = 10  # one viewer frame per ms
COUNTS = 4096  # encoder counts per revolution (1024 lines, x4)
TRIP = 1.5  # trip current as a multiple of the rated current
BAND = 0.01  # speed band, fraction of the reference
RPM = 60.0 / (2 * math.pi)
COMMISSION_BUDGET_S = 30.0
MAX_RUN_MS = 3000.0
MAX_ROWS = 300
MAX_CALLS = 40
T_REF = 0.02  # s into the acceptance test

FAMILY = ((1.0, 1.0), (0.7, 0.7), (0.7, 1.4), (1.4, 0.7), (1.4, 1.4))
SLACK = 1.15
_LIMITS: dict[int, dict[str, float]] = {}


@dataclass(frozen=True)
class Motor:
    v_supply: float
    r: float  # ohm
    l: float  # H
    kt: float  # Nm/A, also V s/rad
    j: float  # kg m^2, motor and dynamometer together
    b: float  # Nm s/rad
    i_max: float  # A, rated current: the speed loop's output limit
    noise_a: float  # current sensor noise, A rms

    @property
    def no_load_rpm(self) -> float:
        return self.v_supply / self.kt * RPM


@dataclass
class Gains:
    cur_kp: float = 0.5  # V/A
    cur_ki: float = 50.0  # V/(A s)
    spd_kp: float = 0.002  # A/rpm
    spd_ki: float = 0.02  # A/(rpm s)
    anti_windup: bool = False
    filter_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"current_kp": self.cur_kp, "current_ki": self.cur_ki, "speed_kp": self.spd_kp,
                "speed_ki": self.spd_ki, "anti_windup": self.anti_windup, "filter_ms": self.filter_ms}


FACTORY = Gains()


class Drive:
    def __init__(self, m: Motor, gains: Gains, noise_seed: str) -> None:
        self.m = m
        self.g = replace(gains)
        self.k = 0
        self.i = 0.0
        self.w = 0.0
        self.theta = 0.0
        self.count_prev = 0
        self.mode = "off"
        self.value = 0.0  # V, A or rpm depending on the mode
        self.load = 0.0
        self.locked = False
        self.i_ref = 0.0
        self.int_i = 0.0
        self.int_w = 0.0
        self.w_meas = 0.0  # raw encoder speed, rad/s
        self.w_f = 0.0  # filtered
        self.i_meas = 0.0
        self.v_next = 0.0
        self.v = 0.0
        self.e_rl = math.exp(-m.r * TS / m.l)
        self.rng = SeededRandom(noise_seed)
        self.frames: list[list[Any]] = []
        self.incidents: list[dict[str, Any]] = []
        self.i_peak = 0.0
        self._record()

    @property
    def t(self) -> float:
        return self.k * TS

    def set_mode(self, mode: str, value: float) -> None:
        if mode != self.mode:
            self.int_i = self.v if mode in ("current", "speed") else 0.0
            self.int_w = self.i_ref if mode == "speed" else 0.0
        self.mode, self.value = mode, value

    def _record(self) -> None:
        speed_ref = self.value if self.mode == "speed" else None
        i_ref = self.i_ref if self.mode in ("current", "speed") else None
        self.frames.append([round(1000 * self.t, 1), round(self.w * RPM, 2), round(self.w_meas * RPM, 1),
                            round(self.i, 4), None if i_ref is None else round(i_ref, 4), round(self.v, 3),
                            round(self.load, 4), speed_ref])

    def _speed_loop(self) -> None:
        g, m = self.g, self.m
        count = math.floor(self.theta * COUNTS / (2 * math.pi))
        self.w_meas = (count - self.count_prev) * 2 * math.pi / COUNTS / (SPEED_EVERY * TS)
        self.count_prev = count
        tf = g.filter_ms / 1000.0
        ts = SPEED_EVERY * TS
        self.w_f = self.w_meas if tf <= 0 else self.w_f + ts / (tf + ts) * (self.w_meas - self.w_f)
        if self.mode != "speed":
            return
        e = self.value - self.w_f * RPM
        p = g.spd_kp * e
        di = g.spd_ki * ts * e
        trial = p + self.int_w + di
        if not (g.anti_windup and ((trial > m.i_max and di > 0) or (trial < -m.i_max and di < 0))):
            self.int_w += di
        self.i_ref = max(-m.i_max, min(m.i_max, p + self.int_w))

    def advance(self, seconds: float) -> None:
        m, g = self.m, self.g
        for _ in range(round(seconds / TS)):
            self.i_meas = self.i + m.noise_a * self.rng.gauss()
            if self.k % SPEED_EVERY == 0:
                self._speed_loop()
            if self.mode == "current":
                self.i_ref = max(-m.i_max, min(m.i_max, self.value))
            if self.mode in ("current", "speed"):
                e = self.i_ref - self.i_meas
                di = g.cur_ki * TS * e
                trial = g.cur_kp * e + self.int_i + di
                if not ((trial > m.v_supply and di > 0) or (trial < -m.v_supply and di < 0)):
                    self.int_i += di  # the current loop always has anti-windup
                v_cmd = g.cur_kp * e + self.int_i
            elif self.mode == "voltage":
                v_cmd = self.value
            else:
                v_cmd = 0.0
            self.v = self.v_next  # one sample of computation delay
            self.v_next = max(-m.v_supply, min(m.v_supply, v_cmd))
            if self.mode == "off":
                self.i = 0.0
                self.v = 0.0
            else:
                i_ss = (self.v - m.kt * self.w) / m.r
                self.i = i_ss + (self.i - i_ss) * self.e_rl
            if self.locked:
                self.w = 0.0
            else:
                torque = m.kt * self.i - m.b * self.w - self.load
                self.w += torque * TS / m.j
                self.theta += self.w * TS
            self.k += 1
            self.i_peak = max(self.i_peak, abs(self.i))
            if abs(self.i) > TRIP * m.i_max:
                self.incidents.append({"t_ms": round(1000 * self.t, 2), "alarm": "overcurrent trip",
                                       "current_a": round(self.i, 2)})
                self.mode, self.value = "off", 0.0
                self.i = self.v = self.v_next = self.int_i = self.int_w = self.i_ref = 0.0
            if self.k % FRAME_EVERY == 0:
                self._record()


FRAME_COLUMNS = ["t_ms", "speed_rpm", "measured_rpm", "current_a", "current_ref_a", "voltage_v", "load_nm", "speed_ref_rpm"]


@dataclass(frozen=True)
class Scenario:
    speed_rpm: float
    load_nm: float
    w_acc: float  # s
    w_load: float  # s

    @property
    def t1(self) -> float:
        return T_REF + self.w_acc

    @property
    def t2(self) -> float:
        return self.t1 + self.w_load

    @property
    def t_end(self) -> float:
        return self.t2 + self.w_load


def _settle(ts: list[float], ws: list[float], ref: float, t0: float, t1: float) -> float:
    last_out = None
    for t, w in zip(ts, ws):
        if t0 < t <= t1 and abs(w - ref) > BAND * ref:
            last_out = t
    if last_out is None:
        return 0.0
    if last_out >= t1 - 1e-3:
        return math.inf
    return last_out + 1e-3 - t0


def acceptance(m: Motor, sc: Scenario, gains: Gains, noise_seed: str) -> dict[str, Any]:
    d = Drive(m, gains, noise_seed)
    d.set_mode("speed", 0.0)
    d.advance(T_REF)
    d.set_mode("speed", sc.speed_rpm)
    d.advance(sc.w_acc)
    d.load = sc.load_nm
    d.advance(sc.w_load)
    d.load = 0.0
    d.advance(sc.w_load)
    ts = [f[0] / 1000.0 for f in d.frames]
    ws = [f[1] for f in d.frames]
    ref = sc.speed_rpm

    def seg(a: float, b: float) -> list[float]:
        return [w for t, w in zip(ts, ws) if a < t <= b]

    return {
        "frames": d.frames,
        "trip": bool(d.incidents),
        "i_peak_a": d.i_peak,
        "settle_s": _settle(ts, ws, ref, T_REF, sc.t1),
        "overshoot_pct": max(0.0, 100 * (max(seg(T_REF, sc.t1)) - ref) / ref),
        "dip_rpm": max(0.0, ref - min(seg(sc.t1, sc.t2)), max(seg(sc.t2, sc.t_end)) - ref),
        "recovery_s": max(_settle(ts, ws, ref, sc.t1, sc.t2), _settle(ts, ws, ref, sc.t2, sc.t_end)),
    }


def textbook(m: Motor, a: float = 1.0, b: float = 1.0) -> Gains:
    wc = 2 * math.pi * (1 / TS) / 20  # current loop crossover
    ws = wc / 10  # speed loop crossover
    kp_rad = m.j * ws / m.kt  # A per rad/s
    kp = kp_rad / RPM * a  # A per rpm
    ti = 4 / ws * b
    return Gains(cur_kp=m.l * wc, cur_ki=m.r * wc, spd_kp=kp, spd_ki=kp / ti, anti_windup=True,
                 filter_ms=1000 / (4 * ws))


PROMPT = """\
You are commissioning the speed drive of a permanent-magnet DC motor on a test bench. The bench is simulated; \
treat it as a real one.

Drive
- A bridge fed from a {v:g} V supply makes the armature voltage, -{v:g}..{v:g} V.
- Current loop: PI at 10 kHz, voltage = Kp_i * e + Ki_i * integral of e dt, e = current reference - measured current \
(A), output limited to the supply, with anti-windup. The voltage it computes is applied one sample (0.1 ms) later. \
The current sensor has some noise.
- Speed loop: PI at 1 kHz, current reference = Kp_w * e + Ki_w * integral of e dt, e = speed reference - measured \
speed (rpm), limited to +/-{imax:g} A, the drive's rated current. Optional anti-windup (integration stops while the \
output is limited). Optional first-order filter on the measured speed (time constant in ms).
- Speed is measured with a 1024-line encoder (4096 counts per revolution), as the count difference over each \
1 ms speed loop sample.
- Overcurrent trip: if the motor current ever exceeds {trip:g} A ({tf:g} times the rated current), the bridge switches \
off. set_mode starts it again.
- A dynamometer on the shaft can apply a braking torque, and can lock the shaft at standstill.
- Resistance, inductance, torque constant and inertia are not given. Find out what you need by experimenting.

Modes: "off"; "voltage" (you set the armature voltage directly, no current control); "current" (you set the current \
reference, speed loop off); "speed" (cascade: you set the speed reference in rpm). Now the motor is at rest, the \
drive off. Factory gains: current Kp {ckp:g} V/A, Ki {cki:g} V/(A s); speed Kp {skp:g} A/rpm, Ki {ski:g} A/(rpm s), \
no anti-windup, no speed filter.

Acceptance test. It runs after you call submit, with the gains you leave:
1. From rest in speed mode, at t = 20 ms the speed reference steps from 0 to {speed:g} rpm.
2. {wacc:g} ms later the dynamometer applies {load:g} Nm; {wload:g} ms after that it releases it; the test ends \
{wload:g} ms later.
Requirements, on the true speed:
- no overcurrent trip;
- within +/-1 % of {speed:g} rpm no later than {settle} ms after the reference step, and staying there until the load;
- overshoot at most {overshoot} %;
- speed error at most {dip} rpm when the load goes on and when it comes off;
- back within +/-1 % within {recovery} ms after each, and staying there.

You have {budget:g} s of motor time and at most {calls} tool calls. Work like a drives engineer: identify the \
motor, tune the current loop, then the speed loop, check on the bench, then call submit.
"""


class DcMotor:
    id = "control.dc_motor"
    domain = "control"
    title = "DC motor speed drive"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        v = rng.choice((24.0, 36.0, 48.0))
        r = rng.uniform(0.4, 2.5)
        self.motor = Motor(
            v_supply=v, r=r, l=rng.uniform(0.4e-3, 4e-3), kt=rng.uniform(0.04, 0.2), j=rng.uniform(5e-5, 1e-3),
            b=rng.uniform(1e-5, 1e-4), i_max=round(0.3 * v / r, 1), noise_a=0.0,
        )
        m = replace(self.motor, noise_a=0.005 * round(0.3 * v / r, 1))
        self.motor = m
        speed = float(round(0.6 * m.no_load_rpm, -1))
        ws = 2 * math.pi * (1 / TS) / 200
        t_acc = m.j * speed / RPM / (m.kt * m.i_max)
        self.scenario = Scenario(
            speed_rpm=speed, load_nm=round(0.5 * m.kt * m.i_max, 3),
            w_acc=round(max(0.05, 3 * t_acc + 40 / ws), 2), w_load=round(max(0.05, 40 / ws), 2),
        )
        if seed not in _LIMITS:
            family = [acceptance(m, self.scenario, textbook(m, a, b), self._noise("acceptance")) for a, b in FAMILY]
            _LIMITS[seed] = {
                "settle_ms": math.ceil(SLACK * 1000 * max(f["settle_s"] for f in family)),
                "overshoot_pct": max(1.0, math.ceil(10 * SLACK * max(f["overshoot_pct"] for f in family)) / 10),
                "dip_rpm": max(5, math.ceil(SLACK * max(f["dip_rpm"] for f in family))),
                "recovery_ms": max(5, math.ceil(SLACK * 1000 * max(f["recovery_s"] for f in family))),
            }
        self.limits = dict(_LIMITS[seed])
        self.drive = Drive(m, FACTORY, self._noise("commission"))
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
        return self.drive.t

    def prompt(self) -> str:
        m, sc, lim = self.motor, self.scenario, self.limits
        return PROMPT.format(
            v=m.v_supply, imax=m.i_max, trip=round(TRIP * m.i_max, 2), tf=TRIP, ckp=FACTORY.cur_kp, cki=FACTORY.cur_ki,
            skp=FACTORY.spd_kp, ski=FACTORY.spd_ki, speed=sc.speed_rpm, wacc=round(1000 * sc.w_acc),
            wload=round(1000 * sc.w_load), load=sc.load_nm, settle=f"{lim['settle_ms']:g}",
            overshoot=f"{lim['overshoot_pct']:g}", dip=f"{lim['dip_rpm']:g}", recovery=f"{lim['recovery_ms']:g}",
            budget=COMMISSION_BUDGET_S, calls=MAX_CALLS,
        )

    def tools(self) -> list[dict[str, Any]]:
        num = {"type": "number"}
        return [
            tool("set_mode", "Set the drive mode and its value: off; voltage (V); current (A); speed (rpm).",
                 {"mode": {"type": "string", "enum": ["off", "voltage", "current", "speed"]}, "value": num}, ["mode"]),
            tool("set_current_pi", "Current loop gains: kp in V/A, ki in V/(A s).",
                 {"kp": {**num, "minimum": 0}, "ki": {**num, "minimum": 0}}, ["kp", "ki"]),
            tool("set_speed_pi", "Speed loop gains: kp in A/rpm, ki in A/(rpm s); anti-windup; speed filter in ms. "
                 "Omitted ones keep their value.",
                 {"kp": {**num, "minimum": 0}, "ki": {**num, "minimum": 0}, "anti_windup": {"type": "boolean"},
                  "filter_ms": {**num, "minimum": 0, "maximum": 100}}),
            tool("set_load", "Dynamometer braking torque in Nm (0 to release).",
                 {"torque_nm": {**num, "minimum": 0}}, ["torque_nm"]),
            tool("lock_shaft", "Lock the shaft (only at standstill) or release it.",
                 {"locked": {"type": "boolean"}}, ["locked"]),
            tool("run", f"Let the drive run for `ms` milliseconds and return measured speed (encoder), measured current "
                 f"and voltage, at most {MAX_ROWS} rows (finer for short runs, at most every 0.1 ms).",
                 {"ms": {**num, "minimum": 0.1, "maximum": MAX_RUN_MS}}, ["ms"]),
            tool("submit", "Finish commissioning. The acceptance test then runs with the current gains."),
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise ToolError("already submitted")
        self.calls += 1
        if self.calls > MAX_CALLS and name != "submit":
            raise ToolError(f"tool call limit ({MAX_CALLS}) reached; call submit")
        d, m = self.drive, self.motor
        if name == "set_mode":
            mode = args.get("mode")
            if mode not in ("off", "voltage", "current", "speed"):
                raise ToolError("mode must be off, voltage, current or speed")
            value = 0.0 if mode == "off" else _num(args, "value", -1e9, 1e9)
            lim = {"off": 0.0, "voltage": m.v_supply, "current": m.i_max, "speed": 1.2 * m.no_load_rpm}[mode]
            if abs(value) > lim + 1e-9:
                raise ToolError(f"{mode} value must be within +/-{lim:.4g}")
            d.set_mode(mode, value)
            return self._status()
        if name == "set_current_pi":
            d.g.cur_kp = _num(args, "kp", 0.0, 1e4)
            d.g.cur_ki = _num(args, "ki", 0.0, 1e8)
            return self._status()
        if name == "set_speed_pi":
            unknown = set(args) - {"kp", "ki", "anti_windup", "filter_ms"}
            if unknown:
                raise ToolError(f"unknown parameters: {', '.join(sorted(unknown))}")
            if "kp" in args:
                d.g.spd_kp = _num(args, "kp", 0.0, 1e3)
            if "ki" in args:
                d.g.spd_ki = _num(args, "ki", 0.0, 1e6)
            if "anti_windup" in args:
                if not isinstance(args["anti_windup"], bool):
                    raise ToolError("anti_windup must be true or false")
                d.g.anti_windup = args["anti_windup"]
            if "filter_ms" in args:
                d.g.filter_ms = _num(args, "filter_ms", 0.0, 100.0)
            return self._status()
        if name == "set_load":
            d.load = _num(args, "torque_nm", 0.0, 1e3)
            return self._status()
        if name == "lock_shaft":
            locked = args.get("locked")
            if not isinstance(locked, bool):
                raise ToolError("locked must be true or false")
            if locked and abs(d.w) * RPM > 1.0:
                raise ToolError("the shaft can only be locked at standstill")
            d.locked = locked
            return self._status()
        if name == "run":
            ms = _num(args, "ms", 0.1, MAX_RUN_MS)
            left = COMMISSION_BUDGET_S - d.t
            if left <= 1e-9:
                raise ToolError("motor time is used up; call submit")
            n_total = max(1, round(min(ms / 1000, left) / TS))
            n_sample = max(1, math.ceil(n_total / MAX_ROWS))
            n_inc = len(d.incidents)
            rows, done = [], 0
            while done < n_total:
                n = min(n_sample, n_total - done)
                d.advance(n * TS)
                done += n
                rows.append([round(1000 * d.t, 2), round(d.w_meas * RPM, 1), round(d.i_meas, 3), round(d.v, 3)])
            out = self._status()
            out["trend"] = {"columns": ["t_ms", "speed_rpm", "current_a", "voltage_v"], "rows": rows}
            if len(d.incidents) > n_inc:
                out["alarms"] = d.incidents[n_inc:]
            return out
        if name == "submit":
            self._submitted = True
            return {"submitted": True, "t_ms": round(1000 * d.t, 2)}
        raise ToolError(f"unknown tool {name!r}")

    def _status(self) -> dict[str, Any]:
        d = self.drive
        return {
            "t_ms": round(1000 * d.t, 2), "mode": d.mode, "value": d.value, "speed_rpm": round(d.w_meas * RPM, 1),
            "current_a": round(d.i_meas, 3), "voltage_v": round(d.v, 3), "load_nm": d.load, "shaft_locked": d.locked,
            "gains": d.g.to_dict(), "time_left_ms": round(1000 * (COMMISSION_BUDGET_S - d.t), 1),
        }

    def finish(self) -> Score:
        if self.result is None:
            self.result = acceptance(self.motor, self.scenario, self.drive.g, self._noise("acceptance"))
        r, lim = self.result, self.limits
        trips = len(self.drive.incidents)
        ratios = {
            "settle_ratio": scoring.at_most(1000 * r["settle_s"], lim["settle_ms"]),
            "overshoot_ratio": scoring.at_most(r["overshoot_pct"], lim["overshoot_pct"]),
            "dip_ratio": scoring.at_most(r["dip_rpm"], lim["dip_rpm"]),
            "recovery_ratio": scoring.at_most(1000 * r["recovery_s"], lim["recovery_ms"]),
            "commissioning_ratio": 1.0 if trips == 0 else 0.5,
        }
        metrics = {
            "submitted": self._submitted, "tool_calls": self.calls, "motor_time_ms": round(1000 * self.drive.t, 1),
            "commissioning_trips": trips,
            **{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items() if k != "frames"},
            "final_gains": self.drive.g.to_dict(), **ratios,
        }
        if r["trip"]:
            return Score(score=0.0, passed=False, error="overcurrent trip in the acceptance test", metrics=metrics)
        score = scoring.combine(*ratios.values())
        return Score(score=score, passed=score >= 1.0, metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        g = result.get("gains") or {}
        if name == "set_mode":
            unit = {"voltage": "V", "current": "A", "speed": "rpm"}.get(result["mode"], "")
            return f"mode {result['mode']}" + (f", {result['value']:g} {unit}" if unit else ""), False
        if name == "set_current_pi":
            return f"current loop Kp {g['current_kp']:.4g} V/A, Ki {g['current_ki']:.4g} V/(A s)", False
        if name == "set_speed_pi":
            return (f"speed loop Kp {g['speed_kp']:.4g} A/rpm, Ki {g['speed_ki']:.4g} A/(rpm s), anti-windup "
                    f"{'on' if g['anti_windup'] else 'off'}, filter {g['filter_ms']:g} ms"), False
        if name == "set_load":
            return f"dynamometer {result['load_nm']:g} Nm", False
        if name == "lock_shaft":
            return "shaft locked" if result["shaft_locked"] else "shaft released", False
        if name == "run":
            rows = result["trend"]["rows"]
            sp = [r_[1] for r_ in rows]
            cu = [r_[2] for r_ in rows]
            text = (f"ran to {result['t_ms']:.1f} ms in {result['mode']} mode: speed {sp[0]:.0f} -> {sp[-1]:.0f} rpm, "
                    f"current {min(cu):.2f}..{max(cu):.2f} A, {len(rows)} rows")
            alarms = result.get("alarms") or []
            for a in alarms:
                text += f"; OVERCURRENT TRIP at {a['t_ms']:.1f} ms ({a['current_a']:.1f} A)"
            return text, bool(alarms)
        if name == "submit":
            return "handed over for the acceptance test", False
        return "", False

    def visual(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def view(self) -> dict[str, Any]:
        m, sc, lim = self.motor, self.scenario, self.limits
        phases: list[dict[str, Any]] = [
            {"key": "commissioning", "name": "Commissioning", "rows": _seconds(self.drive.frames), "events": [], "marks": []}
        ]
        if self.result is not None:
            phases.append({
                "key": "acceptance", "name": "Acceptance test", "rows": _seconds(self.result["frames"]),
                "events": [{"t_s": T_REF, "label": f"{sc.speed_rpm:g} rpm"},
                           {"t_s": sc.t1, "label": f"load {sc.load_nm:g} Nm"},
                           {"t_s": sc.t2, "label": "load off"}],
                "marks": [{"t_s": T_REF + lim["settle_ms"] / 1000, "label": "settle limit"},
                          {"t_s": sc.t1 + lim["recovery_ms"] / 1000, "label": "recovery limit"},
                          {"t_s": sc.t2 + lim["recovery_ms"] / 1000, "label": "recovery limit"}],
            })
        top = math.ceil(m.no_load_rpm / 500) * 500
        imax = m.i_max
        out: dict[str, Any] = {
            "kind": "motor",
            "columns": ["t_s"] + FRAME_COLUMNS[1:],
            "phases": phases,
            "scene": {"no_load_rpm": m.no_load_rpm, "i_max": imax, "trip_a": TRIP * imax, "v_supply": m.v_supply,
                      "load_max": m.kt * imax, "final_gains": self.drive.g.to_dict()},
            "panels": [
                {"title": "Speed", "unit": "rpm", "range": [-0.1 * top, top], "ticks": [0, top / 2, top], "height": 1.5,
                 "clip": True,
                 "series": [{"col": 7, "label": "speed reference", "color": "ink", "dash": True},
                            {"col": 2, "label": "encoder speed", "color": "measured", "width": 1},
                            {"col": 1, "label": "speed (true)", "color": "curve", "width": 2}],
                 "bands": {"acceptance": [{"lo": sc.speed_rpm * (1 - BAND), "hi": sc.speed_rpm * (1 + BAND),
                                           "label": "±1 % band"}]}},
                {"title": "Current", "unit": "A", "range": [-TRIP * imax * 1.1, TRIP * imax * 1.1],
                 "ticks": [round(-TRIP * imax, 1), round(-imax, 1), 0, round(imax, 1), round(TRIP * imax, 1)],
                 "height": 1.2, "clip": True,
                 "series": [{"col": 4, "label": "current reference", "color": "ink", "dash": True},
                            {"col": 3, "label": "current", "color": "output", "width": 1.5}],
                 "hlines": [{"y": TRIP * imax, "label": "overcurrent trip", "color": "critical"},
                            {"y": -TRIP * imax, "color": "critical"}]},
                {"title": "Voltage and load", "unit": "V", "range": [-m.v_supply, m.v_supply],
                 "ticks": [-m.v_supply, 0, m.v_supply], "height": 0.8,
                 "series": [{"col": 5, "label": "armature voltage", "color": "demand", "width": 1}]},
            ],
        }
        if self.result is not None:
            out["result"] = self._result_view()
        return out

    def _result_view(self) -> dict[str, Any]:
        r, lim, m, sc = self.result, self.limits, self.motor, self.scenario
        trips = len(self.drive.incidents)
        ms = lambda v: "never" if math.isinf(v) else f"{1000 * v:.0f} ms"  # noqa: E731
        reqs = [
            ("No overcurrent trip", "trip" if r["trip"] else f"peak {r['i_peak_a']:.1f} A", f"{TRIP * m.i_max:.1f} A",
             not r["trip"]),
            (f"Within ±1 % of {sc.speed_rpm:g} rpm after the step", ms(r["settle_s"]), f"{lim['settle_ms']:g} ms",
             1000 * r["settle_s"] <= lim["settle_ms"]),
            ("Overshoot", f"{r['overshoot_pct']:.1f} %", f"{lim['overshoot_pct']:g} %", r["overshoot_pct"] <= lim["overshoot_pct"]),
            ("Speed error when the load goes on or off", f"{r['dip_rpm']:.0f} rpm", f"{lim['dip_rpm']:g} rpm",
             r["dip_rpm"] <= lim["dip_rpm"]),
            ("Back within ±1 % after the load changes", ms(r["recovery_s"]), f"{lim['recovery_ms']:g} ms",
             1000 * r["recovery_s"] <= lim["recovery_ms"]),
            ("Trips while commissioning", str(trips), "0", trips == 0),
        ]
        g, ref = self.drive.g, textbook(m)
        return {
            "requirements": [{"label": a, "measured": b, "limit": c, "ok": bool(d)} for a, b, c, d in reqs],
            "hidden": [
                ["Armature resistance / inductance", f"{m.r:.2f} Ω / {1000 * m.l:.2f} mH"],
                ["Torque constant", f"{m.kt:.3f} Nm/A"],
                ["Inertia", f"{1e4 * m.j:.2f} kg cm²"],
                ["Electrical time constant", f"{1000 * m.l / m.r:.2f} ms"],
                ["No-load speed", f"{m.no_load_rpm:.0f} rpm"],
            ],
            "final": {
                "title": "Final gains", "columns": ["", "agent", "textbook cascade"],
                "rows": [
                    ["Current Kp", f"{g.cur_kp:.4g} V/A", f"{ref.cur_kp:.4g}"],
                    ["Current Ki", f"{g.cur_ki:.4g} V/(A s)", f"{ref.cur_ki:.4g}"],
                    ["Speed Kp", f"{g.spd_kp:.4g} A/rpm", f"{ref.spd_kp:.4g}"],
                    ["Speed Ki", f"{g.spd_ki:.4g} A/(rpm s)", f"{ref.spd_ki:.4g}"],
                    ["Anti-windup", "on" if g.anti_windup else "off", "on"],
                    ["Speed filter", f"{g.filter_ms:.2f} ms", f"{ref.filter_ms:.2f} ms"],
                ],
            },
        }


def _seconds(frames: list[list[Any]]) -> list[list[Any]]:
    """Frames with time in seconds, as the viewer expects."""
    return [[f[0] / 1000.0] + f[1:] for f in frames]


def _num(args: dict[str, Any], key: str, lo: float, hi: float) -> float:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise ToolError(f"{key} must be between {lo:g} and {hi:g}")
    return float(v)


# Scripted agents. They work only through the tools, like a model, and narrate.


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def reference_agent(ep: Any) -> None:
    env = ep.env
    v, imax = env.motor.v_supply, env.motor.i_max
    sc = env.scenario
    ep.model_turn("First the electrical data with the shaft locked: a small voltage step, a few percent of the supply, "
                  "so the locked-rotor current stays far below the trip. Resistance from the final current, inductance "
                  "from the 63 % rise time.")
    ep.tool_call("lock_shaft", {"locked": True})
    vt = round(0.05 * v, 3)
    ep.tool_call("set_mode", {"mode": "voltage", "value": vt})
    rows = ep.tool_call("run", {"ms": 30})["trend"]["rows"]
    t0 = rows[0][0] - 0.1
    i_end = _mean([r[2] for r in rows[-20:]])
    r_est = vt / i_end
    t63 = next(r[0] for r in rows if r[2] >= 0.632 * i_end) - t0 - 0.1  # minus the one-sample delay
    l_est = r_est * t63 / 1000
    ep.tool_call("set_mode", {"mode": "off"})
    if t63 * 4 > 30:  # slow electrics: measure again over a longer window
        ep.tool_call("set_mode", {"mode": "voltage", "value": vt})
        rows = ep.tool_call("run", {"ms": 120})["trend"]["rows"]
        t0 = rows[0][0] - 0.4
        i_end = _mean([r[2] for r in rows[-20:]])
        r_est = vt / i_end
        t63 = next(r[0] for r in rows if r[2] >= 0.632 * i_end) - t0 - 0.1
        l_est = r_est * t63 / 1000
        ep.tool_call("set_mode", {"mode": "off"})
    wc = 2 * math.pi * 500
    ep.model_turn(f"R = {r_est:.3f} ohm, L = {1000 * l_est:.3f} mH (tau {1000 * l_est / r_est:.2f} ms). Current PI by "
                  f"pole cancellation at 500 Hz, a twentieth of the 10 kHz loop rate to leave room for the one-sample "
                  f"delay: Kp = L*wc = {l_est * wc:.3f} V/A, Ki = R*wc = {r_est * wc:.1f} V/(A s).")
    ep.tool_call("set_current_pi", {"kp": round(l_est * wc, 5), "ki": round(r_est * wc, 3)})
    ep.tool_call("lock_shaft", {"locked": False})
    ep.model_turn("Now the mechanics: a constant current, and the acceleration it gives is Kt*i/J. I fit the slope only "
                  "while the speed is low, so the back-EMF never limits the current, and I brake back to rest with the "
                  "same current reversed.")
    i_test, accel = imax / 5, 0.0
    for _ in range(5):
        i_test = round(i_test, 4)
        ep.tool_call("set_mode", {"mode": "current", "value": i_test})
        rows = ep.tool_call("run", {"ms": 40})["trend"]["rows"]
        t_start = rows[0][0]
        low = [(r[0], r[1] / RPM) for r in rows if r[0] - t_start > 2 and r[1] < 0.4 * sc.speed_rpm]
        reached = next((r[0] for r in rows if r[1] >= 0.4 * sc.speed_rpm), rows[-1][0])
        ep.tool_call("set_mode", {"mode": "current", "value": -i_test})
        brake_ms = reached - t_start + (40 if reached >= rows[-1][0] else 0)
        ep.tool_call("run", {"ms": round(min(MAX_RUN_MS, max(1.0, brake_ms)), 1)})
        ep.tool_call("set_mode", {"mode": "off"})
        if reached - t_start >= 10 and len(low) >= 20:
            ts_, ws_ = [p[0] / 1000 for p in low], [p[1] for p in low]
            mt, mw = _mean(ts_), _mean(ws_)
            accel = sum((a - mt) * (b - mw) for a, b in zip(ts_, ws_)) / sum((a - mt) ** 2 for a in ts_)
            break
        i_test /= 4  # too light a rotor for this current: try again gentler
    j_over_kt = i_test / accel
    ws = wc / 10
    kp = j_over_kt * ws / RPM
    ti = 4 / ws
    filt = 1000 / (4 * ws)
    ep.model_turn(f"Acceleration {accel:.0f} rad/s^2 at {i_test:g} A, so J/Kt = {j_over_kt:.3e} A s^2/rad. Speed PI at a "
                  f"tenth of the current loop (50 Hz): Kp = {kp:.4g} A/rpm, Ti = {1000 * ti:.1f} ms, a {filt:.2f} ms filter "
                  "against encoder quantization, and anti-windup, since the start-up runs into the current limit.")
    ep.tool_call("set_speed_pi", {"kp": round(kp, 7), "ki": round(kp / ti, 6), "anti_windup": True, "filter_ms": round(filt, 3)})
    ep.tool_call("set_mode", {"mode": "speed", "value": 0})
    ep.tool_call("run", {"ms": 20})
    ep.model_turn(f"Check: the step to {sc.speed_rpm:g} rpm, then the load on and off.")
    ep.tool_call("set_mode", {"mode": "speed", "value": sc.speed_rpm})
    left = round(1000 * sc.w_acc)
    while left > 0:  # one run is at most 3 s
        ep.tool_call("run", {"ms": min(MAX_RUN_MS, left)})
        left -= MAX_RUN_MS
    ep.tool_call("set_load", {"torque_nm": sc.load_nm})
    ep.tool_call("run", {"ms": round(1000 * sc.w_load)})
    ep.tool_call("set_load", {"torque_nm": 0})
    ep.tool_call("run", {"ms": round(1000 * sc.w_load)})
    ep.model_turn("Clean start on the current limit, small dips on the load steps. Submitting.")
    ep.tool_call("submit", {})


def factory_agent(ep: Any) -> None:
    ep.model_turn("The drive has factory gains. I submit them as they are.")
    ep.tool_call("submit", {})


def voltage_bump_agent(ep: Any) -> None:
    ep.model_turn("Let me first see it spin: the full supply voltage, open loop.")
    ep.tool_call("set_mode", {"mode": "voltage", "value": ep.env.motor.v_supply})
    ep.tool_call("run", {"ms": 50})
    ep.model_turn("It tripped, but now I know the drive works. I'll use the textbook cascade.")
    g = textbook(ep.env.motor)
    ep.tool_call("set_current_pi", {"kp": g.cur_kp, "ki": g.cur_ki})
    ep.tool_call("set_speed_pi", {"kp": g.spd_kp, "ki": g.spd_ki, "anti_windup": True, "filter_ms": g.filter_ms})
    ep.tool_call("submit", {})


def no_antiwindup_agent(ep: Any) -> None:
    ep.model_turn("Textbook cascade, without anti-windup on the speed loop.")
    g = textbook(ep.env.motor)
    ep.tool_call("set_current_pi", {"kp": g.cur_kp, "ki": g.cur_ki})
    ep.tool_call("set_speed_pi", {"kp": g.spd_kp, "ki": g.spd_ki, "anti_windup": False, "filter_ms": g.filter_ms})
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "factory": factory_agent, "voltage_bump": voltage_bump_agent,
          "no_antiwindup": no_antiwindup_agent}
# no_antiwindup is a demo, not a negative: on very light rotors the start-up is too short to wind up.
NEGATIVE_AGENTS = ("factory", "voltage_bump")
