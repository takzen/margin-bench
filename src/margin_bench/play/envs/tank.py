"""process.tank_level: commission a level control loop on a process tank.

The plant is a vertical tank filled through a control valve and drained by
gravity through an outlet line, so the outflow follows the square root of the
level and the downstream demand. The level transmitter reads through a
transport delay and adds noise; the valve actuator has its own lag. A PID
block with a 0.5 s sample time drives the valve.

The model commissions the loop the way an engineer would: it drives the valve
by hand, asks operations to change the outlet demand, sets the PID
parameters, switches to automatic and lets time run, reading the trend after
every step. When it submits, a fixed acceptance test runs with the tuning it
left in the controller: start low, reach the setpoint, then ride out two
demand steps.

Difficulty rule: the acceptance limits are the worst result over a family of
sensible tunings run through the same test on the true plant: SIMC PI (tc
equal to the effective delay) with anti-windup, with Kc and Ti each scaled by
0.7, 1 and 1.4. Times and overshoot get 10 % slack on top (overshoot at least
2 cm), valve travel 50 %. So a PI tuning within about 40 % of SIMC passes once
anti-windup is on; the factory tuning, an aggressive Ziegler-Nichols tuning
and most tunings without anti-windup (the start-up saturates the valve) fail.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from margin_bench.core import Score
from margin_bench.grading import scoring
from margin_bench.play.env import ToolError, tool
from margin_bench.tasks.rng import SeededRandom

DT = 0.1  # s, integration step
TS = 0.5  # s, controller sample time
CTRL_EVERY = round(TS / DT)
FRAME_EVERY = round(1.0 / DT)  # one viewer frame per simulated second
SPAN = 2.0  # m, tank height and transmitter span
HH = 1.90  # m, high-high alarm: the tank spills
LL = 0.20  # m, low-low alarm: the pump runs dry
SP = 1.20  # m, acceptance setpoint
BAND = 0.03  # m
OUTLET_RANGE = (30.0, 80.0)  # %, what operations allow
OUTLET_NOMINAL = 50.0
COMMISSION_BUDGET_S = 4 * 3600.0
MAX_WAIT_S = 1800.0
MAX_TREND_POINTS = 300
MAX_CALLS = 40

T_AUTO = 10.0  # s into the acceptance test
WINDOW_S = 600.0  # s between acceptance events, before jitter

TUNING_SCALES = (0.7, 1.0, 1.4)
TIME_SLACK = 1.1
TRAVEL_SLACK = 1.5


@dataclass(frozen=True)
class Plant:
    area: float  # m^2
    q_max: float  # m^3/s with the valve fully open
    c_out: float  # outflow = c_out * demand/100 * sqrt(level)
    delay: float  # s, transmitter transport delay
    valve_lag: float  # s, actuator time constant
    noise: float  # m, transmitter noise (standard deviation)

    def valve_for(self, level: float, outlet: float) -> float:
        """Valve opening in % that holds `level` steady at outlet demand `outlet`."""
        return 100.0 * self.c_out * outlet / 100.0 * math.sqrt(level) / self.q_max


@dataclass
class Tuning:
    kc: float = 0.5
    ti_s: float = 300.0  # 0 disables integral action
    td_s: float = 0.0
    pv_filter_s: float = 0.0
    anti_windup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kc": self.kc,
            "ti_s": self.ti_s,
            "td_s": self.td_s,
            "pv_filter_s": self.pv_filter_s,
            "anti_windup": self.anti_windup,
        }


FACTORY_TUNING = Tuning()


@dataclass
class Pid:
    """Ideal-form PID in percent of span; derivative on the measurement."""

    tuning: Tuning = field(default_factory=Tuning)
    auto: bool = False
    sp: float = SP  # m
    out: float = 0.0  # %
    integ: float = 0.0
    pv_f: float | None = None
    pv_prev: float | None = None
    d: float = 0.0

    def _p(self) -> float:
        return self.tuning.kc * (100.0 * self.sp / SPAN - (self.pv_f or 0.0))

    def bumpless(self) -> None:
        self.integ = self.out - self._p() - self.d

    def sample(self, pv: float) -> float:
        t = self.tuning
        if self.pv_f is None or t.pv_filter_s <= 0:
            self.pv_f = pv
        else:
            self.pv_f += TS / (t.pv_filter_s + TS) * (pv - self.pv_f)
        if self.pv_prev is None:
            self.pv_prev = self.pv_f
        if not self.auto:
            self.pv_prev, self.d = self.pv_f, 0.0
            return self.out
        e = 100.0 * self.sp / SPAN - self.pv_f
        if t.td_s > 0:
            raw = -t.kc * t.td_s * (self.pv_f - self.pv_prev) / TS
            tf = t.td_s / 10.0
            self.d += TS / (tf + TS) * (raw - self.d)
        else:
            self.d = 0.0
        self.pv_prev = self.pv_f
        p = t.kc * e
        if t.ti_s > 0:
            di = t.kc * TS / t.ti_s * e
            trial = p + self.integ + di + self.d
            if not (t.anti_windup and ((trial > 100.0 and di > 0) or (trial < 0.0 and di < 0))):
                self.integ += di
        self.out = min(100.0, max(0.0, p + self.integ + self.d))
        return self.out


class Sim:
    """The plant and its controller, advanced in fixed steps."""

    def __init__(self, plant: Plant, level: float, outlet: float, tuning: Tuning, noise_seed: str) -> None:
        self.p = plant
        self.k = 0  # integration steps taken
        self.h = level
        self.outlet = outlet
        self.valve = plant.valve_for(level, outlet)  # actual position, %
        self.pid = Pid(tuning=replace(tuning), out=self.valve)
        self.pid.pv_f = self.pid.pv_prev = 100.0 * level / SPAN
        self.buf: deque[float] = deque([level] * (round(plant.delay / DT) + 1), maxlen=round(plant.delay / DT) + 1)
        self.rng = SeededRandom(noise_seed)
        self.meas = level
        self.frames: list[list[Any]] = []
        self.incidents: list[dict[str, Any]] = []
        self._alarm: str | None = None
        self.travel = 0.0
        self._record()

    @property
    def t(self) -> float:
        return self.k * DT

    def _measure(self) -> float:
        m = self.buf[0] + self.p.noise * self.rng.gauss()
        self.meas = min(SPAN, max(0.0, m))
        return 100.0 * self.meas / SPAN

    def _record(self) -> None:
        self.frames.append([
            round(self.t, 1),
            round(self.h, 4),
            round(self.meas, 4),
            round(self.pid.out, 2),
            round(self.valve, 2),
            round(self.outlet, 1),
            round(self.pid.sp, 3) if self.pid.auto else None,
        ])

    def advance(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            if self.k % CTRL_EVERY == 0:
                before = self.pid.out
                self.pid.sample(self._measure())
                self.travel += abs(self.pid.out - before)
            self.valve += (self.pid.out - self.valve) * DT / self.p.valve_lag
            q_in = self.p.q_max * self.valve / 100.0
            q_out = self.p.c_out * self.outlet / 100.0 * math.sqrt(max(self.h, 0.0))
            self.h = min(SPAN, max(0.0, self.h + (q_in - q_out) * DT / self.p.area))
            self.buf.append(self.h)
            self.k += 1
            alarm = "high-high" if self.h >= HH else "low-low" if self.h <= LL else None
            if alarm and alarm != self._alarm:
                self.incidents.append({"t_s": round(self.t, 1), "alarm": alarm, "level_m": round(self.h, 3)})
            self._alarm = alarm
            if self.k % FRAME_EVERY == 0:
                self._record()

    def set_manual(self, valve: float) -> None:
        self.pid.auto = False
        self.pid.out = valve

    def set_auto(self, sp: float) -> None:
        self.pid.sp = sp
        if not self.pid.auto:
            # The integral term starts from the current output, so the
            # proportional term acts at once on any setpoint error.
            self.pid.auto = True
            self.pid.integ, self.pid.d = self.pid.out, 0.0

    def set_tuning(self, tuning: Tuning) -> None:
        self.pid.tuning = tuning
        if self.pid.auto:
            self.pid.bumpless()


FRAME_COLUMNS = ["t_s", "level_m", "measured_m", "output_pct", "valve_pct", "outlet_pct", "setpoint_m"]


def _settle_time(ts: list[float], hs: list[float], t0: float, t1: float) -> float:
    """Seconds after t0 until the level stays inside the band through t1 (inf if it never does)."""
    last_out = None
    for t, h in zip(ts, hs):
        if t0 <= t <= t1 and abs(h - SP) > BAND:
            last_out = t
    if last_out is None:
        return 0.0
    if last_out >= t1 - 1.0:
        return math.inf
    return last_out + 1.0 - t0


def _ceil(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


@dataclass(frozen=True)
class Scenario:
    commission_level: float
    start_level: float
    t_step1: float
    t_step2: float
    t_end: float
    outlet1: float
    outlet2: float


def acceptance(plant: Plant, sc: Scenario, tuning: Tuning, noise_seed: str) -> dict[str, Any]:
    sim = Sim(plant, sc.start_level, OUTLET_NOMINAL, tuning, noise_seed)
    sim.advance(T_AUTO)
    sim.set_auto(SP)
    sim.advance(sc.t_step1 - T_AUTO)
    sim.outlet = sc.outlet1
    sim.advance(sc.t_step2 - sc.t_step1)
    sim.outlet = sc.outlet2
    sim.advance(sc.t_end - sc.t_step2)
    ts = [f[0] for f in sim.frames]
    hs = [f[1] for f in sim.frames]
    start = [h for t, h in zip(ts, hs) if T_AUTO <= t <= sc.t_step1]
    return {
        "frames": sim.frames,
        "max_level_m": max(hs),
        "min_level_m": min(hs),
        "alarm": bool(sim.incidents),
        "start_s": _settle_time(ts, hs, T_AUTO, sc.t_step1),
        "overshoot_m": max(0.0, max(start) - SP),
        "recovery1_s": _settle_time(ts, hs, sc.t_step1, sc.t_step2),
        "recovery2_s": _settle_time(ts, hs, sc.t_step2, sc.t_end),
        "travel_pct": sim.travel,
    }


def simc_pi(plant: Plant, tc: float | None = None) -> Tuning:
    """SIMC PI from the true plant, linearized at the setpoint and nominal demand."""
    v = plant.valve_for(SP, OUTLET_NOMINAL) / 100.0
    k = SP / v  # %PV per % valve
    q_nom = plant.c_out * OUTLET_NOMINAL / 100.0 * math.sqrt(SP)
    tau = 2.0 * plant.area * SP / q_nom
    theta = plant.delay + plant.valve_lag + TS / 2
    tc = theta if tc is None else tc
    return Tuning(kc=tau / (k * (tc + theta)), ti_s=min(tau, 4 * (tc + theta)), anti_windup=True)


def _fmt(x: float, unit: str) -> str:
    return f"{x:g} {unit}"


PROMPT = """\
You are commissioning a level control loop on a process tank. The plant is simulated, but treat it as a real one.

Plant
- A vertical tank, 2.0 m tall. Liquid comes in through a control valve and leaves by gravity through an outlet \
line, so the outflow depends on the level and on the outlet demand (30 to 80 %, set by operations).
- Level transmitter LT range 0 to 2.000 m. The level controller works in percent of span: PV = 100 * level / 2.0 m, \
output = valve opening in percent, limited to 0..100 %. Sample time 0.5 s.
- Controller form: ideal PID, output = Kc * (e + (1/Ti) * integral of e dt) + derivative term, with e = SP - PV in \
percent of span and the derivative acting on the measurement only: -Kc * Td * dPV/dt, filtered with Td/10. \
Ti = 0 switches integral action off. Optional first-order filter on PV (time constant in seconds). Optional \
anti-windup (integration stops while the output is limited). On the switch to automatic the integral term starts from the current output, so the proportional term acts at once on the setpoint error. Changing the tuning in automatic is bumpless.
- Tank area, valve size, outlet characteristics, transmitter delay and noise, and actuator speed are not given. \
Find out what you need by experimenting on the plant.
- Alarms: high-high at 1.90 m (the tank spills) and low-low at 0.20 m (the pump runs dry). Never reach either, \
including during your experiments.

Now: the controller is in manual, the valve at {valve:.1f} %, the level steady at about {level:.2f} m, outlet demand {outlet:g} %. \
Factory tuning: Kc = {kc:g}, Ti = {ti:g} s, Td = 0 s, no PV filter, anti-windup off.

Acceptance test. It runs after you call submit, with the tuning you leave in the controller:
1. The tank starts steady at a level between 0.50 and 0.70 m with the controller in manual. At t = 10 s the \
controller switches to automatic with setpoint 1.20 m.
2. About 600 s later the outlet demand steps up to 65..75 %. About 600 s after that it steps down to 35..45 %.
Requirements, on the true level:
- never at or above 1.90 m, never at or below 0.20 m;
- inside 1.20 +/- 0.03 m no later than {start} s after the switch to automatic, and staying inside until the first demand step;
- overshoot above 1.20 m during the start at most {overshoot} m;
- after each demand step, back inside 1.20 +/- 0.03 m for good within {recovery} s;
- total valve travel during the test (sum of all controller output changes) at most {travel} %.

You have {budget} hours of plant time for commissioning and at most {max_calls} tool calls. Work like a \
commissioning engineer: run experiments, read the trends, tune, check your tuning on the plant, then call submit.
"""


class TankLevel:
    id = "process.tank_level"
    domain = "control"
    title = "Tank level loop"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        tau = rng.uniform(60.0, 240.0)  # s, linearized at the setpoint
        q_frac = rng.uniform(0.40, 0.50)  # valve opening that holds the setpoint at nominal demand
        area = rng.uniform(0.8, 3.0)
        q_nom = 2.0 * area * SP / tau
        self.plant = Plant(
            area=area,
            q_max=q_nom / q_frac,
            c_out=q_nom / (OUTLET_NOMINAL / 100.0 * math.sqrt(SP)),
            delay=rng.uniform(3.0, 15.0),
            valve_lag=rng.uniform(1.0, 4.0),
            noise=rng.uniform(0.002, 0.006),
        )
        t1 = T_AUTO + WINDOW_S + rng.uniform(-60.0, 60.0)
        t2 = t1 + WINDOW_S + rng.uniform(-60.0, 60.0)
        self.scenario = Scenario(
            commission_level=rng.uniform(0.70, 1.00),
            start_level=rng.uniform(0.50, 0.70),
            t_step1=round(t1),
            t_step2=round(t2),
            t_end=round(t2 + WINDOW_S),
            outlet1=round(rng.uniform(65.0, 75.0)),
            outlet2=round(rng.uniform(35.0, 45.0)),
        )
        base = simc_pi(self.plant)
        family = [
            acceptance(self.plant, self.scenario, replace(base, kc=base.kc * a, ti_s=base.ti_s * b), self._noise("acceptance"))
            for a in TUNING_SCALES
            for b in TUNING_SCALES
        ]
        self.limits = {
            "start_s": _ceil(TIME_SLACK * max(r["start_s"] for r in family), 10.0),
            "overshoot_m": max(0.02, _ceil(TIME_SLACK * max(r["overshoot_m"] for r in family), 0.005)),
            "recovery_s": _ceil(TIME_SLACK * max(max(r["recovery1_s"], r["recovery2_s"]) for r in family), 10.0),
            "travel_pct": _ceil(TRAVEL_SLACK * max(r["travel_pct"] for r in family), 50.0),
        }
        self.sim = Sim(self.plant, self.scenario.commission_level, OUTLET_NOMINAL, FACTORY_TUNING, self._noise("commission"))
        self.calls = 0
        self._submitted = False
        self.result: dict[str, Any] | None = None

    def _noise(self, what: str) -> str:
        return f"{self.id}:{self.seed}:{what}"

    # Environment protocol ------------------------------------------------

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def sim_time(self) -> float:
        return self.sim.t

    def prompt(self) -> str:
        s = self.sim
        return PROMPT.format(
            valve=s.pid.out,
            level=s.h,
            outlet=s.outlet,
            kc=FACTORY_TUNING.kc,
            ti=FACTORY_TUNING.ti_s,
            start=f"{self.limits['start_s']:g}",
            overshoot=f"{self.limits['overshoot_m']:.3f}",
            recovery=f"{self.limits['recovery_s']:g}",
            travel=f"{self.limits['travel_pct']:g}",
            budget=f"{COMMISSION_BUDGET_S / 3600:g}",
            max_calls=MAX_CALLS,
        )

    def tools(self) -> list[dict[str, Any]]:
        num = {"type": "number"}
        return [
            tool("set_manual", "Switch the level controller to manual and set the valve output in percent.",
                 {"valve_percent": {**num, "minimum": 0, "maximum": 100}}, ["valve_percent"]),
            tool("set_auto", "Switch the level controller to automatic with this level setpoint in metres (or change the setpoint).",
                 {"setpoint_m": {**num, "minimum": 0.3, "maximum": 1.8}}, ["setpoint_m"]),
            tool("set_tuning", "Change controller parameters. Omitted ones keep their value. Returns the full tuning.",
                 {"kc": {**num, "minimum": 0, "maximum": 100}, "ti_s": {**num, "minimum": 0},
                  "td_s": {**num, "minimum": 0}, "pv_filter_s": {**num, "minimum": 0},
                  "anti_windup": {"type": "boolean"}}),
            tool("set_outlet_demand", "Ask operations to set the outlet demand, 30 to 80 percent.",
                 {"percent": {**num, "minimum": OUTLET_RANGE[0], "maximum": OUTLET_RANGE[1]}}, ["percent"]),
            tool("wait", "Let the plant run and return the trend: time, measured level, controller output, outlet "
                 "demand. sample_s sets the trend spacing (default: seconds/100, at least 0.5 s, at most 300 rows).",
                 {"seconds": {**num, "minimum": 0.5, "maximum": MAX_WAIT_S}, "sample_s": {**num, "minimum": 0.5}},
                 ["seconds"]),
            tool("submit", "Finish commissioning. The acceptance test then runs with the current tuning."),
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise ToolError("already submitted")
        self.calls += 1
        if self.calls > MAX_CALLS and name != "submit":
            raise ToolError(f"tool call limit ({MAX_CALLS}) reached; call submit")
        s = self.sim
        if name == "set_manual":
            s.set_manual(_num(args, "valve_percent", 0.0, 100.0))
            return self._status()
        if name == "set_auto":
            s.set_auto(_num(args, "setpoint_m", 0.3, 1.8))
            return self._status()
        if name == "set_tuning":
            t = replace(s.pid.tuning)
            for key in ("kc", "ti_s", "td_s", "pv_filter_s"):
                if key in args:
                    setattr(t, key, _num(args, key, 0.0, 100.0 if key == "kc" else 1e5))
            if "anti_windup" in args:
                if not isinstance(args["anti_windup"], bool):
                    raise ToolError("anti_windup must be true or false")
                t.anti_windup = args["anti_windup"]
            unknown = set(args) - {"kc", "ti_s", "td_s", "pv_filter_s", "anti_windup"}
            if unknown:
                raise ToolError(f"unknown parameters: {', '.join(sorted(unknown))}")
            s.set_tuning(t)
            return self._status()
        if name == "set_outlet_demand":
            s.outlet = _num(args, "percent", *OUTLET_RANGE)
            return self._status()
        if name == "wait":
            seconds = _num(args, "seconds", 0.5, MAX_WAIT_S)
            left = COMMISSION_BUDGET_S - s.t
            if left <= 0:
                raise ToolError("commissioning time is used up; call submit")
            n_total = max(1, round(min(seconds, left) / DT))
            sample = args.get("sample_s")
            sample = max(TS, seconds / 100) if sample is None else _num(args, "sample_s", TS, MAX_WAIT_S)
            n_sample = max(1, round(sample / DT), math.ceil(n_total / MAX_TREND_POINTS))
            n_inc = len(s.incidents)
            rows, done = [], 0
            while done < n_total:
                n = min(n_sample, n_total - done)
                s.advance(n * DT)
                done += n
                rows.append([round(s.t, 1), round(s.meas, 3), round(s.pid.out, 1), round(s.outlet, 1)])
            out = self._status()
            out["trend"] = {"columns": ["t_s", "level_m", "output_pct", "outlet_pct"], "rows": rows}
            if len(s.incidents) > n_inc:
                out["alarms"] = s.incidents[n_inc:]
            return out
        if name == "submit":
            self._submitted = True
            return {"submitted": True, "t_s": round(s.t, 1)}
        raise ToolError(f"unknown tool {name!r}")

    def _status(self) -> dict[str, Any]:
        s = self.sim
        return {
            "t_s": round(s.t, 1),
            "mode": "auto" if s.pid.auto else "manual",
            "setpoint_m": s.pid.sp if s.pid.auto else None,
            "level_m": round(s.meas, 3),
            "output_pct": round(s.pid.out, 1),
            "outlet_pct": s.outlet,
            "tuning": s.pid.tuning.to_dict(),
            "time_left_s": round(COMMISSION_BUDGET_S - s.t, 1),
        }

    def finish(self) -> Score:
        if self.result is None:
            self.result = acceptance(self.plant, self.scenario, self.sim.pid.tuning, self._noise("acceptance"))
        r, lim = self.result, self.limits
        incidents = len(self.sim.incidents)
        ratios = {
            "start_ratio": scoring.at_most(r["start_s"], lim["start_s"]),
            "overshoot_ratio": scoring.at_most(r["overshoot_m"], lim["overshoot_m"]),
            "recovery_ratio": scoring.at_most(max(r["recovery1_s"], r["recovery2_s"]), lim["recovery_s"]),
            "travel_ratio": scoring.at_most(r["travel_pct"], lim["travel_pct"]),
            "commissioning_ratio": 1.0 if incidents == 0 else 0.5,
        }
        metrics = {
            "submitted": self._submitted,
            "tool_calls": self.calls,
            "commissioning_s": round(self.sim.t, 1),
            "commissioning_alarms": incidents,
            "acceptance_alarm": r["alarm"],
            **{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items() if k != "frames"},
            "final_tuning": self.sim.pid.tuning.to_dict(),
            **ratios,
        }
        if r["alarm"]:
            return Score(score=0.0, passed=False, error="alarm during the acceptance test", metrics=metrics)
        score = scoring.combine(*ratios.values())
        return Score(score=score, passed=score >= 1.0, metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        """One line for the viewer's log, and whether it reports an alarm."""
        tu = result.get("tuning") or {}
        if name == "set_manual":
            return f"manual, valve {result['output_pct']:.1f} %", False
        if name == "set_auto":
            return f"automatic, setpoint {result['setpoint_m']:.2f} m", False
        if name == "set_outlet_demand":
            return f"outlet demand {result['outlet_pct']:g} %", False
        if name == "set_tuning":
            ti = f"{tu['ti_s']:.1f} s" if tu["ti_s"] > 0 else "off"
            return (f"Kc {tu['kc']:.3f} · Ti {ti} · Td {tu['td_s']:.1f} s · PV filter {tu['pv_filter_s']:.1f} s · "
                    f"anti-windup {'on' if tu['anti_windup'] else 'off'}"), False
        if name == "wait":
            rows = result["trend"]["rows"]
            lv = [r[1] for r in rows]
            text = (f"plant ran to {result['t_s']:.0f} s: level {lv[0]:.3f} -> {lv[-1]:.3f} m (min {min(lv):.3f}, "
                    f"max {max(lv):.3f}), output {result['output_pct']:.1f} %, {len(rows)} trend rows returned")
            alarms = result.get("alarms") or []
            for a in alarms:
                text += f"; {a['alarm'].upper()} ALARM at {a['t_s']:.0f} s ({a['level_m']:.3f} m)"
            return text, bool(alarms)
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
            sc = self.scenario
            phases.append({
                "key": "acceptance",
                "name": "Acceptance test",
                "rows": self.result["frames"],
                "events": [
                    {"t_s": T_AUTO, "label": f"auto, SP {SP:.2f} m"},
                    {"t_s": sc.t_step1, "label": f"demand {sc.outlet1:g} %"},
                    {"t_s": sc.t_step2, "label": f"demand {sc.outlet2:g} %"},
                ],
                "marks": [
                    {"t_s": T_AUTO + lim["start_s"], "label": "start limit"},
                    {"t_s": sc.t_step1 + lim["recovery_s"], "label": "recovery limit"},
                    {"t_s": sc.t_step2 + lim["recovery_s"], "label": "recovery limit"},
                ],
            })
        out: dict[str, Any] = {
            "kind": "tank",
            "smooth": True,
            "columns": FRAME_COLUMNS,
            "phases": phases,
            "scene": {"span_m": SPAN, "hh_m": HH, "ll_m": LL, "setpoint_m": SP, "band_m": BAND,
                      "final_tuning": self.sim.pid.tuning.to_dict()},
            "panels": [
                {
                    "title": "Level", "unit": "m", "range": [0.0, SPAN], "ticks": [0.0, 0.5, 1.0, 1.5, 2.0], "height": 2,
                    "series": [
                        {"col": 2, "label": "measured", "color": "measured", "width": 1},
                        {"col": 6, "label": "setpoint", "color": "ink", "dash": True},
                        {"col": 1, "label": "level (true)", "color": "curve", "width": 2},
                    ],
                    "hlines": [{"y": HH, "label": "alarm limits", "color": "critical"}, {"y": LL, "color": "critical"}],
                    "bands": {"acceptance": [{"lo": SP - BAND, "hi": SP + BAND, "label": "±3 cm band"}]},
                },
                {
                    "title": "Controller output and outlet demand", "unit": "%", "range": [0.0, 100.0],
                    "ticks": [0.0, 50.0, 100.0], "height": 1,
                    "series": [
                        {"col": 5, "label": "outlet demand", "color": "demand", "dash": True},
                        {"col": 3, "label": "controller output", "color": "output"},
                    ],
                },
            ],
        }
        if self.result is not None:
            out["result"] = self._result_view()
        return out

    def _result_view(self) -> dict[str, Any]:
        r, lim, p = self.result, self.limits, self.plant
        rec = max(r["recovery1_s"], r["recovery2_s"])
        alarms = len(self.sim.incidents)
        reqs = [
            ("No alarm during the acceptance test", "alarm" if r["alarm"] else "none", "none", not r["alarm"]),
            ("Inside 1.20 ± 0.03 m after the switch to automatic", _secs(r["start_s"]), f"{lim['start_s']:g} s",
             r["start_s"] <= lim["start_s"]),
            ("Overshoot during the start", f"{r['overshoot_m']:.3f} m", f"{lim['overshoot_m']:.3f} m",
             r["overshoot_m"] <= lim["overshoot_m"]),
            ("Back inside the band after each demand step (worst)", _secs(rec), f"{lim['recovery_s']:g} s",
             rec <= lim["recovery_s"]),
            ("Valve travel", f"{r['travel_pct']:,.0f} %", f"{lim['travel_pct']:,.0f} %", r["travel_pct"] <= lim["travel_pct"]),
            ("Alarms while commissioning", str(alarms), "0", alarms == 0),
        ]
        q_nom = p.c_out * OUTLET_NOMINAL / 100.0 * math.sqrt(SP)
        ft, rt = self.sim.pid.tuning, simc_pi(p)

        def ti(x: float) -> str:
            return f"{x:.0f} s" if x > 0 else "off"

        return {
            "requirements": [{"label": a, "measured": b, "limit": c, "ok": bool(d)} for a, b, c, d in reqs],
            "hidden": [
                ["Time constant", f"{2.0 * p.area * SP / q_nom:.0f} s"],
                ["Gain", f"{SP / (p.valve_for(SP, OUTLET_NOMINAL) / 100.0):.2f} %/%"],
                ["Transmitter delay", f"{p.delay:.1f} s"],
                ["Valve lag", f"{p.valve_lag:.1f} s"],
                ["Transmitter noise", f"{1000 * p.noise:.1f} mm"],
                ["Tank area", f"{p.area:.2f} m²"],
            ],
            "final": {
                "title": "Final tuning",
                "columns": ["", "agent", "SIMC reference"],
                "rows": [
                    ["Kc", f"{ft.kc:.2f}", f"{rt.kc:.2f}"],
                    ["Ti", ti(ft.ti_s), ti(rt.ti_s)],
                    ["Td", f"{ft.td_s:.1f} s", f"{rt.td_s:.1f} s"],
                    ["PV filter", f"{ft.pv_filter_s:.1f} s", f"{rt.pv_filter_s:.1f} s"],
                    ["Anti-windup", "on" if ft.anti_windup else "off", "on"],
                ],
            },
        }


def _secs(v: float) -> str:
    return "never" if math.isinf(v) else f"{v:g} s"


def _num(args: dict[str, Any], key: str, lo: float, hi: float) -> float:
    v = args.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ToolError(f"{key} must be a number")
    if not lo <= v <= hi:
        raise ToolError(f"{key} must be between {lo:g} and {hi:g}")
    return float(v)


# Scripted agents: the reference engineer and the known-bad ones. They work only
# through the tools, exactly like a model, and narrate what they do.


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _identify(ep: Any, valve: float, du: float) -> tuple[float, float, float]:
    """Step test in manual from `valve`; returns (gain %/%, time constant s, dead time s)."""
    before = ep.tool_call("wait", {"seconds": 60, "sample_s": 1})["trend"]["rows"]
    y0 = _mean([100 * r[1] / SPAN for r in before[-30:]])
    t_step = before[-1][0]
    ep.tool_call("set_manual", {"valve_percent": round(valve + du, 2)})
    rows = ep.tool_call("wait", {"seconds": 90, "sample_s": 0.5})["trend"]["rows"]
    rows += ep.tool_call("wait", {"seconds": 1200, "sample_s": 5})["trend"]["rows"]
    ts = [r[0] - t_step for r in rows]
    ys = [100 * r[1] / SPAN for r in rows]
    y_end = _mean(ys[-25:])
    dy = y_end - y0
    smooth = [_mean(ys[max(0, i - 2): i + 3]) for i in range(len(ys))]

    def crossing(frac: float) -> float:
        target = y0 + frac * dy
        for i in range(1, len(smooth)):
            if (smooth[i] - target) * (1 if dy > 0 else -1) >= 0:
                a, b = smooth[i - 1], smooth[i]
                w = 0.0 if b == a else (target - a) / (b - a)
                return ts[i - 1] + w * (ts[i] - ts[i - 1])
        return ts[-1]

    # Two-point fit at 10 % and 63 %: the 10 % point sits in the finely sampled part
    # and is far enough above the noise.
    t10, t63 = crossing(0.10), crossing(0.632)
    tau = (t63 - t10) / (1 + math.log(0.9))
    return dy / du, tau, max(TS, t63 - tau)


def reference_agent(ep: Any) -> None:
    ep.model_turn("First I watch the plant at rest for a minute to see the steady level, the noise and the valve position.")
    rows = ep.tool_call("wait", {"seconds": 60, "sample_s": 1})["trend"]["rows"]
    level, valve = _mean([r[1] for r in rows[-30:]]), rows[-1][2]
    v_sp = valve * math.sqrt(SP / level)
    ep.model_turn(
        f"Steady at {level:.3f} m with the valve at {valve:.1f} %. Gravity outflow goes with the square root of the "
        f"level, so about {v_sp:.1f} % should hold 1.20 m. I move the valve there by hand and let the level settle "
        "before the step test, so I identify the plant at the operating point."
    )
    ep.tool_call("set_manual", {"valve_percent": round(v_sp, 2)})
    ep.tool_call("wait", {"seconds": 1200, "sample_s": 10})
    ep.model_turn("Now a +5 % valve step in manual. Fine sampling first to catch the dead time, then a long coarse record for gain and time constant.")
    k, tau, theta = _identify(ep, v_sp, 5.0)
    kc, ti = tau / (k * 2 * theta), min(tau, 8 * theta)
    ep.model_turn(
        f"First order plus dead time fit (two points, 10 % and 63 % of the step): gain {k:.2f} %/%, time constant {tau:.0f} s, "
        f"dead time {theta:.1f} s. SIMC PI with tc equal to the dead time: Kc = {kc:.2f}, Ti = {ti:.0f} s. "
        "The start of the acceptance test saturates the valve, so I switch anti-windup on. No derivative: the "
        "transmitter is noisy and the valve would chatter."
    )
    ep.tool_call("set_tuning", {"kc": round(kc, 3), "ti_s": round(ti, 1), "td_s": 0, "pv_filter_s": 0, "anti_windup": True})
    ep.tool_call("set_auto", {"setpoint_m": SP})
    ep.tool_call("wait", {"seconds": 300, "sample_s": 3})
    ep.model_turn("In automatic and back at 1.20 m. Now I check disturbance rejection with the same demand steps as the acceptance test.")
    ep.tool_call("set_outlet_demand", {"percent": 70})
    ep.tool_call("wait", {"seconds": 500, "sample_s": 5})
    ep.tool_call("set_outlet_demand", {"percent": 40})
    ep.tool_call("wait", {"seconds": 500, "sample_s": 5})
    ep.tool_call("set_outlet_demand", {"percent": OUTLET_NOMINAL})
    ep.tool_call("wait", {"seconds": 300, "sample_s": 5})
    ep.model_turn("Both demand steps recovered well inside the limits and the valve is calm. Submitting.")
    ep.tool_call("submit", {})


def factory_agent(ep: Any) -> None:
    ep.model_turn("The controller already has a factory tuning. I submit it as it is.")
    ep.tool_call("submit", {})


def aggressive_agent(ep: Any) -> None:
    ep.model_turn("I run a quick step test and use the Ziegler-Nichols open-loop rule for a PI controller.")
    rows = ep.tool_call("wait", {"seconds": 60, "sample_s": 1})["trend"]["rows"]
    k, tau, theta = _identify(ep, rows[-1][2], 5.0)
    kc, ti = 0.9 * tau / (k * theta), 3.33 * theta
    ep.model_turn(f"Gain {k:.2f}, time constant {tau:.0f} s, dead time {theta:.1f} s. Ziegler-Nichols PI: Kc = {kc:.2f}, Ti = {ti:.0f} s.")
    ep.tool_call("set_tuning", {"kc": round(kc, 3), "ti_s": round(ti, 1)})
    ep.tool_call("set_auto", {"setpoint_m": SP})
    ep.tool_call("wait", {"seconds": 300, "sample_s": 3})
    ep.model_turn("The loop holds the setpoint. Submitting.")
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "factory": factory_agent, "aggressive": aggressive_agent}
NEGATIVE_AGENTS = ("factory", "aggressive")
