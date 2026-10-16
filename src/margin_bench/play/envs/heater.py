"""process.inline_heater: hold the outlet temperature of a heater with a long pipe.

Liquid flows through an electric heater and then through a long pipe to the
point of use, where a temperature transmitter sits. The transport delay of the
pipe dominates the dynamics, and it is not fixed: at lower flow the delay, the
heater's time constant and the gain all grow. The heater power is set by a PID
block with an optional Smith predictor.

The model commissions the loop: it drives the power by hand, asks operations
to change the flow, sets the controller and its predictor, switches to
automatic and lets time run, reading the trend each time. When it submits, the
acceptance test steps the setpoint up by 10 degrees, drops the inlet
temperature, then cuts the flow to 75 % of nominal.

A flow drop at constant power raises the outlet temperature by a third of the
heating rise, and feedback only sees it one pipe delay later, so no PID tuning
alone keeps the outlet within 3 degrees of the setpoint. The controller offers
flow feedforward (power scaled by the measured flow), which removes the gain
change at its source; finding that out is part of the task.

Difficulty rule: the limits are the worst result over a family of designs run
through the same test on the true plant: flow feedforward on, SIMC PI (tc equal
to the effective delay) for the plant at 75 % flow, the slowest case the test
reaches, with Kc and Ti scaled to the corners and centre of 0.7..1.4. Times get
10 % slack (at least 60 s), power travel 50 %. Without feedforward every tuning overheats on
the flow drop; a PI tuned on the heater lag alone, ignoring the delay, is
unstable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from margin_bench import scoring
from margin_bench.core import Score
from margin_bench.play.env import ToolError, tool
from margin_bench.play.pid import Pid, Tuning
from margin_bench.rng import SeededRandom

DT = 0.5  # s, integration step and controller sample time
FRAME_EVERY = 2  # one viewer frame per simulated second
SPAN = 150.0  # degC, transmitter range 0..150
T_SAFE = 110.0  # degC, over-temperature trip while commissioning
OVER_SP = 3.0  # degC above the setpoint in force: product damage in the acceptance test
BAND = 0.5  # degC
FLOW_RANGE = (60.0, 130.0)  # % of nominal
LOW_FLOW = 75.0
COMMISSION_BUDGET_S = 6 * 3600.0
MAX_WAIT_S = 3600.0
MAX_TREND_POINTS = 300
MAX_CALLS = 40
T_STEP = 10.0  # s into the acceptance test

FAMILY = ((1.0, 1.0), (0.7, 0.7), (0.7, 1.4), (1.4, 0.7), (1.4, 1.4))
TIME_SLACK = 1.1
TRAVEL_SLACK = 1.5

FACTORY_TUNING = Tuning(kc=1.0, ti_s=60.0)
_LIMITS: dict[int, dict[str, float]] = {}


@dataclass(frozen=True)
class Plant:
    gain_c: float  # degC rise at full power and nominal flow
    heater_tau_s: float  # heater time constant at nominal flow
    delay_s: float  # pipe transport delay at nominal flow
    sensor_tau_s: float
    noise_c: float

    def k_pct(self, flow: float) -> float:
        """Gain in percent of span per percent of power at flow (fraction of nominal)."""
        return self.gain_c / flow / SPAN

    def lumped(self, flow: float) -> tuple[float, float, float]:
        """First order plus dead time by the half rule: (gain %/%, time constant, effective delay)."""
        lags = sorted((self.heater_tau_s / flow, self.sensor_tau_s), reverse=True)
        return self.k_pct(flow), lags[0] + lags[1] / 2, self.delay_s / flow + lags[1] / 2 + DT / 2


class Sim:
    """Heater, plug-flow pipe, sensor and controller, advanced in DT steps."""

    def __init__(self, plant: Plant, t_in: float, power: float, tuning: Tuning, noise_seed: str,
                 sp: float | None = None, feedforward: bool = False) -> None:
        self.p = plant
        self.k = 0
        self.flow = 1.0
        self.ff = feedforward
        self.power = power  # what the heater gets: the controller output, times the flow with feedforward
        self.t_in = t_in
        th = t_in + plant.gain_c * power / 100.0
        self.th = th  # heater outlet
        self.out_t = th  # fluid at the sensor point
        self.sensor = th
        self.meas = th
        self.vol = 0.0  # cumulative volume through the heater, in nominal-flow seconds
        self.hist_v = [-plant.delay_s - 1.0, 0.0]
        self.hist_t = [th, th]
        self.j = 0
        self.pid = Pid(span=SPAN, ts=DT, tuning=replace(tuning), out=power)
        self.pid.pv_f = self.pid.pv_prev = 100.0 * th / SPAN
        if sp is not None:
            self.pid.set_auto(sp)
        if self.pid.smith_on:
            self.pid.reset_smith()
        self.rng = SeededRandom(noise_seed)
        self.frames: list[list[Any]] = []
        self.incidents: list[dict[str, Any]] = []
        self.travel = 0.0
        self.max_over = -math.inf  # worst temperature above the setpoint in force, while in automatic
        self._tripped = False
        self._record()

    @property
    def t(self) -> float:
        return self.k * DT

    def _record(self) -> None:
        sp = self.pid.sp if self.pid.auto else None
        self.frames.append([
            round(self.t, 1), round(self.out_t, 3), round(self.meas, 3), round(self.power, 2),
            None if sp is None else round(sp, 2), round(100 * self.flow, 1), round(self.t_in, 2),
            round(self.th, 3), round(self.vol, 2), None if sp is None else round(self.out_t - sp, 3),
        ])

    def _pipe_outlet(self) -> float:
        target = self.vol - self.p.delay_s
        hv, ht = self.hist_v, self.hist_t
        while self.j + 1 < len(hv) - 1 and hv[self.j + 1] <= target:
            self.j += 1
        a, b = hv[self.j], hv[self.j + 1]
        w = 0.0 if b == a else min(1.0, max(0.0, (target - a) / (b - a)))
        if self.j > 4096:  # drop history the pipe has already delivered
            del hv[: self.j], ht[: self.j]
            self.j = 0
        return ht[self.j] + w * (ht[self.j + 1] - ht[self.j]) if self.j + 1 < len(ht) else ht[-1]

    def advance(self, seconds: float) -> None:
        p = self.p
        a_s = math.exp(-DT / p.sensor_tau_s)
        for _ in range(round(seconds / DT)):
            self.meas = min(SPAN, max(0.0, self.sensor + p.noise_c * self.rng.gauss()))
            before = self.power
            u = self.pid.sample(100.0 * self.meas / SPAN)
            self.power = min(100.0, u * self.flow) if self.ff else u
            u = self.power
            self.travel += abs(u - before)
            a_h = math.exp(-DT * self.flow / p.heater_tau_s)
            t_eq = self.t_in + p.gain_c * u / 100.0 / self.flow
            self.th = t_eq + (self.th - t_eq) * a_h
            self.vol += self.flow * DT
            self.hist_v.append(self.vol)
            self.hist_t.append(self.th)
            self.out_t = self._pipe_outlet()
            self.sensor = self.out_t + (self.sensor - self.out_t) * a_s
            self.k += 1
            if self.pid.auto:
                self.max_over = max(self.max_over, self.out_t - self.pid.sp)
            trip = self.out_t >= T_SAFE
            if trip and not self._tripped:
                self.incidents.append({"t_s": round(self.t, 1), "alarm": "over-temperature", "temp_c": round(self.out_t, 2)})
            self._tripped = trip
            if self.k % FRAME_EVERY == 0:
                self._record()


FRAME_COLUMNS = ["t_s", "outlet_c", "measured_c", "power_pct", "setpoint_c", "flow_pct", "inlet_c", "heater_c",
                 "volume", "deviation_c"]


def _settle(frames: list[list[Any]], t0: float, t1: float) -> float:
    last_out = None
    for f in frames:
        if t0 < f[0] <= t1 and abs(f[9]) > BAND:
            last_out = f[0]
    if last_out is None:
        return 0.0
    if last_out >= t1 - 1.0:
        return math.inf
    return last_out + 1.0 - t0


@dataclass(frozen=True)
class Scenario:
    t_in: float
    sp1: float
    sp2: float
    inlet_drop: float
    window_s: float
    commission_power: float

    @property
    def t1(self) -> float:
        return T_STEP + self.window_s

    @property
    def t2(self) -> float:
        return self.t1 + self.window_s

    @property
    def t_end(self) -> float:
        return self.t2 + self.window_s


def acceptance(plant: Plant, sc: Scenario, tuning: Tuning, feedforward: bool, noise_seed: str) -> dict[str, Any]:
    u0 = 100.0 * (sc.sp1 - sc.t_in) / plant.gain_c
    sim = Sim(plant, sc.t_in, u0, tuning, noise_seed, sp=sc.sp1, feedforward=feedforward)
    sim.advance(T_STEP)
    sim.pid.set_auto(sc.sp2)
    sim.advance(sc.t1 - T_STEP)
    sim.t_in = sc.t_in - sc.inlet_drop
    sim.advance(sc.t2 - sc.t1)
    sim.flow = LOW_FLOW / 100.0
    sim.advance(sc.t_end - sc.t2)
    f = sim.frames
    return {
        "frames": f,
        "max_over_c": sim.max_over,
        "step_s": _settle(f, T_STEP, sc.t1),
        "inlet_s": _settle(f, sc.t1, sc.t2),
        "flow_s": _settle(f, sc.t2, sc.t_end),
        "travel_pct": sim.travel,
    }


def simc_pi(plant: Plant, flow: float = LOW_FLOW / 100.0, feedforward: bool = True) -> Tuning:
    """SIMC PI for the plant at `flow`; with feedforward the gain no longer depends on the flow."""
    k, tau, theta = plant.lumped(flow)
    if feedforward:
        k *= flow
    return Tuning(kc=tau / (k * 2 * theta), ti_s=min(tau, 8 * theta), anti_windup=True)


def _ceil(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


PROMPT = """\
You are commissioning the temperature loop of an inline heater. The plant is simulated, but treat it as a real one.

Plant
- Liquid flows through an electric heater, then through a long pipe to the point of use, where the temperature \
transmitter TT sits. The flow is set by operations: 100 % is nominal, and they will run between 60 and 130 %.
- Transmitter range 0 to 150 degC. The controller works in percent of span: PV = 100 * T / 150 degC; output = \
heater power in percent, 0..100 %. Sample time 0.5 s.
- Controller form: ideal PID, output = Kc * (e + (1/Ti) * integral of e dt) + derivative term, with e = SP - PV in \
percent of span and the derivative acting on the measurement only: -Kc * Td * dPV/dt, filtered with Td/10. Ti = 0 \
switches integral action off. Optional first-order filter on PV (seconds). Optional anti-windup (integration stops \
while the output is limited). On the switch to automatic the integral term starts from the current output; changing \
the tuning in automatic is bumpless.
- Optional Smith predictor: give a model gain Km (percent of span per percent of power), a time constant and a dead \
time in seconds. The controller then acts on PV + (model response without the dead time) - (model response with \
it). Km = 0 switches it off.
- Optional flow feedforward: the controller output is multiplied by the measured flow (as a fraction of nominal) \
before it reaches the heater.
- Heater size, pipe length, time constants, transmitter lag and noise are not given. Find out what you need by \
experimenting on the plant.
- Safety: the outlet must never reach {safe:g} degC.

Now: the controller is in manual, power {power:.1f} %, flow 100 %, the outlet steady at about {temp:.1f} degC, \
inlet {t_in:.1f} degC. Factory tuning: Kc = {kc:g}, Ti = {ti:g} s, no derivative, no filter, anti-windup off, no \
Smith predictor, no flow feedforward.

Acceptance test. It runs after you call submit, with the tuning you leave in the controller:
1. The heater runs steady in automatic at setpoint {sp1:g} degC, flow 100 %, inlet {t_in:.1f} degC. At t = 10 s \
the setpoint steps to {sp2:g} degC.
2. {window:g} s later the inlet temperature drops by 5..8 degC. {window:g} s after that the flow drops to 75 %.
Requirements, on the true outlet temperature:
- never more than {over:g} degC above the setpoint in force;
- inside {sp2:g} +/- {band:g} degC no later than {step} s after the setpoint step, and staying there until the inlet drop;
- after the inlet drop, back inside the band for good within {inlet} s;
- after the flow drop, back inside the band for good within {flow} s;
- total power travel during the test (sum of all controller output changes) at most {travel} %.

You have {budget:g} hours of plant time for commissioning and at most {max_calls} tool calls. Work like a \
commissioning engineer: run experiments, read the trends, tune, check your tuning on the plant, then call submit.
"""


class InlineHeater:
    id = "process.inline_heater"
    domain = "control"
    title = "Inline heater with a long pipe"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        self.plant = Plant(
            gain_c=rng.uniform(50.0, 90.0),
            heater_tau_s=rng.uniform(10.0, 40.0),
            delay_s=rng.uniform(40.0, 150.0),
            sensor_tau_s=rng.uniform(3.0, 10.0),
            noise_c=rng.uniform(0.02, 0.08),
        )
        p = self.plant
        t_in = round(rng.uniform(12.0, 25.0), 1)
        sp1 = float(round(t_in + p.gain_c * rng.uniform(0.35, 0.45)))
        _, tau_w, theta_w = p.lumped(LOW_FLOW / 100.0)
        self.scenario = Scenario(
            t_in=t_in, sp1=sp1, sp2=sp1 + 10.0, inlet_drop=round(rng.uniform(5.0, 8.0), 1),
            window_s=_ceil(14 * theta_w + 4 * tau_w, 100.0), commission_power=round(rng.uniform(25.0, 35.0), 1),
        )
        if seed not in _LIMITS:
            base = simc_pi(p)
            family = [acceptance(p, self.scenario, replace(base, kc=base.kc * a, ti_s=base.ti_s * b), True,
                                 self._noise("acceptance")) for a, b in FAMILY]
            _LIMITS[seed] = {
                "step_s": max(60.0, _ceil(TIME_SLACK * max(r["step_s"] for r in family), 10.0)),
                "inlet_s": max(60.0, _ceil(TIME_SLACK * max(r["inlet_s"] for r in family), 10.0)),
                "flow_s": max(60.0, _ceil(TIME_SLACK * max(r["flow_s"] for r in family), 10.0)),
                "travel_pct": _ceil(TRAVEL_SLACK * max(r["travel_pct"] for r in family), 50.0),
            }
        self.limits = dict(_LIMITS[seed])
        self.sim = Sim(p, t_in, self.scenario.commission_power, FACTORY_TUNING, self._noise("commission"))
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
        sc, lim = self.scenario, self.limits
        return PROMPT.format(
            safe=T_SAFE, power=self.sim.pid.out, temp=self.sim.out_t, t_in=sc.t_in, kc=FACTORY_TUNING.kc,
            ti=FACTORY_TUNING.ti_s, sp1=sc.sp1, sp2=sc.sp2, window=sc.window_s, over=OVER_SP, band=BAND,
            step=f"{lim['step_s']:g}", inlet=f"{lim['inlet_s']:g}", flow=f"{lim['flow_s']:g}",
            travel=f"{lim['travel_pct']:g}", budget=COMMISSION_BUDGET_S / 3600, max_calls=MAX_CALLS,
        )

    def tools(self) -> list[dict[str, Any]]:
        num = {"type": "number"}
        return [
            tool("set_manual", "Switch the controller to manual and set the heater power in percent.",
                 {"power_percent": {**num, "minimum": 0, "maximum": 100}}, ["power_percent"]),
            tool("set_auto", "Switch the controller to automatic with this setpoint in degC (or change the setpoint).",
                 {"setpoint_c": {**num, "minimum": 20, "maximum": 105}}, ["setpoint_c"]),
            tool("set_tuning", "Change controller parameters. Omitted ones keep their value. smith_gain 0 switches "
                 "the Smith predictor off. Returns the full tuning.",
                 {"kc": {**num, "minimum": 0, "maximum": 100}, "ti_s": {**num, "minimum": 0},
                  "td_s": {**num, "minimum": 0}, "pv_filter_s": {**num, "minimum": 0},
                  "anti_windup": {"type": "boolean"}, "smith_gain": {**num, "minimum": 0},
                  "smith_tau_s": {**num, "minimum": 0}, "smith_delay_s": {**num, "minimum": 0},
                  "flow_feedforward": {"type": "boolean"}}),
            tool("set_flow", "Ask operations to set the flow, 60 to 130 percent of nominal.",
                 {"percent": {**num, "minimum": FLOW_RANGE[0], "maximum": FLOW_RANGE[1]}}, ["percent"]),
            tool("wait", "Let the plant run and return the trend: time, measured outlet temperature, power, flow. "
                 "sample_s sets the trend spacing (default: seconds/100, at least 0.5 s, at most 300 rows).",
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
            s.pid.auto = False
            s.pid.out = _num(args, "power_percent", 0.0, 100.0)
            s.power = min(100.0, s.pid.out * s.flow) if s.ff else s.pid.out
            return self._status()
        if name == "set_auto":
            s.pid.set_auto(_num(args, "setpoint_c", 20.0, 105.0))
            return self._status()
        if name == "set_tuning":
            keys = ("kc", "ti_s", "td_s", "pv_filter_s", "smith_gain", "smith_tau_s", "smith_delay_s")
            unknown = set(args) - set(keys) - {"anti_windup", "flow_feedforward"}
            if unknown:
                raise ToolError(f"unknown parameters: {', '.join(sorted(unknown))}")
            t = replace(s.pid.tuning)
            for key in keys:
                if key in args:
                    setattr(t, key, _num(args, key, 0.0, 100.0 if key in ("kc", "smith_gain") else 1e5))
            if "anti_windup" in args:
                if not isinstance(args["anti_windup"], bool):
                    raise ToolError("anti_windup must be true or false")
                t.anti_windup = args["anti_windup"]
            if t.smith_gain > 0 and t.smith_delay_s > 3 * 3600:
                raise ToolError("smith_delay_s is unreasonably long")
            if "flow_feedforward" in args:
                if not isinstance(args["flow_feedforward"], bool):
                    raise ToolError("flow_feedforward must be true or false")
                if args["flow_feedforward"] != s.ff and s.flow > 0:
                    # keep the heater power continuous when feedforward is switched
                    s.pid.out = min(100.0, s.power / s.flow) if args["flow_feedforward"] else s.power
                s.ff = args["flow_feedforward"]
            s.pid.set_tuning(t)
            return self._status()
        if name == "set_flow":
            s.flow = _num(args, "percent", *FLOW_RANGE) / 100.0
            return self._status()
        if name == "wait":
            seconds = _num(args, "seconds", 0.5, MAX_WAIT_S)
            left = COMMISSION_BUDGET_S - s.t
            if left <= 0:
                raise ToolError("commissioning time is used up; call submit")
            n_total = max(1, round(min(seconds, left) / DT))
            sample = args.get("sample_s")
            sample = max(DT, seconds / 100) if sample is None else _num(args, "sample_s", DT, MAX_WAIT_S)
            n_sample = max(1, round(sample / DT), math.ceil(n_total / MAX_TREND_POINTS))
            n_inc = len(s.incidents)
            rows, done = [], 0
            while done < n_total:
                n = min(n_sample, n_total - done)
                s.advance(n * DT)
                done += n
                rows.append([round(s.t, 1), round(s.meas, 3), round(s.power, 2), round(100 * s.flow, 1)])
            out = self._status()
            out["trend"] = {"columns": ["t_s", "temp_c", "power_pct", "flow_pct"], "rows": rows}
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
            "t_s": round(s.t, 1), "mode": "auto" if s.pid.auto else "manual",
            "setpoint_c": s.pid.sp if s.pid.auto else None, "temp_c": round(s.meas, 3),
            "output_pct": round(s.pid.out, 2), "power_pct": round(s.power, 2), "flow_pct": round(100 * s.flow, 1),
            "tuning": {**s.pid.tuning.to_dict(smith=True), "flow_feedforward": s.ff},
            "time_left_s": round(COMMISSION_BUDGET_S - s.t, 1),
        }

    def finish(self) -> Score:
        if self.result is None:
            self.result = acceptance(self.plant, self.scenario, self.sim.pid.tuning, self.sim.ff, self._noise("acceptance"))
        r, lim = self.result, self.limits
        alarms = len(self.sim.incidents)
        over = r["max_over_c"] > OVER_SP
        ratios = {
            "step_ratio": scoring.at_most(r["step_s"], lim["step_s"]),
            "inlet_ratio": scoring.at_most(r["inlet_s"], lim["inlet_s"]),
            "flow_ratio": scoring.at_most(r["flow_s"], lim["flow_s"]),
            "travel_ratio": scoring.at_most(r["travel_pct"], lim["travel_pct"]),
            "commissioning_ratio": 1.0 if alarms == 0 else 0.5,
        }
        metrics = {
            "submitted": self._submitted, "tool_calls": self.calls, "commissioning_s": round(self.sim.t, 1),
            "commissioning_alarms": alarms, "over_temperature": over,
            **{k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items() if k != "frames"},
            "final_tuning": {**self.sim.pid.tuning.to_dict(smith=True), "flow_feedforward": self.sim.ff}, **ratios,
        }
        if over:
            return Score(score=0.0, passed=False, error="over-temperature in the acceptance test", metrics=metrics)
        score = scoring.combine(*ratios.values())
        return Score(score=score, passed=score >= 1.0, metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        tu = result.get("tuning") or {}
        if name == "set_manual":
            return f"manual, power {result['power_pct']:.1f} %", False
        if name == "set_auto":
            return f"automatic, setpoint {result['setpoint_c']:g} degC", False
        if name == "set_flow":
            return f"flow {result['flow_pct']:g} %", False
        if name == "set_tuning":
            ti = f"{tu['ti_s']:.1f} s" if tu["ti_s"] > 0 else "off"
            smith = (f", Smith Km {tu['smith_gain']:.3f} tau {tu['smith_tau_s']:.0f} s delay {tu['smith_delay_s']:.0f} s"
                     if tu["smith_gain"] > 0 else ", no Smith predictor")
            return (f"Kc {tu['kc']:.3f} · Ti {ti} · Td {tu['td_s']:.1f} s · PV filter {tu['pv_filter_s']:.1f} s · "
                    f"anti-windup {'on' if tu['anti_windup'] else 'off'}{smith}, flow feedforward "
                    f"{'on' if tu['flow_feedforward'] else 'off'}"), False
        if name == "wait":
            rows = result["trend"]["rows"]
            tv = [r[1] for r in rows]
            text = (f"plant ran to {result['t_s']:.0f} s: outlet {tv[0]:.2f} -> {tv[-1]:.2f} degC (min {min(tv):.2f}, "
                    f"max {max(tv):.2f}), power {result['power_pct']:.1f} %, {len(rows)} trend rows returned")
            alarms = result.get("alarms") or []
            for a in alarms:
                text += f"; OVER-TEMPERATURE at {a['t_s']:.0f} s ({a['temp_c']:.1f} degC)"
            return text, bool(alarms)
        if name == "submit":
            return "handed over for the acceptance test", False
        return "", False

    def visual(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def view(self) -> dict[str, Any]:
        sc, lim = self.scenario, self.limits
        phases: list[dict[str, Any]] = [
            {"key": "commissioning", "name": "Commissioning", "rows": self.sim.frames, "events": [], "marks": []}
        ]
        if self.result is not None:
            phases.append({
                "key": "acceptance", "name": "Acceptance test", "rows": self.result["frames"],
                "events": [{"t_s": T_STEP, "label": f"SP {sc.sp2:g} degC"},
                           {"t_s": sc.t1, "label": f"inlet -{sc.inlet_drop:g} degC"},
                           {"t_s": sc.t2, "label": "flow 75 %"}],
                "marks": [{"t_s": T_STEP + lim["step_s"], "label": "step limit"},
                          {"t_s": sc.t1 + lim["inlet_s"], "label": "inlet limit"},
                          {"t_s": sc.t2 + lim["flow_s"], "label": "flow limit"}],
            })
        out: dict[str, Any] = {
            "kind": "heater",
            "smooth": True,
            "columns": FRAME_COLUMNS,
            "phases": phases,
            "scene": {"span_c": SPAN, "safe_c": T_SAFE, "pipe_volume": self.plant.delay_s, "band_c": BAND,
                      "t_cold": sc.sp1 - 12.0, "t_hot": sc.sp2 + 6.0,
                      "final_tuning": {**self.sim.pid.tuning.to_dict(smith=True), "flow_feedforward": self.sim.ff}},
            "panels": [
                {"title": "Temperatures", "unit": "°C", "range": [0.0, 120.0], "ticks": [0.0, 30.0, 60.0, 90.0, 120.0],
                 "height": 1.5, "clip": True,
                 "series": [{"col": 6, "label": "inlet", "color": "demand", "dash": True},
                            {"col": 2, "label": "measured", "color": "measured", "width": 1},
                            {"col": 4, "label": "setpoint", "color": "ink", "dash": True},
                            {"col": 1, "label": "outlet (true)", "color": "curve", "width": 2}],
                 "hlines": [{"y": T_SAFE, "label": "safety trip", "color": "critical"}]},
                {"title": "Deviation from setpoint (clipped to ±5 °C)", "unit": "°C", "range": [-5.0, 5.0],
                 "ticks": [-5.0, -2.5, 0.0, 2.5, 5.0], "height": 1.0, "clip": True,
                 "series": [{"col": 9, "label": "deviation", "color": "curve", "width": 1.5}],
                 "bands": {"*": [{"lo": -BAND, "hi": BAND, "label": "±0.5 °C band"}]},
                 "hlines": [{"y": OVER_SP, "label": "+3 °C product limit", "color": "critical"}]},
                {"title": "Heater power and flow", "unit": "%", "range": [0.0, 130.0], "ticks": [0.0, 50.0, 100.0],
                 "height": 1.0,
                 "series": [{"col": 5, "label": "flow", "color": "demand", "dash": True},
                            {"col": 3, "label": "heater power", "color": "output"}]},
            ],
        }
        if self.result is not None:
            out["result"] = self._result_view()
        return out

    def _result_view(self) -> dict[str, Any]:
        r, lim, p, sc = self.result, self.limits, self.plant, self.scenario
        alarms = len(self.sim.incidents)
        secs = lambda v: "never" if math.isinf(v) else f"{v:g} s"  # noqa: E731
        reqs = [
            ("Never more than 3 °C above the setpoint", f"{r['max_over_c']:+.2f} °C", f"+{OVER_SP:g} °C",
             r["max_over_c"] <= OVER_SP),
            (f"Inside {sc.sp2:g} ± 0.5 °C after the setpoint step", secs(r["step_s"]), f"{lim['step_s']:g} s",
             r["step_s"] <= lim["step_s"]),
            ("Back in the band after the inlet drop", secs(r["inlet_s"]), f"{lim['inlet_s']:g} s", r["inlet_s"] <= lim["inlet_s"]),
            ("Back in the band after the flow drop", secs(r["flow_s"]), f"{lim['flow_s']:g} s", r["flow_s"] <= lim["flow_s"]),
            ("Power travel", f"{r['travel_pct']:,.0f} %", f"{lim['travel_pct']:,.0f} %", r["travel_pct"] <= lim["travel_pct"]),
            ("Over-temperature trips while commissioning", str(alarms), "0", alarms == 0),
        ]
        ft, rt = self.sim.pid.tuning, simc_pi(p)
        k, tau, theta = p.lumped(1.0)
        kw, tauw, thetaw = p.lumped(LOW_FLOW / 100.0)
        ti = lambda x: f"{x:.0f} s" if x > 0 else "off"  # noqa: E731
        return {
            "requirements": [{"label": a, "measured": b, "limit": c, "ok": bool(d)} for a, b, c, d in reqs],
            "hidden": [
                ["Pipe delay", f"{p.delay_s:.0f} s at 100 % flow, {p.delay_s / 0.75:.0f} s at 75 %"],
                ["Heater time constant", f"{p.heater_tau_s:.0f} s at 100 % flow"],
                ["Gain", f"{k:.3f} %/% at 100 % flow, {kw:.3f} at 75 %"],
                ["Transmitter lag / noise", f"{p.sensor_tau_s:.1f} s / {p.noise_c:.3f} °C"],
                ["Lumped model at 75 % flow", f"time constant {tauw:.0f} s, dead time {thetaw:.0f} s"],
            ],
            "final": {
                "title": "Final tuning", "columns": ["", "agent", "reference: feedforward + SIMC at 75 % flow"],
                "rows": [
                    ["Kc", f"{ft.kc:.3f}", f"{rt.kc:.3f}"], ["Ti", ti(ft.ti_s), ti(rt.ti_s)],
                    ["Td", f"{ft.td_s:.1f} s", "0.0 s"], ["PV filter", f"{ft.pv_filter_s:.1f} s", "0.0 s"],
                    ["Anti-windup", "on" if ft.anti_windup else "off", "on"],
                    ["Smith predictor", (f"Km {ft.smith_gain:.3f}, {ft.smith_tau_s:.0f} s, {ft.smith_delay_s:.0f} s"
                                         if ft.smith_gain > 0 else "off"), "off"],
                    ["Flow feedforward", "on" if self.sim.ff else "off", "on"],
                ],
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _identify(ep: Any, out: float, du: float) -> tuple[float, float, float]:
    """Step test in manual from controller output `out`: (gain %/%, time constant s, dead time s)."""
    before = ep.tool_call("wait", {"seconds": 120, "sample_s": 1})["trend"]["rows"]
    y0 = _mean([100 * r[1] / SPAN for r in before[-60:]])
    t_step = before[-1][0]
    ep.tool_call("set_manual", {"power_percent": round(out + du, 2)})
    rows = ep.tool_call("wait", {"seconds": 450, "sample_s": 1.5})["trend"]["rows"]
    rows += ep.tool_call("wait", {"seconds": 2400, "sample_s": 10})["trend"]["rows"]
    ts = [r[0] - t_step for r in rows]
    ys = [100 * r[1] / SPAN for r in rows]
    dy = _mean(ys[-25:]) - y0
    smooth = [_mean(ys[max(0, i - 2): i + 3]) for i in range(len(ys))]

    def crossing(frac: float) -> float:
        target = y0 + frac * dy
        for i in range(1, len(smooth)):
            if (smooth[i] - target) * (1 if dy > 0 else -1) >= 0:
                a, b = smooth[i - 1], smooth[i]
                w = 0.0 if b == a else (target - a) / (b - a)
                return ts[i - 1] + w * (ts[i] - ts[i - 1])
        return ts[-1]

    t10, t63 = crossing(0.10), crossing(0.632)
    tau = (t63 - t10) / (1 + math.log(0.9))
    return dy / du, tau, max(DT, t63 - tau)


def reference_agent(ep: Any) -> None:
    sc = ep.env.scenario
    ep.model_turn("The test cuts the flow to 75 % with the heater on. At constant power that raises the outlet "
                  "temperature by a third of the heating rise, and the transmitter only sees it one pipe delay later, "
                  "so feedback alone cannot hold 3 degrees. I switch on flow feedforward, so power follows the flow, "
                  "and identify the plant at 75 % flow, where the pipe delay and the heater lag are longest.")
    ep.tool_call("set_tuning", {"flow_feedforward": True})
    ep.tool_call("set_flow", {"percent": 75})
    ep.tool_call("wait", {"seconds": 1500, "sample_s": 10})
    out = ep.tool_call("wait", {"seconds": 1, "sample_s": 1})["output_pct"]
    ep.model_turn("Steady again. Now a +10 % step on the controller output in manual: fine sampling first to see the "
                  "dead time, then a long record for gain and time constant.")
    k, tau, theta = _identify(ep, out, 10.0)
    kc, ti = tau / (k * 2 * theta), min(tau, 8 * theta)
    ep.model_turn(
        f"First order plus dead time at 75 % flow: gain {k:.3f} %/%, time constant {tau:.0f} s, dead time {theta:.0f} s. "
        f"The delay dominates, so a cautious SIMC PI with tc equal to the dead time: Kc = {kc:.3f}, Ti = {ti:.0f} s, "
        "anti-windup on. With feedforward the gain stays the same at any flow, and tuning for the longest delay keeps "
        "the loop stable everywhere."
    )
    ep.tool_call("set_tuning", {"kc": round(kc, 4), "ti_s": round(ti, 1), "td_s": 0, "pv_filter_s": 0, "anti_windup": True})
    ep.tool_call("set_flow", {"percent": 100})
    ep.tool_call("set_auto", {"setpoint_c": sc.sp1})
    ep.tool_call("wait", {"seconds": 2000, "sample_s": 10})
    ep.model_turn(f"Holding {sc.sp1:g} degC. Now the setpoint step to {sc.sp2:g} degC and the flow drop, as in the test.")
    ep.tool_call("set_auto", {"setpoint_c": sc.sp2})
    ep.tool_call("wait", {"seconds": 2400, "sample_s": 10})
    ep.tool_call("set_flow", {"percent": 75})
    ep.tool_call("wait", {"seconds": 1200, "sample_s": 6})
    ep.model_turn("No overshoot to speak of on the step, and the flow drop barely shows. Submitting.")
    ep.tool_call("submit", {})


def factory_agent(ep: Any) -> None:
    ep.model_turn("The controller has a factory tuning. I submit it as it is.")
    ep.tool_call("submit", {})


def no_feedforward_agent(ep: Any) -> None:
    ep.model_turn("A careful SIMC PI tuned at 75 % flow, the slow case, should be robust for the whole test.")
    ep.tool_call("set_flow", {"percent": 75})
    ep.tool_call("wait", {"seconds": 1500, "sample_s": 10})
    out = ep.tool_call("wait", {"seconds": 1, "sample_s": 1})["output_pct"]
    k, tau, theta = _identify(ep, out, 10.0)
    kc, ti = tau / (k * 2 * theta), min(tau, 8 * theta)
    ep.model_turn(f"Gain {k:.3f} %/%, time constant {tau:.0f} s, dead time {theta:.0f} s: Kc = {kc:.3f}, Ti = {ti:.0f} s.")
    ep.tool_call("set_tuning", {"kc": round(kc, 4), "ti_s": round(ti, 1), "anti_windup": True})
    ep.tool_call("set_flow", {"percent": 100})
    ep.tool_call("set_auto", {"setpoint_c": ep.env.scenario.sp1})
    ep.tool_call("wait", {"seconds": 1500, "sample_s": 10})
    ep.model_turn("Stable and smooth. Submitting.")
    ep.tool_call("submit", {})


def lag_only_agent(ep: Any) -> None:
    ep.model_turn("I turn on flow feedforward and tune on the heater time constant, which is what the controller "
                  "acts on directly.")
    ep.tool_call("set_tuning", {"flow_feedforward": True})
    out = ep.tool_call("wait", {"seconds": 1, "sample_s": 1})["output_pct"]
    k, tau, theta = _identify(ep, out, 10.0)
    ep.model_turn(f"Gain {k:.3f} %/%, time constant {tau:.0f} s. Pole cancellation: Ti = {tau:.0f} s, Kc = 1/K = {1 / k:.2f}.")
    ep.tool_call("set_tuning", {"kc": round(1 / k, 4), "ti_s": round(tau, 1), "anti_windup": True})
    ep.tool_call("set_auto", {"setpoint_c": ep.env.scenario.sp1})
    ep.model_turn("Submitting.")
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "factory": factory_agent, "no_feedforward": no_feedforward_agent,
          "lag_only": lag_only_agent}
NEGATIVE_AGENTS = ("factory", "no_feedforward", "lag_only")
