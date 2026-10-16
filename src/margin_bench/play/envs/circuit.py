"""electronics.fault_finding: find and fix the fault in a transistor amplifier.

A common-emitter amplifier (NPN transistor, divider bias, emitter resistor
with a bypass capacitor, coupling capacitors, a load) came back from the field
not working. Exactly one part is faulty: a resistor open, shorted or of the
wrong value, a capacitor open or shorted, or the transistor with an open
junction or a collector-emitter short. Part values come from the seed and are
on the schematic; real parts are within 1 % of them.

The model works like a repair technician: it measures DC node voltages with a
multimeter, measures the gain at 1 kHz with a signal generator and a scope,
lifts parts out of the circuit to test them on their own, and replaces parts.
ngspice simulates the board for every measurement. When it submits, the board
is tested against its spec.

Grading: the board passes when it meets its spec (collector voltage and gain
within 10 % of the design) and no good part was replaced. Each needless
replacement halves the score.

Difficulty rule: the faults are drawn evenly from the fault list; each was
checked to make the board fail its spec. Finding it takes reading the bias
voltages against the design (or the gain, for the capacitor faults that leave
the DC point alone), then confirming on the lifted part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from margin_bench.core import Score
from margin_bench.play.env import ToolError, tool
from margin_bench.rng import SeededRandom
from margin_bench.sims.ngspice import Ngspice, fmt

E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)
NODES = ("vcc", "in", "b", "c", "e", "out")
RESISTORS = ("R1", "R2", "RC", "RE", "RL")
CAPACITORS = ("Cin", "CE", "Cout")
SPEC_TOL = 0.10
MAX_CALLS = 40
F_TEST = 1000.0
METER = 1e-3  # V, multimeter resolution

# (part, mode): the fault list
FAULTS = (
    ("R1", "open"), ("R2", "open"), ("RC", "open"), ("RE", "open"), ("RE", "short"),
    ("RC", "x10"), ("R2", "x10"), ("CE", "open"), ("CE", "short"), ("Cin", "open"), ("Cout", "open"),
    ("Q1", "be_open"), ("Q1", "ce_short"), ("Q1", "bc_open"),
)


def e12(x: float) -> float:
    d = 10 ** math.floor(math.log10(x))
    return min((v * d for v in E12 + (10.0,)), key=lambda v: abs(math.log(v / x)))


@dataclass
class Board:
    vcc: float
    nominal: dict[str, float]  # schematic values
    actual: dict[str, float]  # within 1 %
    fault: tuple[str, str]
    replaced: set[str] = field(default_factory=set)

    def netlist(self, analysis: str) -> list[str]:
        a, part_fault = self.actual, (None if self.fault[0] in self.replaced else self.fault)

        def r(name: str, n1: str, n2: str) -> str:
            v = a[name]
            if part_fault and part_fault[0] == name:
                v = {"open": 1e12, "short": 0.01, "x10": v * 10}[part_fault[1]]
            return f"{name} {n1} {n2} {fmt(v)}"

        def c(name: str, n1: str, n2: str) -> str:
            if part_fault and part_fault[0] == name:
                return f"{name} {n1} {n2} {fmt(1e-15)}" if part_fault[1] == "open" else f"R{name} {n1} {n2} 0.01"
            return f"{name} {n1} {n2} {fmt(a[name])}"

        lines = ["amplifier", f"VCC vcc 0 DC {fmt(self.vcc)}", "VIN in 0 DC 0 AC 1",
                 r("R1", "vcc", "b"), r("R2", "b", "0"), r("RC", "vcc", "c"), r("RE", "e", "0"), r("RL", "out", "0"),
                 c("Cin", "in", "b"), c("CE", "e", "0"), c("Cout", "c", "out")]
        qf = part_fault[1] if part_fault and part_fault[0] == "Q1" else None
        qb, qc = ("bx" if qf == "be_open" else "b"), ("cx" if qf == "bc_open" else "c")
        lines.append(f"Q1 {qc} {qb} e QN")
        if qf == "be_open":
            lines.append("RQB bx b 1e12")
        if qf == "bc_open":
            lines.append("RQC cx c 1e12")
        if qf == "ce_short":
            lines.append("RQS c e 0.05")
        lines += [".model QN NPN(IS=1e-14 BF=160 VAF=100 CJE=8p CJC=4p TF=0.4n)", analysis, ".end"]
        return lines

    def dc(self) -> dict[str, float]:
        r = Ngspice.get().run(self.netlist(".op"), [f"v({n})" for n in NODES])
        return {n: float(r[f"v({n})"].real[0]) for n in NODES}

    def gain(self, f_hz: float = F_TEST) -> float:
        r = Ngspice.get().run(self.netlist(f".ac lin 1 {fmt(f_hz)} {fmt(f_hz)}"), ["v(out)"])
        return float(abs(r["v(out)"][0]))


def design(rng: SeededRandom) -> tuple[float, dict[str, float]]:
    vcc = rng.choice((9.0, 12.0, 15.0))
    ic = rng.uniform(0.6e-3, 3e-3)
    re = e12(rng.uniform(0.1, 0.15) * vcc / ic)
    ve = re * ic
    rc = e12(0.45 * vcc / ic)
    vb = ve + 0.65
    i_div = 12 * ic / 160
    r2 = e12(vb / i_div)
    r1 = e12((vcc - vb) / (i_div + ic / 160))
    rl = e12(rng.uniform(2.0, 5.0) * rc)
    return vcc, {"R1": r1, "R2": r2, "RC": rc, "RE": re, "RL": rl, "Cin": 10e-6, "CE": 100e-6, "Cout": 10e-6}


def _si(v: float, unit: str) -> str:
    for scale, prefix in ((1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n")):
        if abs(v) >= scale:
            return f"{v / scale:.4g} {prefix}{unit}"
    return f"{v:.3g} {unit}"


PROMPT = """\
You are repairing an amplifier board that came back from the field. The board is simulated; treat it as a real one.

Schematic (common-emitter amplifier, NPN transistor Q1):
- supply: VCC = {vcc:g} V from node vcc to ground;
- bias divider: R1 = {R1} from vcc to b, R2 = {R2} from b to ground;
- collector resistor RC = {RC} from vcc to c; emitter resistor RE = {RE} from e to ground, bypassed by CE = {CE};
- input: signal generator at node in, coupled to the base by Cin = {Cin};
- output: from the collector through Cout = {Cout} to node out, loaded by RL = {RL} to ground;
- Q1: collector c, base b, emitter e; a small-signal NPN, current gain around 150.
Parts are within 1 % of their marked values.

The board does not meet its spec. A good board has: collector DC voltage within 10 % of its design value, and a \
voltage gain from in to out at 1 kHz within 10 % of its design value. Exactly one part is faulty.

Tools: a multimeter for DC node voltages (to ground), a signal generator and scope for the gain at a frequency, an \
out-of-circuit test for any single part (it is lifted, tested and put back), and replacement of a part with a new \
good one. Replacing a good part is a mistake: each needless replacement halves the score. At most {calls} tool calls.

Work like a repair technician: measure, work out what the bias should be, find the faulty part, confirm it, \
replace it, check the board, then call submit.
"""


class FaultFinding:
    id = "electronics.fault_finding"
    domain = "electronics"
    title = "Amplifier with a fault"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        rng = SeededRandom(seed)
        vcc, nominal = design(rng)
        actual = {k: v * (1 + rng.uniform(-0.01, 0.01)) for k, v in nominal.items()}
        fault = FAULTS[rng.integer(0, len(FAULTS) - 1)]
        self.board = Board(vcc=vcc, nominal=nominal, actual=actual, fault=fault)
        good = Board(vcc=vcc, nominal=nominal, actual=nominal, fault=("none", "none"))
        dc, g = good.dc(), good.gain()
        self.spec = {"vc": dc["c"], "gain": g}
        self.calls = 0
        self.log: list[list[Any]] = []
        self.replacements: list[str] = []
        self.measured: dict[str, float] = {}
        self._submitted = False
        self.result: dict[str, Any] | None = None
        self._record()

    @property
    def submitted(self) -> bool:
        return self._submitted

    @property
    def sim_time(self) -> float:
        return float(self.calls)

    def _state(self) -> tuple[float, float]:
        return self.board.dc()["c"], self.board.gain()

    def _record(self) -> None:
        vc, g = self._state()
        self.log.append([float(self.calls), round(vc, 4), round(g, 3), len(self.replacements)])

    def prompt(self) -> str:
        n = self.board.nominal
        vals = {k: _si(v, "F" if k.startswith("C") else "ohm") for k, v in n.items()}
        return PROMPT.format(vcc=self.board.vcc, calls=MAX_CALLS, **vals)

    def tools(self) -> list[dict[str, Any]]:
        parts = list(RESISTORS) + list(CAPACITORS) + ["Q1"]
        return [
            tool("measure_dc", "Multimeter: DC voltage of a node to ground (resolution 1 mV).",
                 {"node": {"type": "string", "enum": list(NODES)}}, ["node"]),
            tool("measure_gain", "Signal generator and scope: voltage gain from in to out at this frequency.",
                 {"frequency_hz": {"type": "number", "minimum": 1, "maximum": 1e6}}, ["frequency_hz"]),
            tool("test_part", "Lift a part out of the circuit and test it on its own: resistance for resistors, "
                 "capacitance for capacitors, junction diode drops for the transistor.",
                 {"part": {"type": "string", "enum": parts}}, ["part"]),
            tool("replace_part", "Replace a part with a new good one of the marked value.",
                 {"part": {"type": "string", "enum": parts}}, ["part"]),
            tool("submit", "Finish the repair. The board is then tested against its spec."),
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise ToolError("already submitted")
        self.calls += 1
        if self.calls > MAX_CALLS and name != "submit":
            raise ToolError(f"tool call limit ({MAX_CALLS}) reached; call submit")
        b = self.board
        parts = set(RESISTORS) | set(CAPACITORS) | {"Q1"}
        out: dict[str, Any]
        if name == "measure_dc":
            node = args.get("node")
            if node not in NODES:
                raise ToolError(f"node must be one of {', '.join(NODES)}")
            v = round(b.dc()[node] / METER) * METER
            self.measured[node] = v
            out = {"node": node, "volts": round(v, 3)}
        elif name == "measure_gain":
            f = args.get("frequency_hz")
            if isinstance(f, bool) or not isinstance(f, (int, float)) or not 1 <= f <= 1e6:
                raise ToolError("frequency_hz must be a number from 1 to 1e6")
            out = {"frequency_hz": f, "gain": round(b.gain(float(f)), 3)}
        elif name == "test_part":
            part = args.get("part")
            if part not in parts:
                raise ToolError(f"part must be one of {', '.join(sorted(parts))}")
            out = {"part": part, **self._test(part)}
        elif name == "replace_part":
            part = args.get("part")
            if part not in parts:
                raise ToolError(f"part must be one of {', '.join(sorted(parts))}")
            b.replaced.add(part)
            b.actual[part] = b.nominal.get(part, 0.0)
            self.replacements.append(part)
            out = {"part": part, "replaced": True, "replacements_so_far": len(self.replacements)}
        elif name == "submit":
            self._submitted = True
            out = {"submitted": True}
        else:
            raise ToolError(f"unknown tool {name!r}")
        self._record()
        return out

    def _test(self, part: str) -> dict[str, Any]:
        b = self.board
        faulty = b.fault[0] == part and part not in b.replaced
        mode = b.fault[1] if faulty else None
        if part in RESISTORS:
            v = b.actual[part]
            if mode == "open":
                return {"reading": "OL (open)"}
            if mode == "short":
                return {"reading": "0.01 ohm"}
            if mode == "x10":
                v *= 10
            return {"reading": _si(v, "ohm"), "ohms": round(v, 2)}
        if part in CAPACITORS:
            if mode == "open":
                return {"reading": "no capacitance (open)"}
            if mode == "short":
                return {"reading": "short (0.01 ohm)"}
            return {"reading": _si(b.actual[part], "F")}
        diodes = {"base-emitter": "0.65 V", "base-collector": "0.65 V", "collector-emitter": "open both ways"}
        if mode == "be_open":
            diodes["base-emitter"] = "open"
        if mode == "bc_open":
            diodes["base-collector"] = "open"
        if mode == "ce_short":
            diodes["collector-emitter"] = "short (0.05 ohm)"
        return {"diode_test": diodes}

    def finish(self) -> Score:
        if self.result is None:
            vc, g = self._state()
            fixed = abs(vc - self.spec["vc"]) <= SPEC_TOL * self.spec["vc"] and abs(g - self.spec["gain"]) <= SPEC_TOL * self.spec["gain"]
            wasted = [p for p in self.replacements if p != self.board.fault[0]]
            self.result = {"vc": vc, "gain": g, "fixed": fixed, "wasted": wasted}
        r = self.result
        score = (0.5 ** len(r["wasted"])) if r["fixed"] else 0.0
        metrics = {"submitted": self._submitted, "tool_calls": self.calls, "fault": list(self.board.fault),
                   "replacements": self.replacements, "wasted_replacements": len(r["wasted"]), "fixed": r["fixed"],
                   "vc": r["vc"], "gain": r["gain"], "spec_vc": self.spec["vc"], "spec_gain": self.spec["gain"]}
        return Score(score=score, passed=bool(r["fixed"] and not r["wasted"]), metrics=metrics)

    def describe(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> tuple[str, bool]:
        if name == "measure_dc":
            return f"V({result['node']}) = {result['volts']:.3f} V", False
        if name == "measure_gain":
            return f"gain at {result['frequency_hz']:g} Hz = {result['gain']:.3f}", False
        if name == "test_part":
            if "diode_test" in result:
                return result["part"] + ": " + ", ".join(f"{k} {v}" for k, v in result["diode_test"].items()), False
            return f"{result['part']} out of circuit: {result['reading']}", False
        if name == "replace_part":
            return f"{result['part']} replaced (replacement {result['replacements_so_far']})", False
        if name == "submit":
            return "handed over for the board test", False
        return "", False

    def visual(self, name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def view(self) -> dict[str, Any]:
        b = self.board
        phases: list[dict[str, Any]] = [{"key": "commissioning", "name": "Repair", "rows": self.log, "events": [], "marks": []}]
        if self.result is not None:
            last = self.log[-1]
            phases.append({"key": "acceptance", "name": "Board test",
                           "rows": [[0.0] + last[1:], [1.0] + last[1:]], "events": [], "marks": []})
        vc, g = self.spec["vc"], self.spec["gain"]
        out: dict[str, Any] = {
            "kind": "circuit",
            "speed": 1,
            "time_unit": "step",
            "columns": ["step", "vc_v", "gain", "replacements"],
            "phases": phases,
            "scene": {"vcc": b.vcc, "nominal": {k: _si(v, "F" if k.startswith("C") else "Ω") for k, v in b.nominal.items()},
                      "spec_vc": vc, "spec_gain": g, "fault": list(b.fault) if self.result is not None else None},
            "panels": [
                {"title": "Collector DC voltage (true, after each step)", "unit": "V", "range": [0.0, b.vcc],
                 "ticks": [0.0, round(b.vcc / 2, 1), b.vcc], "height": 1.0,
                 "series": [{"col": 1, "label": "collector voltage", "color": "curve", "width": 2}],
                 "bands": {"*": [{"lo": vc * (1 - SPEC_TOL), "hi": vc * (1 + SPEC_TOL), "label": "spec ±10 %"}]}},
                {"title": "Gain at 1 kHz (true, after each step)", "unit": "", "range": [0.0, max(1.5 * g, 1.0)],
                 "ticks": [0.0, round(g, 1), round(1.5 * g, 1)], "height": 1.0,
                 "series": [{"col": 2, "label": "gain", "color": "output", "width": 2}],
                 "bands": {"*": [{"lo": g * (1 - SPEC_TOL), "hi": g * (1 + SPEC_TOL), "label": "spec ±10 %"}]}},
            ],
        }
        if self.result is not None:
            r = self.result
            out["result"] = {
                "requirements": [
                    {"label": "Collector DC voltage within 10 % of the design", "measured": f"{r['vc']:.2f} V",
                     "limit": f"{vc:.2f} V ± 10 %", "ok": abs(r["vc"] - vc) <= SPEC_TOL * vc},
                    {"label": "Gain at 1 kHz within 10 % of the design", "measured": f"{r['gain']:.1f}",
                     "limit": f"{g:.1f} ± 10 %", "ok": abs(r["gain"] - g) <= SPEC_TOL * g},
                    {"label": "No good part replaced", "measured": ", ".join(r["wasted"]) or "none", "limit": "none",
                     "ok": not r["wasted"]},
                ],
                "hidden": [["The fault", f"{b.fault[0]}: {b.fault[1].replace('_', ' ')}"],
                           ["Replaced", ", ".join(self.replacements) or "nothing"]],
                "final": {"title": "Design values", "columns": ["", "value"],
                          "rows": [[k, _si(v, "F" if k.startswith("C") else "Ω")] for k, v in b.nominal.items()]
                          + [["Collector voltage", f"{vc:.2f} V"], ["Gain at 1 kHz", f"{g:.1f}"]]},
            }
        return out


# Scripted agents. They work only through the tools, like a model, and narrate.


def reference_agent(ep: Any) -> None:
    env = ep.env
    b = env.board
    ep.model_turn("First the DC picture: every node with the multimeter, then the gain. A healthy divider-biased stage "
                  "has the base about 0.65 V above the emitter and the collector near the middle of the supply.")
    meas = {n: ep.tool_call("measure_dc", {"node": n})["volts"] for n in ("vcc", "b", "e", "c")}
    gain = ep.tool_call("measure_gain", {"frequency_hz": 1000})["gain"]
    # fault dictionary: simulate the board with each candidate fault and compare with the measurements
    best, scores = None, []
    for cand in FAULTS:
        trial = Board(vcc=b.vcc, nominal=b.nominal, actual=dict(b.nominal), fault=cand)
        dc, g = trial.dc(), trial.gain()
        err = sum(abs(dc[n] - meas[n]) for n in ("b", "e", "c")) + 0.2 * abs(math.log((g + 1e-3) / (gain + 1e-3)))
        scores.append((err, cand))
    scores.sort()
    ep.model_turn(f"Measured: base {meas['b']:.2f} V, emitter {meas['e']:.2f} V, collector {meas['c']:.2f} V, gain "
                  f"{gain:.1f}. Comparing with what each possible fault would do, the best match is {scores[0][1][0]} "
                  f"({scores[0][1][1].replace('_', ' ')}). I lift it and test it before replacing anything.")
    for err, cand in scores[:4]:
        res = ep.tool_call("test_part", {"part": cand[0]})
        if "diode_test" in res:
            d = res["diode_test"]
            bad = (d["base-emitter"], d["base-collector"], d["collector-emitter"]) != ("0.65 V", "0.65 V", "open both ways")
        elif "ohms" in res:
            bad = abs(res["ohms"] / b.nominal[cand[0]] - 1) > 0.05
        else:
            bad = any(w in res["reading"] for w in ("open", "short", "OL"))
        if bad:
            best = cand[0]
            break
    if best is None:
        best = scores[0][1][0]
    ep.model_turn(f"{best} tests bad. Replacing it and checking the board again.")
    ep.tool_call("replace_part", {"part": best})
    ep.tool_call("measure_dc", {"node": "c"})
    ep.tool_call("measure_gain", {"frequency_hz": 1000})
    ep.model_turn("Collector voltage and gain are back where the design puts them. Submitting.")
    ep.tool_call("submit", {})


def no_fix_agent(ep: Any) -> None:
    ep.model_turn("The supply is there and the transistor looks fine on paper. I return the board as it is.")
    ep.tool_call("measure_dc", {"node": "vcc"})
    ep.tool_call("submit", {})


def shotgun_agent(ep: Any) -> None:
    ep.model_turn("Quickest repair: replace every part on the board.")
    for part in list(RESISTORS) + list(CAPACITORS) + ["Q1"]:
        ep.tool_call("replace_part", {"part": part})
    ep.tool_call("submit", {})


AGENTS = {"reference": reference_agent, "no_fix": no_fix_agent, "shotgun": shotgun_agent}
NEGATIVE_AGENTS = ("no_fix", "shotgun")
