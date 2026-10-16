"""Playground: the scripted reference passes, the known-bad agents fail, and the model loop is wired right."""

from __future__ import annotations

import json

import pytest

from margin_bench.harness.client import ApiError, ChatTurn, Settings
from margin_bench.harness.openrouter import build_chat_body, parse_chat
from margin_bench.play import NEGATIVE, SCRIPTED, make, serve
from margin_bench.play.agent import run_model
from margin_bench.play.env import ToolError
from margin_bench.play.episode import Episode

ENV = "process.tank_level"
ENVS = sorted(SCRIPTED)
SEEDS = range(20)


def _scripted(name: str, seed: int, env_id: str = ENV) -> Episode:
    env = make(env_id, seed)
    ep = Episode(env, {"kind": "scripted", "name": name})
    SCRIPTED[env_id][name](ep)
    ep.finish(env.finish(), stop_reason="submitted")
    return ep


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("seed", SEEDS)
def test_reference_engineer_passes(env_id, seed):
    ep = _scripted("reference", seed, env_id)
    r = ep.data["result"]
    assert r["passed"] and r["score"] == 1.0, r["metrics"]
    assert not [s for s in ep.data["steps"] if s.get("error")]


@pytest.mark.parametrize("env_id,name", [(e, n) for e in ENVS for n in NEGATIVE[e]])
@pytest.mark.parametrize("seed", SEEDS)
def test_known_bad_agents_fail(env_id, name, seed):
    r = _scripted(name, seed, env_id).data["result"]
    assert not r["passed"] and r["score"] < 1.0


@pytest.mark.parametrize("env_id", ENVS)
def test_view_has_what_the_viewer_needs(env_id):
    ep = _scripted("reference", 1, env_id)
    v = ep.data["view"]
    assert [p["key"] for p in v["phases"]] == ["commissioning", "acceptance"]
    cols = len(v["columns"])
    assert all(len(row) == cols for p in v["phases"] for row in p["rows"])
    assert all(s["col"] < cols for panel in v["panels"] for s in panel["series"])
    assert all(r["ok"] for r in v["result"]["requirements"])
    tools = [s for s in ep.data["steps"] if s["kind"] == "tool"]
    assert all(s["summary"] for s in tools)
    json.dumps(ep.data)  # everything is plain JSON


@pytest.mark.parametrize("env_id", ENVS)
def test_episode_is_reproducible(env_id):
    a, b = _scripted("reference", 3, env_id), _scripted("reference", 3, env_id)
    assert a.data["result"] == b.data["result"]
    assert a.data["view"]["phases"] == b.data["view"]["phases"]


def test_prompt_states_the_limits():
    env = make(ENV, 0)
    p = env.prompt()
    for key in ("start_s", "recovery_s", "travel_pct"):
        assert f"{env.limits[key]:g}" in p
    assert f"{env.limits['overshoot_m']:.3f}" in p
    # nothing about the hidden plant leaks into the prompt
    assert f"{env.plant.delay:.1f}" not in p


def test_tools_refuse_bad_calls():
    env = make(ENV, 0)
    with pytest.raises(ToolError):
        env.call("set_manual", {"valve_percent": 140})
    with pytest.raises(ToolError):
        env.call("set_tuning", {"kc": "fast"})
    with pytest.raises(ToolError):
        env.call("set_tuning", {"gain": 2})
    with pytest.raises(ToolError):
        env.call("set_outlet_demand", {"percent": 95})
    with pytest.raises(ToolError):
        env.call("open_drain", {})
    r = env.call("wait", {"seconds": 1800, "sample_s": 0.5})
    assert len(r["trend"]["rows"]) <= 300 and r["t_s"] == 1800.0
    env.call("submit", {})
    with pytest.raises(ToolError):
        env.call("wait", {"seconds": 10})


def test_call_limit_still_allows_submit():
    env = make(ENV, 0)
    for _ in range(40):
        env.call("set_outlet_demand", {"percent": 50})
    with pytest.raises(ToolError):
        env.call("set_outlet_demand", {"percent": 50})
    env.call("submit", {})
    assert env.submitted


def test_spill_during_commissioning_is_an_incident():
    env = make(ENV, 0)
    env.call("set_manual", {"valve_percent": 100})
    r = env.call("wait", {"seconds": 1800})
    assert r["alarms"][0]["alarm"] == "high-high"
    s = env.finish()
    assert not s.passed and s.metrics["commissioning_alarms"] >= 1


class FakeModel:
    """Plays a fixed list of turns: (text, [(tool, arguments as JSON text)])."""

    def __init__(self, turns, fail_at=None):
        self.turns, self.i, self.fail_at, self.seen = turns, 0, fail_at, []

    def chat(self, messages, tools, settings):
        self.seen.append([dict(m) for m in messages])
        if self.i == self.fail_at:
            raise ApiError("HTTP 503")
        text, calls = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        tc = [{"id": f"c{self.i}{k}", "name": n, "arguments": a} for k, (n, a) in enumerate(calls)]
        echo = {"role": "assistant", "content": text}
        if tc:
            echo["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in tc]
        return ChatTurn(message=echo, text=text, tool_calls=tc, latency_s=1.5, prompt_tokens=100,
                        completion_tokens=20, reasoning_tokens=5, cost_usd=0.01)


def test_model_loop_runs_tools_and_records_everything(tmp_path):
    env = make(ENV, 0)
    model = FakeModel([
        ("look first", [("wait", json.dumps({"seconds": 60}))]),
        ("bad json", [("set_manual", "{valve: 50")]),
        ("tune", [("set_tuning", json.dumps({"kc": 4.7, "ti_s": 71, "anti_windup": True})),
                  ("set_auto", json.dumps({"setpoint_m": 1.2}))]),
        ("done", [("submit", "{}"), ("wait", json.dumps({"seconds": 5}))]),
    ])
    path = tmp_path / "ep.json"
    ep = Episode(env, {"kind": "model", "name": "fake"}, path)
    run_model(env, model, Settings(model="fake"), ep)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["status"] == "done" and d["stop_reason"] == "submitted"
    assert d["result"]["metrics"]["final_tuning"]["kc"] == 4.7
    tools = [s for s in d["steps"] if s["kind"] == "tool"]
    assert [s["name"] for s in tools] == ["wait", "set_manual", "set_tuning", "set_auto", "submit"]
    assert tools[1]["error"].startswith("could not parse")
    assert d["usage"]["model_calls"] == 4 and d["usage"]["cost_usd"] == pytest.approx(0.04)
    # every tool call got a tool message back, with the matching id
    last = model.seen[-1]
    replies = [m for m in last if m["role"] == "tool"]
    assert len(replies) == 4 and all(r["tool_call_id"].startswith("c") for r in replies)


def test_model_loop_nudges_then_stops_when_no_tools(tmp_path):
    env = make(ENV, 0)
    ep = Episode(env, {"kind": "model", "name": "fake"}, tmp_path / "ep.json")
    run_model(env, FakeModel([("I think the loop is fine.", [])]), Settings(model="fake"), ep)
    assert ep.data["stop_reason"] == "stopped calling tools"
    assert ep.data["result"]["metrics"]["submitted"] is False


def test_api_error_gives_no_score(tmp_path):
    env = make(ENV, 0)
    ep = Episode(env, {"kind": "model", "name": "fake"}, tmp_path / "ep.json")
    run_model(env, FakeModel([("look", [("wait", '{"seconds": 10}')])], fail_at=1), Settings(model="fake"), ep)
    assert ep.data["status"] == "error" and ep.data["result"] is None


def test_cost_limit_stops_the_run(tmp_path):
    env = make(ENV, 0)
    ep = Episode(env, {"kind": "model", "name": "fake"}, tmp_path / "ep.json")
    run_model(env, FakeModel([("wait", [("wait", '{"seconds": 10}')])]), Settings(model="fake"), ep, max_cost_usd=0.03)
    assert ep.data["stop_reason"] == "cost limit" and ep.data["usage"]["model_calls"] == 3


def test_chat_body_and_reasoning_details_round_trip():
    s = Settings(model="m", reasoning_effort="high", max_tokens=100)
    body = build_chat_body([{"role": "user", "content": "x"}], s, [{"type": "function"}])
    assert body["provider"] == {"require_parameters": True} and body["reasoning"] == {"effort": "high"}
    local = build_chat_body([], s, None, openrouter=False)
    assert "provider" not in local and local["reasoning_effort"] == "high"
    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    turn = parse_chat({"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "reasoning": "think", "reasoning_details": details,
        "tool_calls": [{"id": "a", "type": "function", "function": {"name": "wait", "arguments": "{\"seconds\": 5}"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0.001}}, 0.5)
    assert turn.tool_calls == [{"id": "a", "name": "wait", "arguments": "{\"seconds\": 5}"}]
    assert turn.message["reasoning_details"] == details and turn.message["content"] == ""
    assert turn.reasoning == "think" and turn.cost_usd == 0.001


def test_export_and_server_paths(tmp_path):
    done = _scripted("factory", 0)
    done.path = tmp_path / "run-a.json"
    done.save()
    rows = serve.export([done.path], tmp_path / "site")
    assert rows[0]["id"] == "run-a" and rows[0]["passed"] is False
    assert (tmp_path / "site" / "run-a.json").is_file()
    running = Episode(make(ENV, 0), {"kind": "model", "name": "x"}, tmp_path / "run-b.json")
    with pytest.raises(SystemExit):
        serve.export([running.path], tmp_path / "site")
    handler = type("H", (serve._Handler,), {"episodes_dir": tmp_path})
    h = handler.__new__(handler)
    h.directory = str(tmp_path / "site")
    assert h.translate_path("/episodes/../../secret.json") == str(tmp_path / "secret.json")


# --- dsp.scale_readout ---

SCALE = "dsp.scale_readout"


def test_scale_chain_cost_rate_and_refusals():
    env = make(SCALE, 0)
    fs = env.interference.fs
    r = env.call("set_chain", {"blocks": [{"type": "moving_average", "n": 64}, {"type": "decimate", "factor": 64},
                                          {"type": "lowpass", "kind": "butter", "order": 4, "fc_hz": 2.0}]})
    assert r["cost_per_adc_sample"] == pytest.approx(2 + 10 / 64, abs=1e-3)
    assert r["output_rate_hz"] == pytest.approx(fs / 64)
    with pytest.raises(ToolError, match="budget"):
        env.call("set_chain", {"blocks": [{"type": "fir_lowpass", "taps": 101, "fc_hz": 10, "window": "hann"}]})
    with pytest.raises(ToolError, match="display rate"):
        env.call("set_chain", {"blocks": [{"type": "decimate", "factor": 64}, {"type": "decimate", "factor": 64}]})
    with pytest.raises(ToolError, match="unknown parameters"):
        env.call("set_chain", {"blocks": [{"type": "notch", "f0_hz": 50, "q": 10, "gain": 2}]})
    with pytest.raises(ToolError):
        env.call("set_chain", {"blocks": [{"type": "lowpass", "kind": "butter", "order": 2, "fc_hz": fs}]})
    with pytest.raises(ToolError):
        env.call("spectrum", {"points": 1000, "window": "hann", "averages": 1, "source": "adc"})


def test_scale_spectrum_finds_the_interference():
    env = make(SCALE, 0)
    s = env.interference
    r = env.call("spectrum", {"points": 16384, "window": "hann", "averages": 2, "source": "adc"})
    freqs = [f for f, _ in r["peaks"]]
    for target in (s.mains_hz, 3 * s.mains_hz, s.tone_hz):
        assert min(abs(f - target) for f in freqs) <= 2 * r["resolution_hz"]
    strongest_low = max((p for p in r["peaks"] if p[0] < 100), key=lambda p: p[1])
    assert strongest_low[1] == pytest.approx(s.hum_kg[0], rel=0.2)


def test_scale_signal_budget_is_enforced():
    env = make(SCALE, 0)
    for _ in range(4):
        env.call("run", {"seconds": 300})
    with pytest.raises(ToolError, match="left"):
        env.call("run", {"seconds": 1})


def test_scale_display_follows_the_chain_output():
    env = make(SCALE, 0)
    env.call("set_chain", {"blocks": [{"type": "moving_average", "n": 4000}, {"type": "decimate", "factor": 400}]})
    env.call("set_load", {"kg": 30})
    r = env.call("run", {"seconds": 5})
    assert r["readings"]["rows"][-1][1] == pytest.approx(30, abs=0.3)


# --- process.inline_heater ---

HEATER = "process.inline_heater"


def test_heater_flow_drop_heats_up_without_feedforward_and_not_with_it():
    for ff in (False, True):
        env = make(HEATER, 2)
        if ff:
            env.call("set_tuning", {"flow_feedforward": True})
        before = env.call("wait", {"seconds": 600})["temp_c"]
        env.call("set_flow", {"percent": 75})
        after = env.call("wait", {"seconds": 3000})["temp_c"]
        rise = env.plant.gain_c * env.scenario.commission_power / 100 * (1 / 0.75 - 1)
        if ff:
            assert after == pytest.approx(before, abs=0.3)
        else:
            assert after - before == pytest.approx(rise, rel=0.05)


def test_heater_feedforward_switch_keeps_power():
    env = make(HEATER, 0)
    env.call("set_flow", {"percent": 80})
    p0 = env.call("wait", {"seconds": 10})["power_pct"]
    r = env.call("set_tuning", {"flow_feedforward": True})
    assert r["power_pct"] == pytest.approx(p0, abs=0.01)
    assert r["output_pct"] == pytest.approx(p0 / 0.8, abs=0.01)


def test_heater_refusals():
    env = make(HEATER, 0)
    with pytest.raises(ToolError):
        env.call("set_flow", {"percent": 30})
    with pytest.raises(ToolError, match="unknown"):
        env.call("set_tuning", {"kd": 3})
    with pytest.raises(ToolError):
        env.call("set_tuning", {"flow_feedforward": "yes"})
