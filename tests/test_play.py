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
SEEDS = range(20)


def _scripted(name: str, seed: int) -> Episode:
    env = make(ENV, seed)
    ep = Episode(env, {"kind": "scripted", "name": name})
    SCRIPTED[ENV][name](ep)
    ep.finish(env.finish(), stop_reason="submitted")
    return ep


@pytest.mark.parametrize("seed", SEEDS)
def test_reference_engineer_passes(seed):
    ep = _scripted("reference", seed)
    r = ep.data["result"]
    assert r["passed"] and r["score"] == 1.0, r["metrics"]
    assert r["metrics"]["commissioning_alarms"] == 0


@pytest.mark.parametrize("name", NEGATIVE[ENV])
@pytest.mark.parametrize("seed", SEEDS)
def test_known_bad_agents_fail(name, seed):
    r = _scripted(name, seed).data["result"]
    assert not r["passed"] and r["score"] < 1.0


def test_episode_is_reproducible():
    a, b = _scripted("reference", 3), _scripted("reference", 3)
    assert a.data["result"] == b.data["result"]
    assert a.data["view"]["commissioning"]["rows"] == b.data["view"]["commissioning"]["rows"]


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
