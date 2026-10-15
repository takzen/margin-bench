"""Harness tests. No network: OpenRouter is replaced by httpx.MockTransport or a fake client."""

import json

import httpx
import pytest

from margin_bench.harness.client import ApiError, Completion, Settings
from margin_bench.harness.env import get_secret
from margin_bench.harness.openrouter import OpenRouterClient, build_body
from margin_bench.harness.runner import AttemptStore, RunConfig, estimate, execute, pending, plan
from margin_bench.report.aggregate import summarize
from margin_bench.tasks import get_family

OK_BODY = {
    "id": "gen-123",
    "model": "openai/gpt-x",
    "provider": "OpenAI",
    "choices": [
        {"finish_reason": "stop", "message": {"role": "assistant", "content": "answer", "reasoning": "thinking"}}
    ],
    "usage": {
        "prompt_tokens": 300,
        "completion_tokens": 1200,
        "completion_tokens_details": {"reasoning_tokens": 1000},
        "cost": 0.0123,
    },
}


def make_client(handler):
    return OpenRouterClient("sk-test", transport=httpx.MockTransport(handler), sleep=lambda s: None)


def test_request_body_pins_settings():
    s = Settings("openai/gpt-x", reasoning_effort="high", max_tokens=32000, temperature=0.0, providers=("openai",))
    body = build_body("hi", s)
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_tokens"] == 32000
    assert body["temperature"] == 0.0
    assert body["provider"] == {"require_parameters": True, "order": ["openai"], "allow_fallbacks": False}
    assert build_body("hi", Settings("m"))["provider"] == {"require_parameters": True}


def test_parses_completion_and_usage():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    c = make_client(handler).complete("prompt", Settings("openai/gpt-x", reasoning_effort="low"))
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert (c.text, c.reasoning, c.finish_reason) == ("answer", "thinking", "stop")
    assert (c.prompt_tokens, c.completion_tokens, c.reasoning_tokens) == (300, 1200, 1000)
    assert c.cost_usd == 0.0123
    assert (c.model_served, c.provider_served, c.generation_id) == ("openai/gpt-x", "OpenAI", "gen-123")


def test_retries_transient_errors():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if len(calls) == 2:
            return httpx.Response(200, json={"error": {"code": 502, "message": "provider disconnected"}})
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler).complete("p", Settings("m")).text == "answer"
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    with pytest.raises(ApiError, match="HTTP 400"):
        make_client(handler).complete("p", Settings("m"))
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    client = OpenRouterClient(
        "k", max_retries=2, transport=httpx.MockTransport(lambda r: httpx.Response(503)), sleep=lambda s: None
    )
    with pytest.raises(ApiError, match="gave up after 3 tries"):
        client.complete("p", Settings("m"))


def test_secret_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nOTHER=1\nexport OPENROUTER_API_KEY="sk-from-file"\n', encoding="utf-8")
    assert get_secret("OPENROUTER_API_KEY", env) == "sk-from-file"
    assert get_secret("MISSING", env) is None


class FakeClient:
    """Answers with the reference design for one model and with no JSON for the other."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, settings):
        self.calls += 1
        if settings.model == "good/model":
            fam = get_family("dsp.fir_lowpass_mask")
            task = next(fam.generate(s) for s in range(10) if fam.generate(s).prompt == prompt)
            text = "```json\n" + json.dumps(fam.reference(task)) + "\n```"
        else:
            text = "I would use a Kaiser window."
        return Completion(text=text, prompt_tokens=300, completion_tokens=900, reasoning_tokens=500, cost_usd=0.01)


@pytest.fixture
def cfg(tmp_path):
    return RunConfig(
        name="t",
        families=("dsp.fir_lowpass_mask",),
        seeds={"dsp.fir_lowpass_mask": (0, 1, 2)},
        k=2,
        models=(Settings("good/model", reasoning_effort="high"), Settings("bad/model")),
        output=tmp_path / "t.jsonl",
        workers=2,
    )


def test_run_records_grades_and_resumes(cfg):
    store = AttemptStore(cfg.output)
    attempts = plan(cfg)
    assert len(attempts) == 3 * 2 * 2
    client = FakeClient()
    summary = execute(cfg, pending(attempts, store), client, store, log=lambda s: None)
    assert summary["ok"] == 12 and client.calls == 12
    assert summary["spent_usd"] == pytest.approx(0.12)

    records = store.records()
    assert all("plot_data" not in r["grade"] for r in records)
    assert {r["settings"]["reasoning_effort"] for r in records} == {"high", None}

    again = FakeClient()
    execute(cfg, pending(attempts, store), again, store, log=lambda s: None)
    assert again.calls == 0

    rows = {r["settings_key"]: r for r in summarize(store.records())}
    good, bad = rows["good/model|effort=high"], rows["bad/model"]
    assert good["mean_score"] == 1.0 and good["pass_rate"] == 1.0 and good["mean_score_ci"] == (1.0, 1.0)
    assert bad["mean_score"] == 0.0 and bad["format_errors"] == 6
    assert good["instances"] == 3 and good["attempts"] == 6
    assert good["mean_reasoning_tokens"] == 500


def test_budget_cap_stops_new_calls(cfg):
    capped = RunConfig(**{**cfg.__dict__, "max_cost_usd": 0.035, "workers": 1})
    store = AttemptStore(capped.output)
    summary = execute(capped, plan(capped), FakeClient(), store, log=lambda s: None)
    assert summary["ok"] == 4
    assert summary["skipped_budget"] == 8


def test_api_errors_are_recorded_and_retried(cfg):
    class Failing:
        def complete(self, prompt, settings):
            raise ApiError("HTTP 503")

    store = AttemptStore(cfg.output)
    summary = execute(cfg, plan(cfg), Failing(), store, log=lambda s: None)
    assert summary["api_error"] == 12
    assert len(pending(plan(cfg), store)) == 12


def test_estimate_uses_catalog_prices(cfg):
    catalog = {
        "good/model": {
            "pricing": {"prompt": "0.000001", "completion": "0.00001"},
            "supported_parameters": ["reasoning", "max_tokens"],
            "reasoning": {"supported_efforts": ["high", "low"]},
            "top_provider": {"max_completion_tokens": 20000},
        },
    }
    rows = estimate(cfg, plan(cfg), catalog)
    good, bad = rows
    assert good.calls == 6 and not good.issues
    assert good.expected_output_tokens == 6 * 8000
    assert good.worst_output_tokens == 6 * 20000
    assert good.expected_usd == pytest.approx(good.prompt_tokens * 1e-6 + 48000 * 1e-5)
    assert bad.issues == ["model not found in the OpenRouter catalog"]


def test_run_config_from_toml(tmp_path):
    p = tmp_path / "smoke.toml"
    p.write_text(
        """
name = "smoke"
families = ["dsp.fir_lowpass_mask"]
seeds = [0, 1]
k = 3
max_cost_usd = 5.0

[[models]]
model = "openai/gpt-x"
reasoning_effort = "high"
max_tokens = 32000
providers = ["openai"]
""",
        encoding="utf-8",
    )
    cfg = RunConfig.from_toml(p)
    assert cfg.output.as_posix() == "results/runs/smoke.jsonl"
    assert cfg.models[0].key == "openai/gpt-x|effort=high|max_tokens=32000|providers=openai"
    assert cfg.max_cost_usd == 5.0
    assert cfg.seeds == {"dsp.fir_lowpass_mask": (0, 1)}


def test_hidden_seeds_must_write_to_private(tmp_path):
    seeds = tmp_path / "test-v0.json"
    seeds.write_text('{"dsp.fir_lowpass_mask": [123456789]}', encoding="utf-8")
    base = f"""
name = "hidden"
families = ["dsp.fir_lowpass_mask"]
seeds_file = "{seeds.as_posix()}"
k = 1
[[models]]
model = "m"
"""
    p = tmp_path / "hidden.toml"
    p.write_text(base, encoding="utf-8")
    with pytest.raises(ValueError, match="private"):
        RunConfig.from_toml(p)
    p.write_text(base.replace("k = 1", 'k = 1\noutput = "private/runs/hidden.jsonl"'), encoding="utf-8")
    assert RunConfig.from_toml(p).seeds == {"dsp.fir_lowpass_mask": (123456789,)}
