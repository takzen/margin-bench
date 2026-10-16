"""Model client tests. No network: OpenRouter is replaced by httpx.MockTransport."""

import json

import httpx
import pytest

from margin_bench.harness.client import ApiError, Settings
from margin_bench.harness.env import get_secret
from margin_bench.harness.openrouter import OpenRouterClient, build_chat_body

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
MESSAGES = [{"role": "user", "content": "prompt"}]


def make_client(handler, **kw):
    return OpenRouterClient("sk-test", transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_request_body_pins_settings():
    s = Settings("openai/gpt-x", reasoning_effort="high", max_tokens=32000, temperature=0.0, providers=("openai",))
    body = build_chat_body(MESSAGES, s)
    assert body["reasoning"] == {"effort": "high"}
    assert body["max_tokens"] == 32000
    assert body["temperature"] == 0.0
    assert body["provider"] == {"require_parameters": True, "order": ["openai"], "allow_fallbacks": False}
    assert build_chat_body(MESSAGES, Settings("m"))["provider"] == {"require_parameters": True}


def test_parses_turn_and_usage():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    t = make_client(handler).chat(MESSAGES, [], Settings("openai/gpt-x", reasoning_effort="low"))
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"] == MESSAGES
    assert (t.text, t.reasoning, t.finish_reason, t.tool_calls) == ("answer", "thinking", "stop", [])
    assert (t.prompt_tokens, t.completion_tokens, t.reasoning_tokens) == (300, 1200, 1000)
    assert t.cost_usd == 0.0123
    assert (t.model_served, t.provider_served) == ("openai/gpt-x", "OpenAI")


def test_local_server_gets_a_plain_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    client = OpenRouterClient(None, base_url="http://localhost:11434/v1", transport=httpx.MockTransport(handler))
    client.chat(MESSAGES, [], Settings("qwen3", reasoning_effort="high"))
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in seen["headers"]
    assert "provider" not in seen["body"] and seen["body"]["reasoning_effort"] == "high"


def test_retries_transient_errors():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        if len(calls) == 2:
            return httpx.Response(200, json={"error": {"code": 502, "message": "provider disconnected"}})
        return httpx.Response(200, json=OK_BODY)

    assert make_client(handler).chat(MESSAGES, [], Settings("m")).text == "answer"
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    with pytest.raises(ApiError, match="HTTP 400"):
        make_client(handler).chat(MESSAGES, [], Settings("m"))
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    client = make_client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(ApiError, match="gave up after 3 tries"):
        client.chat(MESSAGES, [], Settings("m"))


def test_secret_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nOTHER=1\nexport OPENROUTER_API_KEY="sk-from-file"\n', encoding="utf-8")
    assert get_secret("OPENROUTER_API_KEY", env) == "sk-from-file"
    assert get_secret("MISSING", env) is None
