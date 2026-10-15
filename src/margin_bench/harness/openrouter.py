"""OpenRouter chat completions client.

Every request sets provider.require_parameters, so a provider that would
silently ignore a setting (reasoning effort, temperature) is never used.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from margin_bench.harness.client import ApiError, Completion, Settings

BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def build_body(prompt: str, settings: Settings) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "provider": {"require_parameters": True},
    }
    if settings.max_tokens is not None:
        body["max_tokens"] = settings.max_tokens
    if settings.temperature is not None:
        body["temperature"] = settings.temperature
    if settings.reasoning_effort is not None:
        body["reasoning"] = {"effort": settings.reasoning_effort}
    if settings.providers:
        body["provider"].update(order=list(settings.providers), allow_fallbacks=False)
    return body


def parse_response(data: dict[str, Any], latency_s: float) -> Completion:
    if data.get("error"):
        err = data["error"]
        raise ApiError(f"error in response body: {err.get('code')} {err.get('message')}")
    choices = data.get("choices") or []
    if not choices:
        raise ApiError("response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return Completion(
        text=message.get("content") or "",
        reasoning=message.get("reasoning"),
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        cost_usd=usage.get("cost"),
        model_served=data.get("model"),
        provider_served=data.get("provider"),
        generation_id=data.get("id"),
        latency_s=latency_s,
        raw_usage=usage,
    )


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        max_retries: int = 5,
        timeout_s: float = 900.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-OpenRouter-Title": "margin-bench",
            },
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            transport=transport,
        )
        self._max_retries = max_retries
        self._sleep = sleep

    def complete(self, prompt: str, settings: Settings) -> Completion:
        body = build_body(prompt, settings)
        last_error = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self._sleep(min(60.0, 2.0**attempt))
            start = time.monotonic()
            try:
                resp = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            latency = time.monotonic() - start
            if resp.status_code in RETRY_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                continue
            if resp.status_code != 200:
                raise ApiError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            try:
                return parse_response(resp.json(), latency)
            except ApiError as e:
                last_error = str(e)
                continue
        raise ApiError(f"gave up after {self._max_retries + 1} tries: {last_error}")


def fetch_models(transport: httpx.BaseTransport | None = None) -> dict[str, dict[str, Any]]:
    """Public model catalog with prices. Needs no API key and costs nothing."""
    with httpx.Client(base_url=BASE_URL, timeout=60.0, transport=transport) as http:
        resp = http.get("/models")
        resp.raise_for_status()
        return {m["id"]: m for m in resp.json()["data"]}
