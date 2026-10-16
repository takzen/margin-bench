"""Model client interface. The environments never see it; it only produces turns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Settings:
    """Everything that defines how a model is queried. Stored with every episode."""

    model: str  # OpenRouter slug, e.g. "openai/gpt-5", or a local server's model name
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    # Provider slugs to pin, in order. When set, fallbacks to other providers are off.
    providers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["providers"] = list(self.providers)
        return d


@dataclass
class ChatTurn:
    """One assistant turn of a tool-using conversation."""

    message: dict[str, Any]  # appended to the conversation as is
    text: str
    tool_calls: list[dict[str, Any]]  # {"id", "name", "arguments" (JSON string)}
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    model_served: str | None = None
    provider_served: str | None = None
    latency_s: float | None = None


class ApiError(Exception):
    """The API call failed. Not the model's fault: the episode ends without a score."""


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], settings: Settings) -> ChatTurn: ...
