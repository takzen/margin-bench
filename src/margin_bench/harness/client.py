"""Model client interface. Graders never see this; it only produces text."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Settings:
    """Everything that defines how a model is queried. Stored with every attempt."""

    model: str  # OpenRouter slug, e.g. "openai/gpt-5"
    reasoning_effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    # Provider slugs to pin, in order. When set, fallbacks to other providers are off.
    providers: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        parts = [self.model]
        if self.reasoning_effort is not None:
            parts.append(f"effort={self.reasoning_effort}")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        if self.temperature is not None:
            parts.append(f"temperature={self.temperature:g}")
        if self.providers:
            parts.append("providers=" + ",".join(self.providers))
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["providers"] = list(self.providers)
        return d


@dataclass
class Completion:
    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None  # includes reasoning tokens
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    model_served: str | None = None
    provider_served: str | None = None
    generation_id: str | None = None
    latency_s: float | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """The API call failed. Not the model's fault, so it is retried on resume."""


class ModelClient(Protocol):
    def complete(self, prompt: str, settings: Settings) -> Completion: ...


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], settings: Settings) -> ChatTurn: ...
