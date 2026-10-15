"""Core types shared by task families, graders and the harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Task:
    """One generated task instance. Everything the grader needs lives in `spec`."""

    family_id: str
    seed: int
    prompt: str
    spec: dict[str, Any]
    answer_format: str = "artifact"

    @property
    def instance_id(self) -> str:
        return f"{self.family_id}/{self.seed}"


@dataclass
class Score:
    """Result of grading one answer.

    `score` is continuous in [0, 1] and equals 1.0 exactly when `passed` is True.
    `format_error` marks answers that could not be parsed or validated, so that
    "cannot solve" stays separate from "cannot follow the answer format".
    """

    score: float
    passed: bool
    format_error: bool = False
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    plot_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def format_failure(cls, error: str) -> Score:
        return cls(score=0.0, passed=False, format_error=True, error=error)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskFamily(Protocol):
    id: str  # e.g. "dsp.fir_lowpass_mask"
    domain: str  # "dsp" | "electronics" | "control"

    def generate(self, seed: int) -> Task:
        """Build the prompt and spec deterministically from the seed."""
        ...

    def grade(self, task: Task, answer: dict[str, Any]) -> Score:
        """Grade a parsed answer by measurement or simulation only."""
        ...

    def reference(self, task: Task) -> dict[str, Any]:
        """A known-good answer that must pass."""
        ...

    def negative(self, task: Task) -> dict[str, Any]:
        """A known-bad answer (a realistic mistake) that must fail."""
        ...
