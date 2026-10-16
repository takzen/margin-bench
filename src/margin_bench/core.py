"""The result of an acceptance test."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Score:
    """Result of grading one episode.

    `score` is continuous in [0, 1] and equals 1.0 exactly when `passed` is True:
    each requirement gives a ratio that reaches 1 when it is met, and the score is
    the smallest one.
    """

    score: float
    passed: bool
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
