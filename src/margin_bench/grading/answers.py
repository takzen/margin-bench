"""Answer validation shared by task families.

A malformed answer (missing key, wrong type, non-finite number) raises
AnswerError and becomes a format error. A well-formed answer that breaks a
design constraint (a value off the E24 series, a negative pole) is the
family's business: it is graded as a failed design, not as a format error.
"""

from __future__ import annotations

import math
from typing import Any


class AnswerError(ValueError):
    pass


def _as_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnswerError(f"{what} must be a number")
    if not math.isfinite(value):
        raise AnswerError(f"{what} must be finite")
    return float(value)


def number(answer: Any, key: str) -> float:
    if not isinstance(answer, dict) or key not in answer:
        raise AnswerError(f'missing "{key}"')
    return _as_number(answer[key], f'"{key}"')


def integer(answer: Any, key: str) -> int:
    v = number(answer, key)
    if not v.is_integer():
        raise AnswerError(f'"{key}" must be an integer')
    return int(v)


def boolean(answer: Any, key: str) -> bool:
    if not isinstance(answer, dict) or key not in answer:
        raise AnswerError(f'missing "{key}"')
    if not isinstance(answer[key], bool):
        raise AnswerError(f'"{key}" must be true or false')
    return answer[key]


def number_list(answer: Any, key: str, min_len: int = 1, max_len: int | None = None) -> list[float]:
    if not isinstance(answer, dict) or key not in answer:
        raise AnswerError(f'missing "{key}"')
    values = answer[key]
    if not isinstance(values, list):
        raise AnswerError(f'"{key}" must be a list of numbers')
    if len(values) < min_len or (max_len is not None and len(values) > max_len):
        span = f"{min_len}" if max_len == min_len else f"{min_len} to {max_len}"
        raise AnswerError(f'"{key}" must have {span} entries')
    return [_as_number(v, f'"{key}"[{i}]') for i, v in enumerate(values)]
