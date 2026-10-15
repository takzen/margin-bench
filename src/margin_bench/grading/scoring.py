"""Continuous scoring from margins against a spec.

Each requirement maps to a ratio in [0, 1] that reaches 1.0 exactly when the
requirement is met. A task's score is the minimum over its requirements, so it
is 1.0 if and only if the answer passes, and otherwise shows how far the
weakest requirement fell short.
"""

from __future__ import annotations

import math


def at_least(achieved: float, required: float) -> float:
    """Score for a 'higher is better' requirement, e.g. stopband attenuation."""
    if required <= 0:
        raise ValueError("required must be positive")
    if math.isnan(achieved):
        return 0.0
    return min(1.0, max(0.0, achieved / required))


def at_most(achieved: float, limit: float) -> float:
    """Score for a 'lower is better' limit, e.g. passband deviation."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if math.isnan(achieved):
        return 0.0
    if achieved <= limit:
        return 1.0
    return limit / achieved


def combine(*ratios: float) -> float:
    return min(ratios)
