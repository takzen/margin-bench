"""Seeded randomness that stays reproducible across Python and NumPy versions.

All draws are built on `random.Random.random()`, the one method whose sequence
Python guarantees not to change for a given seed. NumPy's Generator makes no
such promise, so scenario generators must not use it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


class SeededRandom:
    def __init__(self, seed: int | str) -> None:
        self._r = random.Random(seed)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self._r.random()

    def integer(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], both ends inclusive."""
        return lo + int(self._r.random() * (hi - lo + 1))

    def choice(self, seq: Sequence[T]) -> T:
        return seq[int(self._r.random() * len(seq))]

    def gauss(self) -> float:
        """Standard normal draw (Box-Muller on two uniform draws)."""
        u1 = 1.0 - self._r.random()  # in (0, 1], keeps log finite
        u2 = self._r.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
