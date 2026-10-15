"""Preferred-number series (IEC 60063) and helpers for the electronics families."""

from __future__ import annotations

import math

import numpy as np

E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)
E24 = (1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
       3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1)  # fmt: skip
E96 = (1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
       1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
       1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
       2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
       3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
       4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
       5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
       7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76)  # fmt: skip
SERIES = {"E12": E12, "E24": E24, "E96": E96}


def values(series: str, lo: float, hi: float) -> np.ndarray:
    """Every value of the series in [lo, hi]."""
    base = SERIES[series]
    out = []
    for decade in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in base:
            v = float(f"{m * 10.0**decade:.4g}")
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                out.append(v)
    return np.array(sorted(set(out)))


def in_series(value: float, series: str) -> bool:
    if value <= 0 or not math.isfinite(value):
        return False
    decade = math.floor(math.log10(value) + 1e-12)
    mantissa = value / 10.0**decade
    return any(abs(mantissa - m) <= 1e-6 * m for m in SERIES[series]) or abs(mantissa - 10.0) <= 1e-5


def nearest(value: float, series: str) -> float:
    """The series value closest to `value` on a log scale."""
    candidates = values(series, value / 1.5, value * 1.5)
    return float(candidates[np.argmin(np.abs(np.log(candidates / value)))])


def eng(value: float, unit: str) -> str:
    """Human-readable value with an SI prefix, for prompts."""
    if value == 0:
        return f"0 {unit}"
    prefixes = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}
    exp = max(-12, min(9, 3 * math.floor(math.log10(abs(value)) / 3)))
    return f"{value / 10.0**exp:.4g} {prefixes[exp]}{unit}"
