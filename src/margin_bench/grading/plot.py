"""Plot data shared by every task family. The site draws it with one renderer.

A plot is a list of panels. Each panel has axes, data series, and the spec drawn
as forbidden rectangles and/or an allowed envelope around a curve. Violations
are points of the answer that the grader found outside the spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_POINTS = 1200
MAX_VIOLATIONS = 256


def rounded(values: Any, digits: int = 6) -> list[float]:
    """JSON-friendly floats with a bounded number of significant digits."""
    out = []
    for v in np.asarray(values, dtype=float).ravel():
        out.append(float(f"{v:.{digits}g}") if np.isfinite(v) else None)
    return out


def thin(x: np.ndarray, *ys: np.ndarray, max_points: int = MAX_POINTS) -> tuple[np.ndarray, ...]:
    """Evenly subsample long series so plot data stays small."""
    if len(x) <= max_points:
        return (x, *ys)
    idx = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return (x[idx], *(y[idx] for y in ys))


def series(name: str, x: Any, y: Any, role: str = "answer") -> dict[str, Any]:
    """role: 'answer' (what the model submitted) or 'target' (what the spec asks for)."""
    xs, ys = thin(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return {"name": name, "role": role, "x": rounded(xs), "y": rounded(ys)}


def forbidden(x0: float | None, x1: float | None, y0: float | None, y1: float | None, label: str = "") -> dict[str, Any]:
    """A rectangle the answer must stay out of. None means unbounded on that side."""
    return {"x": [x0, x1], "y": [y0, y1], "label": label}


def envelope(x: Any, lo: Any, hi: Any, label: str = "") -> dict[str, Any]:
    """A corridor the answer must stay inside."""
    xs, los, his = thin(np.asarray(x, dtype=float), np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    return {"x": rounded(xs), "lo": rounded(los), "hi": rounded(his), "label": label}


def marker(axis: str, value: float, label: str) -> dict[str, Any]:
    return {"axis": axis, "value": float(f"{value:.6g}"), "label": label}


def violations(x: Any, y: Any) -> list[list[float]]:
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    if len(pts) > MAX_VIOLATIONS:
        pts = pts[np.linspace(0, len(pts) - 1, MAX_VIOLATIONS).round().astype(int)]
    return [rounded(p) for p in pts]


def panel(
    title: str,
    x_label: str,
    y_label: str,
    series_list: list[dict[str, Any]],
    *,
    x_scale: str = "linear",
    y_scale: str = "linear",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    forbidden_list: list[dict[str, Any]] | None = None,
    envelope_obj: dict[str, Any] | None = None,
    markers: list[dict[str, Any]] | None = None,
    violation_points: list[list[float]] | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "title": title,
        "note": note,
        "x": {"label": x_label, "scale": x_scale, "range": None if x_range is None else rounded(x_range)},
        "y": {"label": y_label, "scale": y_scale, "range": None if y_range is None else rounded(y_range)},
        "series": series_list,
        "forbidden": forbidden_list or [],
        "envelope": envelope_obj,
        "markers": markers or [],
        "violations": violation_points or [],
    }


def plot(*panels: dict[str, Any]) -> dict[str, Any]:
    return {"panels": list(panels)}
