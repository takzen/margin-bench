"""Aggregate attempt records into per-model results with bootstrap intervals.

The bootstrap resamples task instances, not single attempts: the k attempts on
one instance are correlated, so treating them as independent would make the
intervals too narrow.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

N_BOOT = 2000
BOOT_SEED = 0
CI = 0.95


def bootstrap_ci(per_instance: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    if len(per_instance) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_instance), size=(n_boot, len(per_instance)))
    means = per_instance[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return float(lo), float(hi)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (settings_key, family_id). Only successful API calls count."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    api_errors: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        key = (r["settings_key"], r["family_id"])
        if r["status"] == "ok":
            groups[key].append(r)
        else:
            api_errors[key] += 1

    rows = []
    for (settings_key, family_id), recs in sorted(groups.items()):
        by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_instance[r["instance_id"]].append(r)
        inst_score = np.array([np.mean([r["grade"]["score"] for r in rs]) for rs in by_instance.values()])
        inst_pass = np.array([np.mean([r["grade"]["passed"] for r in rs]) for rs in by_instance.values()])

        def total(field: str) -> float | None:
            vals = [r["completion"][field] for r in recs]
            return None if any(v is None for v in vals) else float(sum(vals))

        n = len(recs)
        out_tokens = total("completion_tokens")
        reasoning = total("reasoning_tokens")
        cost = total("cost_usd")
        rows.append(
            {
                "settings_key": settings_key,
                "family_id": family_id,
                "instances": len(by_instance),
                "attempts": n,
                "mean_score": float(inst_score.mean()),
                "mean_score_ci": bootstrap_ci(inst_score),
                "pass_rate": float(inst_pass.mean()),
                "pass_rate_ci": bootstrap_ci(inst_pass),
                "format_errors": sum(r["grade"]["format_error"] for r in recs),
                "api_errors": api_errors[(settings_key, family_id)],
                "mean_output_tokens": None if out_tokens is None else out_tokens / n,
                "mean_reasoning_tokens": None if reasoning is None else reasoning / n,
                "total_cost_usd": cost,
            }
        )
    return rows
