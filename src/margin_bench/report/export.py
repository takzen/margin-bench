"""Export results for the static site.

Writes into the output directory:
  summary.json             leaderboard, scores per domain and family, the attempt index
  attempts/<id>.json       one attempt: prompt, response, usage, grade and plot data

Every attempt is graded again from its stored response, with plot data. The
task is regenerated from its seed, and the export stops if the regenerated
prompt differs from the stored one: that would mean the generator changed
under existing results.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from margin_bench.grading import grade_response
from margin_bench.report.aggregate import per_instance_scores, stratified_mean_ci, summarize
from margin_bench.tasks import FAMILIES, REVIEW_STATUS, get_family

DOMAINS = ("dsp", "electronics", "control")


class ExportError(RuntimeError):
    pass


def model_label(settings: dict[str, Any]) -> str:
    label = settings["model"]
    extras = []
    if settings.get("reasoning_effort"):
        extras.append(f"effort {settings['reasoning_effort']}")
    if settings.get("temperature") is not None:
        extras.append(f"T={settings['temperature']:g}")
    return f"{label} ({', '.join(extras)})" if extras else label


def family_catalog() -> list[dict[str, Any]]:
    out = []
    for fid, fam in FAMILIES.items():
        doc = (sys.modules[type(fam).__module__].__doc__ or "").strip()
        paragraphs = [p.strip().replace("\n", " ") for p in doc.split("\n\n") if p.strip()]
        title = paragraphs[0].split(":", 1)[-1].strip().rstrip(".") if paragraphs else fid
        out.append(
            {
                "id": fid,
                "domain": fam.domain,
                "title": title[:1].upper() + title[1:],
                "description": paragraphs[1:],
                "status": REVIEW_STATUS.get(fid, "awaiting review"),
                "example_prompt": fam.generate(0).prompt,
            }
        )
    return out


def _attempt_payload(record: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    family = get_family(record["family_id"])
    task = family.generate(record["seed"])
    if task.prompt != record["prompt"]:
        raise ExportError(
            f"{record['instance_id']}: the regenerated prompt differs from the stored one; "
            "the generator changed since this run"
        )
    completion = record["completion"]
    score = grade_response(family, task, completion["text"], with_plot=True)
    return {
        "id": attempt_id,
        "run": record["run"],
        "model_key": record["settings_key"],
        "model_label": model_label(record["settings"]),
        "settings": record["settings"],
        "family_id": record["family_id"],
        "domain": family.domain,
        "seed": record["seed"],
        "instance_id": record["instance_id"],
        "attempt": record["attempt"],
        "timestamp": record.get("timestamp"),
        "prompt": record["prompt"],
        "spec": record["spec"],
        "response": completion["text"],
        "reasoning": completion.get("reasoning"),
        "finish_reason": completion.get("finish_reason"),
        "usage": {
            "prompt_tokens": completion.get("prompt_tokens"),
            "completion_tokens": completion.get("completion_tokens"),
            "reasoning_tokens": completion.get("reasoning_tokens"),
        },
        "cost_usd": completion.get("cost_usd"),
        "latency_s": completion.get("latency_s"),
        "model_served": completion.get("model_served"),
        "provider_served": completion.get("provider_served"),
        "grade": {k: v for k, v in score.to_dict().items() if k != "plot_data"},
        "plot_data": score.plot_data,
    }


def build(records: list[dict[str, Any]], demo: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ok = [r for r in records if r["status"] == "ok"]
    attempts = [_attempt_payload(r, f"a{i:05d}") for i, r in enumerate(ok)]
    family_rows = summarize(ok)
    scores = per_instance_scores(ok)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts:
        by_model[a["model_key"]].append(a)

    models = []
    for key, items in by_model.items():
        per_family = scores[key]
        mean, ci = stratified_mean_ci(per_family)
        domains = {}
        for d in DOMAINS:
            subset = {fid: v for fid, v in per_family.items() if get_family(fid).domain == d}
            if subset:
                m, c = stratified_mean_ci(subset)
                domains[d] = {"mean": m, "ci": list(c), "families": len(subset)}
        fams = {
            row["family_id"]: {
                "mean": row["mean_score"],
                "ci": list(row["mean_score_ci"]),
                "pass_rate": row["pass_rate"],
                "attempts": row["attempts"],
                "instances": row["instances"],
                "format_errors": row["format_errors"],
            }
            for row in family_rows
            if row["settings_key"] == key
        }
        costs = [a["cost_usd"] for a in items]
        out_tokens = [a["usage"]["completion_tokens"] for a in items]
        models.append(
            {
                "key": key,
                "label": items[0]["model_label"],
                "settings": items[0]["settings"],
                "overall": {"mean": mean, "ci": list(ci)},
                "domains": domains,
                "families": fams,
                "attempts": len(items),
                "pass_rate": sum(a["grade"]["passed"] for a in items) / len(items),
                "format_errors": sum(a["grade"]["format_error"] for a in items),
                "cost_usd": None if any(c is None for c in costs) else sum(costs),
                "mean_output_tokens": None if any(t is None for t in out_tokens) else sum(out_tokens) / len(items),
            }
        )
    models.sort(key=lambda m: -m["overall"]["mean"])
    summary = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark_version": version("margin-bench"),
        "demo": demo,
        "runs": sorted({r["run"] for r in ok}),
        "families": family_catalog(),
        "models": models,
        "attempts": [
            {
                "id": a["id"],
                "model_key": a["model_key"],
                "family_id": a["family_id"],
                "seed": a["seed"],
                "attempt": a["attempt"],
                "score": a["grade"]["score"],
                "passed": a["grade"]["passed"],
                "format_error": a["grade"]["format_error"],
            }
            for a in attempts
        ],
    }
    return summary, attempts


def write(out_dir: Path, summary: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    attempts_dir = out_dir / "attempts"
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)  # generated by this function; rebuilt from scratch
    attempts_dir.mkdir(parents=True)
    for a in attempts:
        (attempts_dir / f"{a['id']}.json").write_text(json.dumps(a, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, allow_nan=False), encoding="utf-8")


def demo_records(seeds: range) -> list[dict[str, Any]]:
    """Records in the runner's format whose 'models' are the reference and negative answers."""
    records = []
    for fid, fam in FAMILIES.items():
        for seed in seeds:
            task = fam.generate(seed)
            for label, answer in (("reference", fam.reference(task)), ("negative", fam.negative(task))):
                text = (
                    f"Demo answer: the task family's {label} design, graded like a model answer.\n\n"
                    f"```json\n{json.dumps(answer)}\n```\n"
                )
                score = fam.grade(task, answer, with_plot=False)
                grade = score.to_dict()
                grade.pop("plot_data")
                records.append(
                    {
                        "run": "demo",
                        "status": "ok",
                        "settings_key": f"demo/{label}",
                        "settings": {"model": f"demo/{label}", "reasoning_effort": None, "max_tokens": None,
                                     "temperature": None, "providers": []},
                        "family_id": fid,
                        "seed": seed,
                        "instance_id": task.instance_id,
                        "attempt": 0,
                        "timestamp": None,
                        "prompt": task.prompt,
                        "spec": task.spec,
                        "completion": {"text": text, "reasoning": None, "finish_reason": "stop",
                                       "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                                       "cost_usd": None, "latency_s": None, "model_served": None,
                                       "provider_served": None},
                        "grade": grade,
                        "error": None,
                    }
                )  # fmt: skip
    return records
