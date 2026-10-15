"""Run configs, the attempt plan, the dry-run cost estimate and execution.

Attempts are appended to a JSONL file as they finish. A rerun of the same
config skips attempts that already have a successful record, so an
interrupted run resumes where it stopped. Failed API calls are recorded too,
and retried on the next run.
"""

from __future__ import annotations

import json
import math
import threading
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from margin_bench.core import Task
from margin_bench.grading import grade_response
from margin_bench.harness.client import ApiError, ModelClient, Settings
from margin_bench.tasks import get_family

RECORD_SCHEMA = 1
CHARS_PER_TOKEN = 3.5  # conservative: overestimates prompt tokens for English text


@dataclass(frozen=True)
class RunConfig:
    name: str
    families: tuple[str, ...]
    seeds: tuple[int, ...]
    k: int
    models: tuple[Settings, ...]
    output: Path
    expected_output_tokens: int = 8000  # used only by the cost estimate
    max_cost_usd: float | None = None
    workers: int = 4

    @classmethod
    def from_toml(cls, path: Path) -> RunConfig:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        models = tuple(
            Settings(
                model=m["model"],
                reasoning_effort=m.get("reasoning_effort"),
                max_tokens=m.get("max_tokens"),
                temperature=m.get("temperature"),
                providers=tuple(m.get("providers", ())),
            )
            for m in raw["models"]
        )
        name = raw["name"]
        return cls(
            name=name,
            families=tuple(raw["families"]),
            seeds=tuple(raw["seeds"]),
            k=raw["k"],
            models=models,
            output=Path(raw.get("output", f"results/runs/{name}.jsonl")),
            expected_output_tokens=raw.get("expected_output_tokens", 8000),
            max_cost_usd=raw.get("max_cost_usd"),
            workers=raw.get("workers", 4),
        )


@dataclass(frozen=True)
class Attempt:
    settings: Settings
    task: Task
    index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.settings.key, self.task.instance_id, self.index)


def plan(cfg: RunConfig) -> list[Attempt]:
    """All attempts, ordered so that a run cut short stays balanced across models."""
    tasks = [get_family(fid).generate(seed) for fid in cfg.families for seed in cfg.seeds]
    return [Attempt(s, t, i) for i in range(cfg.k) for t in tasks for s in cfg.models]


class AttemptStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def done_keys(self) -> set[tuple[str, str, int]]:
        return {
            (r["settings_key"], r["instance_id"], r["attempt"])
            for r in self.records()
            if r["status"] == "ok"
        }

    def spent_usd(self) -> float:
        return sum(r["completion"]["cost_usd"] or 0.0 for r in self.records() if r["status"] == "ok")

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


@dataclass
class ModelEstimate:
    settings_key: str
    calls: int
    prompt_tokens: int
    expected_output_tokens: int
    worst_output_tokens: int
    usd_per_mtok_in: float | None
    usd_per_mtok_out: float | None
    expected_usd: float | None
    worst_usd: float | None
    issues: list[str] = field(default_factory=list)


def estimate(
    cfg: RunConfig, attempts: list[Attempt], catalog: dict[str, dict[str, Any]]
) -> list[ModelEstimate]:
    """Cost of the pending attempts, from list prices. Makes no paid calls."""
    rows = []
    for s in cfg.models:
        mine = [a for a in attempts if a.settings == s]
        info = catalog.get(s.model)
        issues = []
        prompt_tokens = sum(math.ceil(len(a.task.prompt) / CHARS_PER_TOKEN) for a in mine)
        limit = s.max_tokens
        if info is None:
            issues.append("model not found in the OpenRouter catalog")
        else:
            limit = limit or (info.get("top_provider") or {}).get("max_completion_tokens")
            if s.reasoning_effort is not None:
                if "reasoning" not in info.get("supported_parameters", []):
                    issues.append("model does not list 'reasoning' as a supported parameter")
                efforts = (info.get("reasoning") or {}).get("supported_efforts")
                if efforts and s.reasoning_effort not in efforts:
                    issues.append(f"effort {s.reasoning_effort!r} not in {efforts}")
        per_call = cfg.expected_output_tokens if limit is None else min(cfg.expected_output_tokens, limit)
        expected_out = per_call * len(mine)
        worst_out = (limit or cfg.expected_output_tokens) * len(mine)
        p_in = p_out = None
        if info is not None:
            p_in, p_out = float(info["pricing"]["prompt"]), float(info["pricing"]["completion"])
            if p_in < 0 or p_out < 0:
                issues.append("price is dynamic (router model); estimate unavailable")
                p_in = p_out = None
        rows.append(
            ModelEstimate(
                settings_key=s.key,
                calls=len(mine),
                prompt_tokens=prompt_tokens,
                expected_output_tokens=expected_out,
                worst_output_tokens=worst_out,
                usd_per_mtok_in=None if p_in is None else p_in * 1e6,
                usd_per_mtok_out=None if p_out is None else p_out * 1e6,
                expected_usd=None if p_in is None else prompt_tokens * p_in + expected_out * p_out,
                worst_usd=None if p_in is None else prompt_tokens * p_in + worst_out * p_out,
                issues=issues,
            )
        )
    return rows


def execute(
    cfg: RunConfig,
    attempts: list[Attempt],
    client: ModelClient,
    store: AttemptStore,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    spent = store.spent_usd()
    lock = threading.Lock()
    counts = {"ok": 0, "api_error": 0, "skipped_budget": 0}

    def one(a: Attempt) -> None:
        nonlocal spent
        with lock:
            if cfg.max_cost_usd is not None and spent >= cfg.max_cost_usd:
                counts["skipped_budget"] += 1
                return
        record: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "run": cfg.name,
            "settings_key": a.settings.key,
            "settings": a.settings.to_dict(),
            "family_id": a.task.family_id,
            "seed": a.task.seed,
            "instance_id": a.task.instance_id,
            "attempt": a.index,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt": a.task.prompt,
            "spec": a.task.spec,
        }
        try:
            completion = client.complete(a.task.prompt, a.settings)
        except ApiError as e:
            record.update(status="api_error", error=str(e), completion=None, grade=None)
            store.append(record)
            with lock:
                counts["api_error"] += 1
            log(f"API ERROR  {a.settings.key}  {a.task.instance_id}#{a.index}: {e}")
            return
        score = grade_response(get_family(a.task.family_id), a.task, completion.text)
        grade = score.to_dict()
        grade.pop("plot_data")  # regenerated from the response by the report step
        record.update(status="ok", error=None, completion=completion.to_dict(), grade=grade)
        store.append(record)
        with lock:
            spent += completion.cost_usd or 0.0
            counts["ok"] += 1
            total = spent
        flag = "FORMAT" if score.format_error else ("PASS" if score.passed else "fail")
        cost = "?" if completion.cost_usd is None else f"${completion.cost_usd:.4f}"
        log(
            f"{flag:6} {score.score:.3f}  {a.settings.key}  {a.task.instance_id}#{a.index}"
            f"  out={completion.completion_tokens} cost={cost}  total=${total:.4f}"
        )

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        list(pool.map(one, attempts))
    return {**counts, "spent_usd": spent}


def pending(attempts: list[Attempt], store: AttemptStore) -> list[Attempt]:
    done = store.done_keys()
    return [a for a in attempts if a.key not in done]
