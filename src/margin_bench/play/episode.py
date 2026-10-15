"""Episode record: every model turn, every tool call, the plant trend and the result.

The record is rewritten to disk after every step, so a viewer polling the
file can follow a run live.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from margin_bench.core import Score
from margin_bench.play.env import Environment, ToolError

FORMAT = "margin-bench/play-episode/1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(x: Any) -> Any:
    """JSON-safe copy: infinities and NaN become None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


class Episode:
    def __init__(self, env: Environment, agent: dict[str, Any], path: Path | None = None) -> None:
        self.env = env
        self.path = path
        self._t0 = time.monotonic()
        self.data: dict[str, Any] = {
            "format": FORMAT,
            "env": {"id": env.id, "title": env.title, "domain": env.domain, "seed": env.seed},
            "agent": agent,
            "prompt": env.prompt(),
            "tools": [t["function"]["name"] for t in env.tools()],
            "status": "running",
            "error": None,
            "stop_reason": None,
            "started_at": _now(),
            "finished_at": None,
            "wall_s": 0.0,
            "model_s": 0.0,
            "usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0},
            "steps": [],
            "view": env.view(),
            "result": None,
        }
        self.save()

    @property
    def cost_usd(self) -> float:
        return self.data["usage"]["cost_usd"]

    def _step(self, step: dict[str, Any]) -> dict[str, Any]:
        step = {"i": len(self.data["steps"]), "wall_s": round(time.monotonic() - self._t0, 2), **step}
        self.data["steps"].append(step)
        return step

    def model_turn(
        self,
        text: str,
        *,
        reasoning: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        latency_s: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        cost_usd: float | None = None,
        finish_reason: str | None = None,
    ) -> None:
        u = self.data["usage"]
        if latency_s is not None:
            u["model_calls"] += 1
            self.data["model_s"] = round(self.data["model_s"] + latency_s, 2)
        for key, v in (("prompt_tokens", prompt_tokens), ("completion_tokens", completion_tokens),
                       ("reasoning_tokens", reasoning_tokens)):
            u[key] += v or 0
        u["cost_usd"] = round(u["cost_usd"] + (cost_usd or 0.0), 6)
        self._step({
            "kind": "model",
            "sim_t": round(self.env.sim_time, 1),
            "text": text,
            "reasoning": reasoning,
            "calls": [c["name"] for c in tool_calls or []],
            "latency_s": None if latency_s is None else round(latency_s, 2),
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "reasoning": reasoning_tokens},
            "cost_usd": cost_usd,
            "finish_reason": finish_reason,
        })
        self.save()

    def tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call on the environment and record it. Refused calls return {"error": ...}."""
        t0 = self.env.sim_time
        try:
            result = self.env.call(name, args)
            error = None
        except ToolError as e:
            result, error = {"error": str(e)}, str(e)
        self._step({
            "kind": "tool",
            "name": name,
            "args": args,
            "result": result,
            "error": error,
            "sim_t0": round(t0, 1),
            "sim_t1": round(self.env.sim_time, 1),
        })
        self.data["view"] = self.env.view()
        self.save()
        return result

    def bad_call(self, name: str, raw_args: str, error: str) -> dict[str, Any]:
        """Record a tool call whose arguments could not be parsed; the plant is not touched."""
        t = round(self.env.sim_time, 1)
        self._step({"kind": "tool", "name": name, "args": {"raw": raw_args}, "result": {"error": error},
                    "error": error, "sim_t0": t, "sim_t1": t})
        self.save()
        return {"error": error}

    def finish(self, score: Score | None, *, stop_reason: str, error: str | None = None) -> None:
        self.data["status"] = "error" if error else "done"
        self.data["error"] = error
        self.data["stop_reason"] = stop_reason
        self.data["finished_at"] = _now()
        self.data["wall_s"] = round(time.monotonic() - self._t0, 2)
        self.data["view"] = self.env.view()
        if score is not None:
            self.data["result"] = score.to_dict()
        self.save()

    def save(self) -> None:
        self.data["wall_s"] = round(time.monotonic() - self._t0, 2)
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_clean(self.data), separators=(",", ":")), encoding="utf-8")
        for attempt in range(20):  # a viewer may hold the file open for a moment on Windows
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        os.replace(tmp, self.path)
