"""Local viewer: serves the site and the episode files, so a run can be watched live."""

from __future__ import annotations

import json
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

EPISODES_PREFIX = "/episodes/"


class _Handler(SimpleHTTPRequestHandler):
    episodes_dir: Path

    def translate_path(self, path: str) -> str:
        p = unquote(urlsplit(path).path)
        if p.startswith(EPISODES_PREFIX):
            name = Path(p[len(EPISODES_PREFIX):]).name  # no directories, no traversal
            return str(self.episodes_dir / name)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # keep the terminal for the run
        pass


def start(site_dir: Path, episodes_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"episodes_dir": episodes_dir.resolve()})
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(site_dir.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def summary(data: dict[str, Any], episode_id: str) -> dict[str, Any]:
    result = data.get("result") or {}
    return {
        "id": episode_id,
        "env": data["env"],
        "agent": {k: data["agent"].get(k) for k in ("kind", "name")},
        "status": data["status"],
        "score": result.get("score"),
        "passed": result.get("passed"),
        "wall_s": data.get("wall_s"),
        "model_s": data.get("model_s"),
        "usage": data.get("usage"),
        "tool_calls": sum(1 for s in data["steps"] if s["kind"] == "tool"),
        "started_at": data.get("started_at"),
    }


def export(episodes: list[Path], out_dir: Path) -> list[dict[str, Any]]:
    """Copy finished episodes into the site and rewrite its index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = {e["id"]: e for e in json.loads(index_path.read_text(encoding="utf-8"))["episodes"]} if index_path.is_file() else {}
    for path in episodes:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "done":
            raise SystemExit(f"{path}: episode is not finished ({data.get('status')})")
        shutil.copyfile(path, out_dir / f"{path.stem}.json")
        index[path.stem] = summary(data, path.stem)
    rows = sorted(index.values(), key=lambda e: (e["env"]["id"], -(e["score"] or 0), e["id"]))
    index_path.write_text(json.dumps({"episodes": rows}, indent=1), encoding="utf-8")
    return rows
