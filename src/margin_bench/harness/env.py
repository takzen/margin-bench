"""Read secrets from the environment or a local .env file. Values are never logged."""

from __future__ import annotations

import os
from pathlib import Path


def get_secret(name: str, env_file: Path = Path(".env")) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip().removeprefix("export ").strip() == name:
            return val.strip().strip("'\"") or None
    return None
