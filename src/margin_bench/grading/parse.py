"""Extract the JSON answer from a model's free-text response."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the answer object, or None if the response has none.

    Prefers the last fenced code block holding a JSON object. Falls back to the
    last top-level JSON object anywhere in the text.
    """
    for block in reversed(_FENCE.findall(text)):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    decoder = json.JSONDecoder()
    last = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            last = obj
        i = text.find("{", end)
    return last
