"""Interactive environments: a simulated plant the model operates through tools.

The model gets a task and a set of tools (measure, change settings, let time
run). It works on the plant for as many steps as it needs, then submits. The
environment then runs a fixed acceptance test and grades it by simulation only.
"""

from __future__ import annotations

from typing import Any, Protocol

from margin_bench.core import Score


class ToolError(Exception):
    """A tool call the plant refuses (bad argument, budget used up). Shown to the model."""


def tool(name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    """An OpenAI-style function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


class Environment(Protocol):
    id: str  # e.g. "process.tank_level"
    domain: str
    title: str
    seed: int

    def prompt(self) -> str:
        """The task as the model reads it."""
        ...

    def tools(self) -> list[dict[str, Any]]:
        """Function schemas offered to the model."""
        ...

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call against the plant. Raises ToolError for refused calls."""
        ...

    @property
    def submitted(self) -> bool: ...

    @property
    def sim_time(self) -> float: ...

    def finish(self) -> Score:
        """Run the acceptance test with the plant as the model left it and grade it."""
        ...

    def view(self) -> dict[str, Any]:
        """Everything the viewer needs to animate the episode so far."""
        ...
