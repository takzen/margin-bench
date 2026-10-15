"""Playground: models operate simulated plants through tools, and viewers watch."""

from __future__ import annotations

from margin_bench.play.env import Environment
from margin_bench.play.envs import tank
from margin_bench.play.envs.tank import TankLevel

ENVIRONMENTS = {TankLevel.id: TankLevel}
# Scripted agents per environment: "reference" passes, the ones in NEGATIVE fail.
SCRIPTED = {TankLevel.id: tank.AGENTS}
NEGATIVE = {TankLevel.id: tank.NEGATIVE_AGENTS}


def make(env_id: str, seed: int) -> Environment:
    try:
        cls = ENVIRONMENTS[env_id]
    except KeyError:
        raise SystemExit(f"unknown environment {env_id!r}; known: {', '.join(sorted(ENVIRONMENTS))}") from None
    return cls(seed)
