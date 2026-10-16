"""Playground: models operate simulated plants through tools, and viewers watch."""

from __future__ import annotations

from margin_bench.play.env import Environment
from margin_bench.play.envs import scale, tank

ENVIRONMENTS = {tank.TankLevel.id: tank.TankLevel, scale.ScaleReadout.id: scale.ScaleReadout}
# Scripted agents per environment: "reference" passes, the ones in NEGATIVE fail.
SCRIPTED = {tank.TankLevel.id: tank.AGENTS, scale.ScaleReadout.id: scale.AGENTS}
NEGATIVE = {tank.TankLevel.id: tank.NEGATIVE_AGENTS, scale.ScaleReadout.id: scale.NEGATIVE_AGENTS}


def make(env_id: str, seed: int) -> Environment:
    try:
        cls = ENVIRONMENTS[env_id]
    except KeyError:
        raise SystemExit(f"unknown environment {env_id!r}; known: {', '.join(sorted(ENVIRONMENTS))}") from None
    return cls(seed)
