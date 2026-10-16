"""Playground: models operate simulated plants through tools, and viewers watch."""

from __future__ import annotations

from margin_bench.play.env import Environment
from margin_bench.play.envs import heater, scale, tank

ENVIRONMENTS = {
    tank.TankLevel.id: tank.TankLevel,
    scale.ScaleReadout.id: scale.ScaleReadout,
    heater.InlineHeater.id: heater.InlineHeater,
}
# Scripted agents per environment: "reference" passes, the ones in NEGATIVE fail.
SCRIPTED = {cls.id: mod.AGENTS for cls, mod in ((tank.TankLevel, tank), (scale.ScaleReadout, scale),
                                               (heater.InlineHeater, heater))}
NEGATIVE = {cls.id: mod.NEGATIVE_AGENTS for cls, mod in ((tank.TankLevel, tank), (scale.ScaleReadout, scale),
                                                        (heater.InlineHeater, heater))}


def make(env_id: str, seed: int) -> Environment:
    try:
        cls = ENVIRONMENTS[env_id]
    except KeyError:
        raise SystemExit(f"unknown environment {env_id!r}; known: {', '.join(sorted(ENVIRONMENTS))}") from None
    return cls(seed)
