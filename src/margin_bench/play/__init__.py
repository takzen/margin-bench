"""Playground: models operate simulated plants through tools, and viewers watch."""

from __future__ import annotations

from margin_bench.play.env import Environment
from margin_bench.play.envs import circuit, heater, motor, pendulum, scale, tank

ENVIRONMENTS = {
    tank.TankLevel.id: tank.TankLevel,
    scale.ScaleReadout.id: scale.ScaleReadout,
    heater.InlineHeater.id: heater.InlineHeater,
    motor.DcMotor.id: motor.DcMotor,
    pendulum.CartPendulum.id: pendulum.CartPendulum,
    circuit.FaultFinding.id: circuit.FaultFinding,
}
# Scripted agents per environment: "reference" passes, the ones in NEGATIVE fail.
_MODULES = ((tank.TankLevel, tank), (scale.ScaleReadout, scale), (heater.InlineHeater, heater),
            (motor.DcMotor, motor), (pendulum.CartPendulum, pendulum), (circuit.FaultFinding, circuit))
SCRIPTED = {cls.id: mod.AGENTS for cls, mod in _MODULES}
NEGATIVE = {cls.id: mod.NEGATIVE_AGENTS for cls, mod in _MODULES}


def make(env_id: str, seed: int) -> Environment:
    try:
        cls = ENVIRONMENTS[env_id]
    except KeyError:
        raise SystemExit(f"unknown environment {env_id!r}; known: {', '.join(sorted(ENVIRONMENTS))}") from None
    return cls(seed)
