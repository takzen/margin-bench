"""Helpers shared by the control task families."""

from __future__ import annotations

import math

import control as ct
import numpy as np


def sig(x: float, digits: int = 2) -> float:
    """Round to `digits` significant digits."""
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def sig_up(x: float, digits: int = 2) -> float:
    """Round up to `digits` significant digits (a looser limit, never a tighter one)."""
    if x <= 0:
        raise ValueError("x must be positive")
    scale = 10.0 ** (math.floor(math.log10(x)) - digits + 1)
    return float(f"{math.ceil(x / scale - 1e-9) * scale:.{digits}g}")


def lag_plant(gain: float, taus: list[float], integrator: bool = False) -> ct.TransferFunction:
    """gain / (s^i * prod(tau*s + 1))"""
    den = np.array([1.0])
    for tau in taus:
        den = np.polymul(den, [tau, 1.0])
    if integrator:
        den = np.polymul(den, [1.0, 0.0])
    return ct.tf([gain], den)


def lag_plant_text(gain: float, taus: list[float], integrator: bool = False) -> str:
    factors = "".join(f"({tau:g} s + 1)" for tau in taus)
    if integrator:
        factors = "s " + factors
    return f"{gain:g} / ({factors})"


def is_stable(sys: ct.LTI) -> bool:
    poles = sys.poles()
    if sys.isdtime():
        return bool(np.all(np.abs(poles) < 1.0 - 1e-9))
    return bool(np.all(poles.real < -1e-9))


def pid(kp: float, ki: float, kd: float, tf: float) -> ct.TransferFunction:
    """Parallel PID with a first-order filter on the derivative: Kp + Ki/s + Kd s/(Tf s + 1)."""
    s = ct.tf("s")
    return kp + ki / s + kd * s / (tf * s + 1)


def settling_time(t: np.ndarray, y: np.ndarray, final: float, band: float) -> float:
    """Time after which |y - final| stays within band*|final|. inf if it never settles."""
    outside = np.abs(y - final) > band * abs(final)
    if not outside.any():
        return 0.0
    last = int(np.nonzero(outside)[0][-1])
    if last == len(t) - 1:
        return math.inf
    return float(t[last + 1])
