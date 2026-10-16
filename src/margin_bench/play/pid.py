"""The PID block shared by the process environments.

Ideal form in percent of transmitter span, derivative on the measurement with
a Td/10 filter, optional first-order PV filter, optional anti-windup
(conditional integration) and an optional Smith predictor. On the switch to
automatic the integral term starts from the current output, so the
proportional term acts at once on any setpoint error; changing the tuning in
automatic is bumpless.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tuning:
    kc: float = 0.5
    ti_s: float = 300.0  # 0 disables integral action
    td_s: float = 0.0
    pv_filter_s: float = 0.0
    anti_windup: bool = False
    # Smith predictor model, in percent of span per percent of output; gain 0 switches it off.
    smith_gain: float = 0.0
    smith_tau_s: float = 0.0
    smith_delay_s: float = 0.0

    def to_dict(self, smith: bool = False) -> dict[str, Any]:
        d = {"kc": self.kc, "ti_s": self.ti_s, "td_s": self.td_s, "pv_filter_s": self.pv_filter_s,
             "anti_windup": self.anti_windup}
        if smith:
            d.update(smith_gain=self.smith_gain, smith_tau_s=self.smith_tau_s, smith_delay_s=self.smith_delay_s)
        return d


@dataclass
class Pid:
    span: float  # engineering units for 100 % of span
    ts: float  # sample time, s
    tuning: Tuning = field(default_factory=Tuning)
    auto: bool = False
    sp: float = 0.0  # engineering units
    out: float = 0.0  # %
    integ: float = 0.0
    pv_f: float | None = None
    pv_prev: float | None = None
    d: float = 0.0
    sm_fast: float = 0.0
    sm_buf: deque[float] | None = None

    @property
    def smith_on(self) -> bool:
        return self.tuning.smith_gain > 0

    def _pv_eff(self) -> float:
        pv = self.pv_f or 0.0
        if self.smith_on and self.sm_buf:
            pv += self.sm_fast - self.sm_buf[0]
        return pv

    def _p(self) -> float:
        return self.tuning.kc * (100.0 * self.sp / self.span - self._pv_eff())

    def bumpless(self) -> None:
        self.integ = self.out - self._p() - self.d

    def reset_smith(self) -> None:
        """Start the predictor model in steady state at the present output."""
        t = self.tuning
        self.sm_fast = t.smith_gain * self.out
        n = max(1, round(t.smith_delay_s / self.ts)) + 1
        self.sm_buf = deque([self.sm_fast] * n, maxlen=n)

    def sample(self, pv: float) -> float:
        t, ts = self.tuning, self.ts
        if self.smith_on:
            if self.sm_buf is None:
                self.reset_smith()
            a = ts / (t.smith_tau_s + ts) if t.smith_tau_s > 0 else 1.0
            self.sm_fast += a * (t.smith_gain * self.out - self.sm_fast)
            self.sm_buf.append(self.sm_fast)
        if self.pv_f is None or t.pv_filter_s <= 0:
            self.pv_f = pv
        else:
            self.pv_f += ts / (t.pv_filter_s + ts) * (pv - self.pv_f)
        pv_e = self._pv_eff()
        if self.pv_prev is None:
            self.pv_prev = pv_e
        if not self.auto:
            self.pv_prev, self.d = pv_e, 0.0
            return self.out
        e = 100.0 * self.sp / self.span - pv_e
        if t.td_s > 0:
            raw = -t.kc * t.td_s * (pv_e - self.pv_prev) / ts
            tf = t.td_s / 10.0
            self.d += ts / (tf + ts) * (raw - self.d)
        else:
            self.d = 0.0
        self.pv_prev = pv_e
        p = t.kc * e
        if t.ti_s > 0:
            di = t.kc * ts / t.ti_s * e
            trial = p + self.integ + di + self.d
            if not (t.anti_windup and ((trial > 100.0 and di > 0) or (trial < 0.0 and di < 0))):
                self.integ += di
        self.out = min(100.0, max(0.0, p + self.integ + self.d))
        return self.out

    def set_auto(self, sp: float) -> None:
        self.sp = sp
        if not self.auto:
            self.auto = True
            self.integ, self.d = self.out, 0.0

    def set_tuning(self, tuning: Tuning) -> None:
        smith_changed = (tuning.smith_gain, tuning.smith_tau_s, tuning.smith_delay_s) != (
            self.tuning.smith_gain, self.tuning.smith_tau_s, self.tuning.smith_delay_s)
        self.tuning = tuning
        if smith_changed:
            self.sm_buf = None
            if self.smith_on:
                self.reset_smith()
        if self.auto:
            self.bumpless()
