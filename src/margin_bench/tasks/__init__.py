from margin_bench.core import TaskFamily
from margin_bench.tasks.control.discrete_pid import DiscretePid
from margin_bench.tasks.control.loop_margins import LoopMargins
from margin_bench.tasks.control.pid_step_spec import PidStepSpec
from margin_bench.tasks.dsp.ema_shift import EmaShift
from margin_bench.tasks.dsp.fir_lowpass_mask import FirLowpassMask
from margin_bench.tasks.dsp.q15_biquad import Q15Biquad
from margin_bench.tasks.electronics.divider_tolerance import DividerTolerance
from margin_bench.tasks.electronics.opamp_gain_bw import OpampGainBw
from margin_bench.tasks.electronics.sallen_key_e24 import SallenKeyE24

FAMILIES: dict[str, TaskFamily] = {
    f.id: f
    for f in (
        FirLowpassMask(),
        Q15Biquad(),
        EmaShift(),
        SallenKeyE24(),
        OpampGainBw(),
        DividerTolerance(),
        PidStepSpec(),
        LoopMargins(),
        DiscretePid(),
    )
}

# Every family is approved by the author before it counts. Anything not listed is awaiting review.
REVIEW_STATUS: dict[str, str] = {
    "dsp.fir_lowpass_mask": "accepted",
}


def get_family(family_id: str) -> TaskFamily:
    try:
        return FAMILIES[family_id]
    except KeyError:
        raise KeyError(f"unknown task family {family_id!r}; known: {sorted(FAMILIES)}") from None
