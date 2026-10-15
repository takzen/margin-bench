from margin_bench.core import TaskFamily
from margin_bench.tasks.dsp.fir_lowpass_mask import FirLowpassMask

FAMILIES: dict[str, TaskFamily] = {f.id: f for f in (FirLowpassMask(),)}


def get_family(family_id: str) -> TaskFamily:
    try:
        return FAMILIES[family_id]
    except KeyError:
        raise KeyError(f"unknown task family {family_id!r}; known: {sorted(FAMILIES)}") from None
