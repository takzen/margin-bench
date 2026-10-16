"""The ngspice wrapper, kept for the electronics environments."""

import pytest

from margin_bench.sims import ngspice

pytestmark = pytest.mark.skipif(not ngspice.available(), reason="ngspice shared library not found")


def test_operating_point_of_a_loaded_divider():
    r = ngspice.Ngspice.get().run(
        ["divider", "V1 in 0 DC 10", "R1 in out 1k", "R2 out 0 3k", "RL out 0 3k", ".op", ".end"],
        ["v(out)"],
    )
    # R2 || RL = 1.5k, so v(out) = 10 * 1.5 / 2.5
    assert r["v(out)"].real[0] == pytest.approx(6.0, rel=1e-6)


def test_ac_sweep_of_an_rc_lowpass():
    r = ngspice.Ngspice.get().run(
        ["rc", "V1 in 0 DC 0 AC 1", "R1 in out 1k", "C1 out 0 159.155n", ".ac dec 10 10 100k", ".end"],
        ["frequency", "v(out)"],
    )
    f, h = r["frequency"].real, abs(r["v(out)"])
    k = abs(f - 1000).argmin()
    assert h[k] == pytest.approx(2**-0.5, rel=0.01)  # -3 dB at 1 kHz
