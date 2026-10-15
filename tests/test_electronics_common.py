from margin_bench.tasks.electronics.common import E12, E24, E96, eng, in_series, nearest, values


def test_series_sizes():
    assert (len(E12), len(E24), len(E96)) == (12, 24, 96)


def test_membership():
    assert in_series(4.7e-9, "E12")
    assert in_series(1e6, "E24")
    assert in_series(48.7, "E96")
    assert not in_series(4.8e-9, "E12")
    assert not in_series(0.0, "E24")
    assert not in_series(-10.0, "E24")


def test_values_and_nearest():
    assert list(values("E12", 1e3, 1.5e3)) == [1000.0, 1200.0, 1500.0]
    assert nearest(4800.0, "E24") == 4700.0
    assert eng(4.7e-9, "F") == "4.7 nF"
