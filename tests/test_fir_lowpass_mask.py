import pytest

from margin_bench.tasks import get_family

family = get_family("dsp.fir_lowpass_mask")


@pytest.fixture
def task():
    return family.generate(0)


def test_seed_0_spec_is_pinned(task):
    # The public dev set must not change silently. Update only on purpose.
    assert task.spec == {
        "sample_rate_hz": 48000.0,
        "passband_edge_hz": 7300.0,
        "stopband_edge_hz": 12300.0,
        "passband_ripple_db": 0.5,
        "stopband_atten_db": 60.0,
        "max_taps": 40,
    }


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"coefficients": [0.5, 0.5]},
        {"taps": []},
        {"taps": "0.5, 0.5"},
        {"taps": [0.5, "0.5"]},
        {"taps": [True, False]},
        {"taps": [0.5, float("nan")]},
        {"taps": [0.5, float("inf")]},
    ],
)
def test_malformed_answers_are_format_errors(task, answer):
    score = family.grade(task, answer)
    assert score.format_error
    assert not score.passed
    assert score.score == 0.0


def test_too_many_taps_scores_zero(task):
    taps = family.reference(task)["taps"] + [0.0]
    score = family.grade(task, {"taps": taps})
    assert not score.passed
    assert not score.format_error
    assert score.score == 0.0
    assert "exceeds" in score.error


def test_identity_filter_has_no_stopband_attenuation(task):
    score = family.grade(task, {"taps": [1.0]})
    assert score.metrics["passband_dev_db"] == pytest.approx(0.0, abs=1e-12)
    assert score.metrics["stopband_atten_db"] == pytest.approx(0.0, abs=1e-12)
    assert score.score == 0.0


def test_gain_error_gives_partial_score(task):
    # The reference scaled by 1.2 is +1.58 dB in the passband: fails a 0.5 dB mask.
    taps = [1.2 * t for t in family.reference(task)["taps"]]
    score = family.grade(task, {"taps": taps})
    assert not score.passed
    assert 0.0 < score.score < 1.0
    assert score.metrics["passband_ratio"] == pytest.approx(score.score)


def test_tap_order_does_not_matter(task):
    taps = family.reference(task)["taps"]
    forward = family.grade(task, {"taps": taps})
    backward = family.grade(task, {"taps": taps[::-1]})
    assert forward.metrics["stopband_atten_db"] == pytest.approx(backward.metrics["stopband_atten_db"])


def test_plot_data_marks_violations(task):
    good = family.grade(task, family.reference(task))
    bad = family.grade(task, family.negative(task))
    assert good.plot_data["violations"] == []
    assert bad.plot_data["violations"]
    assert len(good.plot_data["curve"]["x"]) == len(good.plot_data["curve"]["y"])
