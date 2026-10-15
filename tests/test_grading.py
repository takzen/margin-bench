import pytest

from margin_bench.grading import grade_response, scoring
from margin_bench.grading.parse import extract_json
from margin_bench.tasks import get_family


def test_prefers_last_fenced_block():
    text = 'Draft:\n```json\n{"taps": [1]}\n```\nFinal:\n```json\n{"taps": [2]}\n```\n'
    assert extract_json(text) == {"taps": [2]}


def test_skips_broken_fenced_block():
    text = '```json\n{"taps": [1]}\n```\n```json\n{"taps": [2,\n```'
    assert extract_json(text) == {"taps": [1]}


def test_falls_back_to_bare_json():
    text = 'I chose a Kaiser window. {"note": {"beta": 5}} Answer: {"taps": [0.25, 0.5, 0.25]} done.'
    assert extract_json(text) == {"taps": [0.25, 0.5, 0.25]}


def test_no_json_returns_none():
    assert extract_json("I cannot design this filter.") is None
    assert extract_json("[1, 2, 3]") is None


def test_grade_response_end_to_end():
    family = get_family("dsp.fir_lowpass_mask")
    task = family.generate(0)
    taps = family.reference(task)["taps"]
    text = "Equiripple design.\n```json\n" + f'{{"taps": {taps}}}' + "\n```"
    assert grade_response(family, task, text).passed
    no_answer = grade_response(family, task, "Sorry, no taps.")
    assert no_answer.format_error and no_answer.score == 0.0


def test_scoring_ratios():
    assert scoring.at_least(60, 60) == 1.0
    assert scoring.at_least(30, 60) == 0.5
    assert scoring.at_least(-5, 60) == 0.0
    assert scoring.at_most(0.25, 0.5) == 1.0
    assert scoring.at_most(1.0, 0.5) == 0.5
    assert scoring.at_least(float("nan"), 60) == 0.0
    with pytest.raises(ValueError):
        scoring.at_least(1, 0)
