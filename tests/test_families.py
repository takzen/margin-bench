"""Checks every registered task family must pass before it enters the benchmark."""

import json
from functools import cache

import pytest

from margin_bench.tasks import FAMILIES

DEFAULT_SEEDS = 100
# Families whose generator runs simulations to set the spec are checked on fewer seeds.
SEEDS = {"control.pid_step_spec": 30, "dsp.ema_shift": 50}


@cache
def task(family_id, seed):
    return FAMILIES[family_id].generate(seed)


def seeds(family_id):
    return range(SEEDS.get(family_id, DEFAULT_SEEDS))


@pytest.fixture(params=sorted(FAMILIES), ids=sorted(FAMILIES))
def family(request):
    return FAMILIES[request.param]


def test_reference_passes(family):
    for seed in seeds(family.id):
        t = task(family.id, seed)
        score = family.grade(t, family.reference(t), with_plot=False)
        assert score.passed, (seed, score.metrics)
        assert score.score == 1.0
        assert not score.format_error


def test_negative_fails(family):
    for seed in seeds(family.id):
        t = task(family.id, seed)
        score = family.grade(t, family.negative(t), with_plot=False)
        assert not score.passed, (seed, score.metrics)
        assert 0.0 <= score.score < 1.0
        assert not score.format_error


def test_generation_is_deterministic(family):
    for seed in range(3):
        assert family.generate(seed) == task(family.id, seed)


def test_seeds_give_different_tasks(family):
    prompts = {task(family.id, seed).prompt for seed in range(20)}
    assert len(prompts) == 20


def test_malformed_answers_are_format_errors(family):
    t = task(family.id, 0)
    for answer in ({}, {"x": 1}, []):
        score = family.grade(t, answer)
        assert score.format_error and score.score == 0.0 and not score.passed


def test_results_are_json_serializable(family):
    t = task(family.id, 0)
    json.dumps(t.spec)
    for answer in (family.reference(t), family.negative(t)):
        score = family.grade(t, answer)
        json.dumps(score.to_dict(), allow_nan=False)


def test_plot_data_has_panels(family):
    t = task(family.id, 0)
    plot = family.grade(t, family.reference(t)).plot_data
    assert plot["panels"]
    for p in plot["panels"]:
        assert p["series"]
        for s in p["series"]:
            assert len(s["x"]) == len(s["y"]) > 1
