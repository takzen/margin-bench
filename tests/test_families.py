"""Checks every registered task family must pass before it enters the benchmark."""

import json

import pytest

from margin_bench.tasks import FAMILIES

SEEDS = range(100)


@pytest.fixture(params=sorted(FAMILIES), ids=sorted(FAMILIES))
def family(request):
    return FAMILIES[request.param]


def test_reference_passes(family):
    for seed in SEEDS:
        task = family.generate(seed)
        score = family.grade(task, family.reference(task))
        assert score.passed, (seed, score.metrics)
        assert score.score == 1.0
        assert not score.format_error


def test_negative_fails(family):
    for seed in SEEDS:
        task = family.generate(seed)
        score = family.grade(task, family.negative(task))
        assert not score.passed, (seed, score.metrics)
        assert 0.0 <= score.score < 1.0
        assert not score.format_error


def test_generation_is_deterministic(family):
    for seed in range(10):
        assert family.generate(seed) == family.generate(seed)


def test_seeds_give_different_tasks(family):
    prompts = {family.generate(seed).prompt for seed in range(20)}
    assert len(prompts) == 20


def test_results_are_json_serializable(family):
    task = family.generate(0)
    json.dumps(task.spec)
    json.dumps(family.grade(task, family.reference(task)).to_dict())
    json.dumps(family.grade(task, family.negative(task)).to_dict())
