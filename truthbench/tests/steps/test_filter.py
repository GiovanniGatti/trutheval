import re
import unittest

import pytest

from truthbench.steps.filter import FilterFactualDataStep


@pytest.mark.parametrize("keep", [-0.1, 0.0, 1.1])
def test_invalid_keep_parameter(keep):
    with pytest.raises(ValueError, match="Should be a percentage of items to keep"):
        FilterFactualDataStep(keep=keep)


def test_filter_step_filters_top_n_and_excludes_blacklist():
    step = FilterFactualDataStep(keep=0.5)
    sample = {
        "ranked_factual_data": ["Paris", "2021", "with confidence", "Eiffel Tower"],
        "blacklisted": {"paris", "eiffel tower"}
    }

    tracker = {}
    step.step(sample, tracker)

    # ceil(4 * 0.5) = 2 → take "Paris", "2021" → "Paris" is blacklisted
    assert sample["factual_data"] == ["2021"]


def test_filter_step_full_keep_without_blacklist():
    step = FilterFactualDataStep(keep=1.0)
    sample = {
        "ranked_factual_data": ["Berlin", "during WWII", "high population"],
        "blacklisted": set()
    }

    tracker = {}
    step.step(sample, tracker)

    assert sample["factual_data"] == ["Berlin", "during WWII", "high population"]


def test_filter_step_with_empty_ranked_data():
    step = FilterFactualDataStep()
    sample = {
        "ranked_factual_data": [],
        "blacklisted": {"anything"}
    }

    tracker = {}
    step.step(sample, tracker)

    assert sample["factual_data"] is None


def test_filter_step_with_none_blacklist():
    step = FilterFactualDataStep()
    sample = {
        "ranked_factual_data": ["something"],
        "blacklisted": None
    }

    tracker = {}
    step.step(sample, tracker)

    assert sample["factual_data"] is None


def test_filter_step_case_insensitive_blacklist():
    step = FilterFactualDataStep(keep=1.0)
    sample = {
        "ranked_factual_data": ["NASA", "Mars", "the Moon"],
        "blacklisted": {"nasa", "mars"}
    }

    tracker = {}
    step.step(sample, tracker)

    assert sample["factual_data"] == ["the Moon"]


@pytest.mark.parametrize(
    "sample, error",
    [
        (
                {"ranked_factual_data": ["...", "..."], "x": set()},
                r"FactualDataStep requires ['blacklisted', 'ranked_factual_data'], but some are missing from the "
                r"sample: ['blacklisted']. Check pipeline dependencies before proceeding."
        ),
        (
                {"x": ["...", "..."], "blacklisted": set()},
                r"FactualDataStep requires ['blacklisted', 'ranked_factual_data'], but some are missing from the "
                r"sample: ['ranked_factual_data']. Check pipeline dependencies before proceeding."
        ),
        (
                {"x": ["...", "..."], "y": set()},
                r"FactualDataStep requires ['blacklisted', 'ranked_factual_data'], but some are missing from the "
                r"sample: ['blacklisted', 'ranked_factual_data']. Check pipeline dependencies before proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = FilterFactualDataStep()

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
