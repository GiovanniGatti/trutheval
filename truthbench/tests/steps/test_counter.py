import re
import unittest

import pytest

from truthbench.steps.counter import CounterStep


def test_counter_increments_when_expected_levels_match():
    step = CounterStep(expected_levels=2)
    sample = {"answers": {"AO": "...", "A1": "..."}}
    tracker = {"output_samples": 0}

    step.step(sample, tracker)

    assert tracker["output_samples"] == 1


def test_counter_does_not_increment_on_mismatch():
    step = CounterStep(expected_levels=3)
    sample = {"answers": {"AO": "...", "A1": "..."}}  # only 2
    tracker = {"output_samples": 0}

    step.step(sample, tracker)

    assert tracker["output_samples"] == 0


@pytest.mark.parametrize("sample", ({"answers": {}}, {"answers": None}))
def test_counter_does_not_increment_on_empty_answers(sample):
    step = CounterStep(expected_levels=1)
    tracker = {"output_samples": 0}

    step.step(sample, tracker)

    assert tracker["output_samples"] == 0


def test_counter_multiple_calls():
    step = CounterStep(expected_levels=2)
    tracker = {"output_samples": 0}

    samples = [
        {"answers": {"AO": "...", "A1": "..."}},
        {"answers": {"AO": "..."}},
        {"answers": None},
        {"answers": {"AO": "...", "A1": "..."}}
    ]

    for sample in samples:
        step.step(sample, tracker)

    assert tracker["output_samples"] == 2


@pytest.mark.parametrize(
    "sample, error",
    [
        (
                {"x": {"A0": "...", "A1": "..."}},  # wrong key
                r"CounterStep requires ['answers'], but some are missing from the sample: ['answers']. Check pipeline "
                r"dependencies before proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = CounterStep(expected_levels=2)

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
