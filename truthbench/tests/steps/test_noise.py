import re
import unittest
from unittest import mock

import pytest

from truthbench.steps.noise import CreateNoiseExamplesStep


def test_creates_perturbed_variants():
    llm = mock.MagicMock()
    llm.query.return_value = (
        "<thinking>I need to replace term1 with something believable... Humm, maybe termX?</thinking>\n\n"
        "<output>This is a sentence with termX to be modified and another {{term2}} to stay unchanged.</output>"
    )
    step = CreateNoiseExamplesStep(llm=llm, levels=2)
    sample = {
        "answers": {
            "A0": "This is a sentence with term1 to be modified and another term2 to stay unchanged."
        },
        "with_brackets": {
            "A0": "This is a sentence with [term1] to be modified and another {{term2}} to stay unchanged."
        },
        "factual_data": ["term1"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert sample["thinking"]["A1"] == "I need to replace term1 with something believable... Humm, maybe termX?"
    assert sample["with_brackets"]["A1"] == (
        "This is a sentence with termX to be modified and another [term2] to stay unchanged."
    )
    assert sample["answers"]["A1"] == (
        "This is a sentence with termX to be modified and another term2 to stay unchanged."
    )

    llm.query.assert_called_once_with([
        {"role": "system", "content": CreateNoiseExamplesStep.PROMPT},
        {"role": "user", "content": "```\nThis is a sentence with [term1] to be modified and another {{term2}} "
                                    "to stay unchanged.\n```"}
    ])


def test_llm_failed_to_comply_with_thinking_formatting():
    llm = mock.MagicMock()
    llm.query.return_value = (
        "<think>I need to replace term1 with something believable... Humm, maybe termX?</think>\n\n"  # not expected tag
        "<output>This is a sentence with termX to be modified and another {{term2}} to stay unchanged.</output>"
    )
    step = CreateNoiseExamplesStep(llm=llm, levels=2)
    sample = {
        "answers": {
            "A0": "This is a sentence with term1 to be modified and another term2 to stay unchanged."
        },
        "with_brackets": {
            "A0": "This is a sentence with [term1] to be modified and another {{term2}} to stay unchanged."
        },
        "factual_data": ["term1"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert "A1" not in sample["thinking"]
    assert sample["with_brackets"]["A1"] == (
        "This is a sentence with termX to be modified and another [term2] to stay unchanged."
    )
    assert sample["answers"]["A1"] == (
        "This is a sentence with termX to be modified and another term2 to stay unchanged."
    )


def test_llm_failed_to_comply_with_output_formatting():
    llm = mock.MagicMock()
    llm.query.return_value = (
        "<thinking>I need to replace term1 with something believable... Humm, maybe termX?</thinking>\n\n"
        # failed closing tag
        "<output>This is a sentence with termX to be modified and another {{term2}} to stay unchanged.<|output>"
    )
    step = CreateNoiseExamplesStep(llm=llm, levels=2)
    sample = {
        "answers": {
            "A0": "This is a sentence with term1 to be modified and another term2 to stay unchanged."
        },
        "with_brackets": {
            "A0": "This is a sentence with [term1] to be modified and another {{term2}} to stay unchanged."
        },
        "factual_data": ["term1"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert sample["thinking"]["A1"] == "I need to replace term1 with something believable... Humm, maybe termX?"
    assert "A1" not in sample["with_brackets"]
    assert "A1" not in sample["answers"]


@pytest.mark.parametrize("sample", [
    {
        "answers": {
            "A0": "..."
        },
        "with_brackets": {
            "A0": "..."
        },
        "factual_data": []  # missing factual data
    },
    {
        "answers": {
            "AX": "..."  # wrong key
        },
        "with_brackets": {
            "A0": "..."
        },
        "factual_data": ["..."]
    },
    {
        "answers": {
            "A0": "..."
        },
        "with_brackets": {
            "AX": "..."  # wrong key
        },
        "factual_data": ["..."]
    },
])
def test_missing_required_dependencies(sample):
    step = CreateNoiseExamplesStep(llm=mock.MagicMock(), levels=2)
    tracker = {}

    step.step(sample, tracker)

    assert not sample["thinking"]


@pytest.mark.parametrize("level", [-1, 0, 1])
def test_bad_level(level):
    with pytest.raises(ValueError, match="Number of noisy levels must be larger than 2."):
        CreateNoiseExamplesStep(llm=mock.MagicMock(), levels=level)


@pytest.mark.parametrize("num_terms,num_groups,expected", [
    (1, 1, [[0]]),
    (1, 3, [[0], [], []]),
    (5, 3, [[4], [3, 0], [2, 1]]),
    (10, 4, [[9, 2, 1], [8, 3, 0], [7, 4], [6, 5]])
])
def test_split_groups(num_terms, num_groups, expected):
    step = CreateNoiseExamplesStep(llm=mock.MagicMock(), levels=2)
    groups = step.split_groups(num_terms, num_groups)

    assert groups == expected


@pytest.mark.parametrize(
    "sample, error",
    [
        (
                {"x": {"A0": "..."}, "with_brackets": {"A0": "..."}, "factual_data": ["..."]},
                r"CreateNoiseExamplesStep requires ['answers', 'factual_data', 'with_brackets'], but some are missing "
                r"from the sample: ['answers']. Check pipeline dependencies before proceeding."
        ),
        (
                {"answers": {"A0": "..."}, "x": {"A0": "..."}, "factual_data": ["..."]},
                r"CreateNoiseExamplesStep requires ['answers', 'factual_data', 'with_brackets'], but some are missing "
                r"from the sample: ['with_brackets']. Check pipeline dependencies before proceeding."
        ),
        (
                {"answers": {"A0": "..."}, "with_brackets": {"A0": "..."}, "x": ["..."]},
                r"CreateNoiseExamplesStep requires ['answers', 'factual_data', 'with_brackets'], but some are missing "
                r"from the sample: ['factual_data']. Check pipeline dependencies before proceeding."
        ),
        (
                {"x": {"A0": "..."}, "y": {"A0": "..."}, "z": ["..."]},
                r"CreateNoiseExamplesStep requires ['answers', 'factual_data', 'with_brackets'], but some are missing "
                r"from the sample: ['answers', 'factual_data', 'with_brackets']. Check pipeline dependencies before "
                r"proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = CreateNoiseExamplesStep(llm=mock.MagicMock())

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
