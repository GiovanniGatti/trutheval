import re
import unittest
from unittest.mock import MagicMock

import pytest

from truthbench.steps.paraphrase import ParaphraseStep


def test_paraphrase_successful():
    llm = MagicMock()
    sample = {"ground_truth": "Water boils at 100 degrees Celsius."}
    llm.query.return_value = "Water reaches its boiling point at 100°C."

    tracker = {}
    step = ParaphraseStep(llm)
    step.step(sample, tracker)

    llm.query.assert_called_once_with([
        {'role': 'user', 'content': 'Rewrite the provided sentence to express the same idea in slightly different words'
                                    ' while preserving full accuracy, completeness, and meaning. Ensure the content '
                                    'remains faithful to the original and includes all key details. Do not add any '
                                    'note.\n\nOriginal:\nWater boils at 100 degrees Celsius.\n\nParaphrased version:'}
    ])
    assert "answers" in sample
    assert sample["answers"]["A0"] == "Water reaches its boiling point at 100°C."


def test_missing_ground_truth():
    sample = {"ground_truth": None}
    tracker = {}

    step = ParaphraseStep(llm=MagicMock())
    step.step(sample, tracker)

    assert sample["answers"] is None


@pytest.mark.parametrize(
    "sample, error",
    [
        (
                {"x": "..."},  # missing ground_truth
                r"ParaphraseStep requires ['ground_truth'], but some are missing from the sample: ['ground_truth']. "
                r"Check pipeline dependencies before proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = ParaphraseStep(llm=MagicMock())

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
