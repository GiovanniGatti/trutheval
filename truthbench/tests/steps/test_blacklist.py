import re
import unittest

import pytest

from truthbench.steps.blacklist import BlacklistItemsFromQuestionStep


def test_blacklisting_basic():
    step = BlacklistItemsFromQuestionStep(stop_words=set())
    sample = {
        "question": "What causes wildfires in California?",
        "raw_factual_data": ["forest density", "Wildfires", "in the forest", "drought"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert "wildfires" in sample["blacklisted"]
    assert "forest density" not in sample["blacklisted"]
    assert "in the forest" in sample["blacklisted"]
    assert "drought" not in sample["blacklisted"]


def test_blacklisting_with_stop_words():
    step = BlacklistItemsFromQuestionStep(stop_words={"what", "causes", "in"})
    sample = {
        "question": "What causes wildfires in California?",
        "raw_factual_data": ["forest density", "Wildfires", "in the forest", "drought"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert "wildfires" in sample["blacklisted"]
    assert "forest density" not in sample["blacklisted"]
    assert "in the forest" not in sample["blacklisted"]
    assert "drought" not in sample["blacklisted"]


def test_blacklisting_case_insensitive():
    step = BlacklistItemsFromQuestionStep(stop_words=set())
    sample = {
        "question": "What Is Quantum Physics?",
        "raw_factual_data": ["quantum mechanics", "relativity", "Quantum entanglement"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert set(sample["blacklisted"]) == {"quantum mechanics", "quantum entanglement"}


def test_blacklisting_no_matches():
    step = BlacklistItemsFromQuestionStep(stop_words=set())
    sample = {
        "question": "How do airplanes fly?",
        "raw_factual_data": ["Wing shape", "Air pressure", "Jet engine"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert sample["blacklisted"] == []


def test_empty_question_returns_none():
    step = BlacklistItemsFromQuestionStep(stop_words=set())
    sample = {
        "question": "",
        "raw_factual_data": ["Some fact"]
    }
    tracker = {}

    step.step(sample, tracker)

    assert sample["blacklisted"] is None


def test_on_empty_factual_data():
    step = BlacklistItemsFromQuestionStep(stop_words=set())
    sample = {
        "question": "Why is the sky blue?",
        "raw_factual_data": []
    }
    tracker = {}

    step.step(sample, tracker)

    assert sample["blacklisted"] is None


@pytest.mark.parametrize(
    "sample,error", [
        (
                {"x": "Why is the sky blue?", "raw_factual_data": []},
                r"BlacklistItemsFromQuestionStep requires ['question', 'raw_factual_data'], but some are missing from "
                r"the sample: ['question']. Check pipeline dependencies before proceeding."
        ),
        (
                {"question": "Why is the sky blue?", "x": []},
                r"BlacklistItemsFromQuestionStep requires ['question', 'raw_factual_data'], but some are missing from "
                r"the sample: ['raw_factual_data']. Check pipeline dependencies before proceeding."
        ),
        (
                {"x": "Why is the sky blue?", "y": []},
                r"BlacklistItemsFromQuestionStep requires ['question', 'raw_factual_data'], but some are missing from "
                r"the sample: ['question', 'raw_factual_data']. Check pipeline dependencies before proceeding."
        ),
    ]
)
def test_validate(sample, error):
    step = BlacklistItemsFromQuestionStep(stop_words=set())

    with pytest.raises(ValueError, match=re.escape(error)):
        step.validate(sample)


if __name__ == "__main__":
    unittest.main()
