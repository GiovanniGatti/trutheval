import unittest
from unittest.mock import MagicMock

import pytest

from truthbench.steps.rank import RankFactualDataStep


def test_rank_factual_data_success():
    llm = MagicMock()
    llm.query.return_value = "<thinking>...</thinking>\n\nOUTPUT: [1, 2, 0]"
    step = RankFactualDataStep(llm)
    sample = {
        "with_brackets": {"A0": "Ozone affects [climate] and [air quality] in [urban areas]."},
        "raw_factual_data": ["climate", "air quality", "urban areas"]
    }
    tracker = {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }

    step.step(sample, tracker)

    llm.query.assert_called_once_with(messages=[
        {'role': 'user', 'content': "Output the indexes of terms in square brackets [ ] from the text between "
                                    "triple backticks ``` by terms that shape what the text is about, who it involves, "
                                    "consequences, hard numbers, dates, and facts. Downrank marked terms that are "
                                    "vague references, general connectors, or dependent on other terms in square "
                                    "brackets. You are given a free space to decide your ranking strategy between the "
                                    "tags <thinking></thinking>\n\nExample:\n\n```\nRecent studies have shown "
                                    "[a correlation:0] between [social media use:1] and [increased anxiety:2] among "
                                    "[teenagers:3]. Although some researchers argue that online interaction can "
                                    "promote [social connection:4], others warn about [its impact:5] on "
                                    "[self-esteem:6] and [sleep patterns:7]. Debates intensified after a "
                                    "[whistleblower:8] revealed internal data from [a major tech company:9] indicating "
                                    "[awareness:10] of [these risks:11].\n```\n<thinking>\nThe main terms are "
                                    "[social media use:1], [teenagers:3], [a major tech company:9]...Did I forget "
                                    "any missing terms?...</thinking>\nOUTPUT: [1, 3, 9, 2, 8, 6, 7, 4, 10, 0, 11, 5]"
                                    "\n\nNow it's your turn.\n\n```\nOzone affects [climate:0] and [air quality:1] in "
                                    "[urban areas:2].\n```\n"}
    ])

    assert sample["ranked_factual_data"] == ["air quality", "urban areas", "climate"]
    assert tracker == {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }


def test_rank_factual_data_failure_due_to_broken_json():
    llm = MagicMock()
    llm.query.return_value = "<thinking>...</thinking>\n\nOUTPUT: [1, 2, 0"  # broken json
    step = RankFactualDataStep(llm=llm, max_retries=3)
    sample = {
        "with_brackets": {"A0": "Ozone affects [climate] and [air quality] in [urban areas]."},
        "raw_factual_data": ["climate", "air quality", "urban areas"]
    }
    tracker = {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }

    step.step(sample, tracker)

    assert tracker == {
        "ranking_factual_data_error": 1,
        "json_parse_ranking_error": 3,  # num of max_retries
        "index_ranking_error": 0,
    }


def test_rank_factual_data_failure_due_incorrect_format():
    llm = MagicMock()
    llm.query.return_value = "<thinking>...</thinking>\n\noutput: [1, 2, 0]"  # lowercase
    step = RankFactualDataStep(llm=llm, max_retries=3)
    sample = {
        "with_brackets": {"A0": "Ozone affects [climate] and [air quality] in [urban areas]."},
        "raw_factual_data": ["climate", "air quality", "urban areas"]
    }
    tracker = {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }

    step.step(sample, tracker)

    assert tracker == {
        "ranking_factual_data_error": 1,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }


def test_rank_factual_data_failure_due_duplicated_separator():
    llm = MagicMock()
    llm.query.return_value = "<thinking>...OUTPUT:...</thinking>\n\nOUTPUT: [1, 2, 0]"  # double OUTPUT
    step = RankFactualDataStep(llm=llm, max_retries=3)
    sample = {
        "with_brackets": {"A0": "Ozone affects [climate] and [air quality] in [urban areas]."},
        "raw_factual_data": ["climate", "air quality", "urban areas"]
    }
    tracker = {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }

    step.step(sample, tracker)

    assert tracker == {
        "ranking_factual_data_error": 1,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }


@pytest.mark.parametrize("output", [
    "[1, 2]",  # missing item
    "[1, 2, 0, 3]",  # unexpected item
    "[1, 2, 2, 0]",  # duplicated item
])
def test_rank_factual_data_failure_with_unexpected_items(output):
    llm = MagicMock()
    llm.query.return_value = f"<thinking>...</thinking>\n\nOUTPUT: {output}"
    step = RankFactualDataStep(llm=llm, max_retries=3)
    sample = {
        "with_brackets": {"A0": "Ozone affects [climate] and [air quality] in [urban areas]."},
        "raw_factual_data": ["climate", "air quality", "urban areas"]
    }
    tracker = {
        "ranking_factual_data_error": 0,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 0,
    }

    step.step(sample, tracker)

    assert tracker == {
        "ranking_factual_data_error": 1,
        "json_parse_ranking_error": 0,
        "index_ranking_error": 3,
    }


def test_rank_factual_data_skips_when_missing_fields():
    llm_mock = MagicMock()
    step = RankFactualDataStep(llm=llm_mock)
    sample = {"with_brackets": None, "raw_factual_data": None}
    tracker = {}

    step.step(sample, tracker)

    assert sample["ranked_factual_data"] is None
    llm_mock.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
