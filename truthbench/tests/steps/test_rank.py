import unittest
from unittest.mock import MagicMock

import pytest

from truthbench.steps.rank import RankFactualDataStep


def test_rank_factual_data_success():
    llm = MagicMock()
    llm.query.return_value = "<thinking>...</thinking>\n\nOUTPUT: [1, 2, 0]"
    step = RankFactualDataStep(llm)
    sample = {
        "question": "What does the ozone gas?",
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
        {'role': 'user', 'content': "Output the indexes of terms in square brackets [ ] from the text between triple "
                                    "backticks ``` by terms that shape what the text is about, how it answers the "
                                    "question the text is answering, who it involves, consequences, hard numbers, "
                                    "dates, and facts. Downrank marked terms that are vague references, general "
                                    "connectors, dependent on other terms in square brackets, or are unrelated to "
                                    "the question. You are given a free space to decide your ranking strategy between "
                                    "the tags <thinking></thinking>\n\nExample:\n\nQuestion: What is the relationship "
                                    "between social media use and mental health in teenagers?```\nRecent studies have "
                                    "shown [a correlation:0] between [social media use:1] and [increased anxiety:2] "
                                    "among [teenagers:3]. Although some researchers argue that online interaction "
                                    "can promote [social connection:4], others warn about [its impact:5] on "
                                    "[self-esteem:6] and [sleep patterns:7]. Debates intensified after a "
                                    "[whistleblower:8] revealed internal data from [a major tech company:9] indicating "
                                    "[awareness:10] of [these risks:11].\n```\n<thinking>\nThe question asks about the "
                                    "relationship between social media and mental health in teenagers.The most central "
                                    "concept is [social media use:1], as it's the main cause under investigation. "
                                    "[teenagers:3] is the primary affected group, making it essential. "
                                    "[a major tech company:9] is important because it provided key internal data, "
                                    "which adds weight to the argument.\n[increased anxiety:2] is one of the main "
                                    "documented effects, so it's ranked high. [whistleblower:8] is less central but "
                                    "still critical since they enabled the release of impactful information.\n"
                                    "[self-esteem:6] and [sleep patterns:7] are concrete, measurable consequences of "
                                    "social media use, so they deserve mid-level ranking. [social connection:4] is a "
                                    "counterpoint and thus relevant but slightly less important.\n[awareness:10] and "
                                    "[a correlation:0] are abstract and support other terms but are not impactful "
                                    "alone. [these risks:11] and [its impact:5] are vague or dependent on previous "
                                    "terms, so they are ranked lowest.\n</thinking>\nOUTPUT: [1, 3, 9, 2, 8, 6, 7, "
                                    "4, 10, 0, 11, 5]\n\nNow it's your turn.\n\nQuestion: What does the ozone gas?\n"
                                    "```\nOzone affects [climate:0] and [air quality:1] in [urban areas:2].\n```\n"}
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
        "question": "What does the ozone gas?",
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
        "question": "What does the ozone gas?",
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
        "question": "What does the ozone gas?",
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
        "question": "What does the ozone gas?",
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
    sample = {"question": None, "with_brackets": None, "raw_factual_data": None}
    tracker = {}

    step.step(sample, tracker)

    assert sample["ranked_factual_data"] is None
    llm_mock.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
