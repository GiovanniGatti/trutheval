import re
from typing import Set, Dict, Any

from truthbench.pipeline import Step


class BlacklistItemsFromQuestionStep(Step):
    """
    Filters out any factual item that has overlapping words with the question.
    If any token in the bracketed item appears in the question, we skip that item.
    """

    def __init__(self, stop_words: Set[str]):
        self._stop_words = stop_words
        super().__init__(
            required_fields=frozenset({"question", "raw_factual_data"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if not sample["question"]:
            sample["blacklisted"] = None
            return

        assert len(sample["raw_factual_data"]) > 0

        # Simple tokenization of the question
        # (strip punctuation, lowercase, then split on whitespace)
        question_words = set(re.findall(r"\w+", sample["question"].lower()))
        question_words = question_words - self._stop_words
        sample["blacklisted"] = [
            term.lower() for term in sample["raw_factual_data"]
            if any(word.lower() in question_words for word in term.split())
        ]
