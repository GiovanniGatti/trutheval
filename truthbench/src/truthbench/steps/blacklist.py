import re
from typing import Set, Dict, Any

from truthbench.pipeline import Step


class BlacklistItemsFromQuestionStep(Step):
    """
    Pipeline step that filters out factual items which contain words already present in the question.

    For each item in `raw_factual_data`, this step checks whether any token in the item
    (split by whitespace) overlaps with the question tokens. If so, the item is added to
    the `blacklisted` list in the sample. Tokens are lowercased and stripped of punctuation.

    Stop words can be provided to exclude common words from the question during matching.

    Attributes:
        - stop_words (Set[str]): A set of words to ignore when matching against the question.

    Expected Sample Fields:
        - question (str): The question text.
        - raw_factual_data (List[str]): List of candidate factual items.

    Modifies:
        - sample["blacklisted"] (List[str] | None): List of lowercased blacklisted items,
          or None if the question or factual data is empty.

    Counter:
        - Increments no counters

    Notes:
        - This step does not increment or modify the tracker.
        - Matching is case-insensitive and ignores punctuation.
        - If either `question` or `raw_factual_data` is empty, `blacklisted` is set to None.

    Example:
        question: "What is climate change?"
        raw_factual_data: ["Climate models", "Carbon emissions", "Solar activity"]
        -> blacklisted: ["climate models", "carbon emissions"]
    """

    def __init__(self, stop_words: Set[str]):
        self._stop_words = stop_words
        super().__init__(
            required_fields=frozenset({"question", "raw_factual_data"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if not sample["question"] or not sample["raw_factual_data"]:
            sample["blacklisted"] = None
            return

        # Simple tokenization of the question
        # (strip punctuation, lowercase, then split on whitespace)
        question_words = set(re.findall(r"\w+", sample["question"].lower()))
        question_words = question_words - self._stop_words
        sample["blacklisted"] = [
            term.lower() for term in sample["raw_factual_data"]
            if any(word.lower() in question_words for word in term.split())
        ]
