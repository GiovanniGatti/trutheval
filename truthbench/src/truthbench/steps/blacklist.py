import re
from typing import Set, Protocol, Optional, List, Generic, TypeVar

from truthbench.pipeline import Step


class SupportsBlacklisting(Protocol):
    question: Optional[str] = None
    raw_factual_data: Optional[List[str]] = []
    blacklisted: Optional[List[str]] = None

    def is_initialized(self) -> bool:
        return False


T = TypeVar('T', bound=SupportsBlacklisting)


class BlacklistItemsFromQuestionStep(Step, Generic[T]):
    """
    Filters out any factual item that has overlapping words with the question.
    If any token in the bracketed item appears in the question, we skip that item.
    """

    def __init__(self, stop_words: Set[str]):
        self._stop_words = stop_words
        super().__init__(
            required_fields=("question", "raw_factual_data", "blacklisted")
        )

    def step(self, sample: SupportsBlacklisting, tracker: Tracker) -> None:
        super().step(sample, tracker)
        if not sample.is_initialized():
            return

        assert sample.question is not None
        assert sample.raw_factual_data is not None
        assert len(sample.raw_factual_data) > 0

        # Simple tokenization of the question
        # (strip punctuation, lowercase, then split on whitespace)
        question_words = set(re.findall(r"\w+", sample.question.lower()))
        question_words = question_words - self._stop_words
        sample.blacklisted = [
            term.lower() for term in sample.raw_factual_data
            if any(word.lower() in question_words for word in term.split())
        ]
