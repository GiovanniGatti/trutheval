import math
from typing import Protocol, List, Optional, TypeVar, Generic

from truthbench.pipeline import Step


class SupportsRanking(Protocol):
    ranked_factual_data: Optional[List[str]]
    blacklisted: Optional[List[str]]
    factual_data: Optional[List[str]]

    def is_ranked(self) -> bool:
        return False


T = TypeVar('T', bound=SupportsRanking)


class FilterFactualDataStep(Step, Generic[T]):

    def __init__(self, keep: float = 0.8):
        if not 0. < keep <= 1.:
            raise ValueError(f"Should be a percentage of items to keep, but got {keep}")

        self._keep = keep

        super().__init__(
            required_fields=("ranked_factual_data", "factual_data", "blacklisted"),
            required_validators=("is_ranked",)
        )

    def step(self, sample: SupportsRanking, tracker: Tracker) -> None:
        if not sample.is_ranked():
            return

        selected = sample.ranked_factual_data[:math.ceil(len(sample.ranked_factual_data) * self._keep)]
        sample.factual_data = [s for s in selected if s.lower() not in sample.blacklisted]
