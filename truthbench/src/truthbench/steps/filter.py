import math
from typing import Dict, Any

from truthbench.pipeline import Step


class FilterFactualDataStep(Step):

    def __init__(self, keep: float = 0.8):
        if not 0. < keep <= 1.:
            raise ValueError(f"Should be a percentage of items to keep, but got {keep}")

        self._keep = keep

        super().__init__(
            required_fields=frozenset({"ranked_factual_data", "blacklisted"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if not sample["ranked_factual_data"] or sample["blacklisted"] is None:
            sample["factual_data"] = None
            return

        selected = sample["ranked_factual_data"][:math.ceil(len(sample["ranked_factual_data"]) * self._keep)]
        sample["factual_data"] = [s for s in selected if s.lower() not in sample["blacklisted"]]
