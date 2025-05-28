from typing import Dict, Any

from truthbench.pipeline import Step


class CounterStep(Step):

    def __init__(self, expected_levels: int):
        self._expected_levels = expected_levels
        super().__init__(
            required_fields=frozenset({"answers"}),
            counters=frozenset({"output_samples"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if sample["answers"] and len(sample["answers"]) == self._expected_levels:
            tracker["output_samples"] += 1
            return
