from typing import Dict, Any

from truthbench.pipeline import Step


class CounterStep(Step):

    def __init__(self):
        super().__init__(
            required_fields=frozenset({"answers"}),
            counters=frozenset({"valid_samples", "rejected_samples"})
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if sample["answers"] and len(sample["answers"]) > 1:
            tracker["valid_samples"] += 1
            return

        tracker["rejected_samples"] += 1
