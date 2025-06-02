from typing import Dict, Any

from truthbench.pipeline import Step


class CounterStep(Step):
    """
    Pipeline step that counts samples with the expected number of answers.

    This step checks whether the `answers` field exists and contains exactly the expected number
    of answer levels. If so, it increments the `output_samples` counter in the tracker.

    Attributes:
        - expected_levels (int): The required number of answer levels for a valid sample.

    Expected Sample Fields:
        - answers (List[Any]): A list of answers to be validated.

    Modifies:
        - It does not modify any field.

    Counter:
        - tracker["output_samples"] (int): Incremented by 1 if the sample passes the check.

    Notes:
        - This step does not modify the sample.
        - The sample is only counted if `answers` is present and its length matches the expected level.
        - Does nothing if `answers` is None or its length does not match `expected_levels`.

    Example:
        expected_levels: 3
        sample["answers"]: ["yes", "no", "maybe"]
        -> tracker["output_samples"] += 1
    """

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
