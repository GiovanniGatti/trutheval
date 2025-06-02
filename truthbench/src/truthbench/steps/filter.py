import math
from typing import Dict, Any

from truthbench.pipeline import Step


class FilterFactualDataStep(Step):
    """
        Filters a ranked list of factual data to retain only the top items that are not blacklisted.

        This step takes as input a list of ranked factual spans (e.g., based on informativeness or reliability)
        and a set of blacklisted strings that should be excluded. It retains only the top-N% of spans,
        as defined by the `keep` parameter, and removes any items that are present in the blacklist.
        The result is stored in the `factual_data` field of the sample.

        Attributes:
            - keep (float): The proportion of ranked items to retain (must be in the range (0.0, 1.0]).

        Expected Sample Fields:
            - ranked_factual_data (List[str]): A list of spans ordered by decreasing priority.
            - blacklisted (Set[str] or List[str]): A set of strings that should be excluded from final output.

        Modifies:
            - factual_data (List[str] or None): The filtered and selected factual spans.

        Counter:
            - None.

        Notes:
            - If `ranked_factual_data` is empty or `blacklisted` is None, the step will output `factual_data = None`.
            - Filtering is case-insensitive (`lower()` is applied to each candidate before checking the blacklist).
            - The number of items kept is computed as `ceil(len(ranked_factual_data) * keep)`.

        Example:
            sample = {
            ...     "ranked_factual_data": ["Paris", "2021", "with confidence"],
            ...     "blacklisted": {"paris"}
            ... }
            step = FilterFactualDataStep(keep=0.66)
            step.step(sample, tracker={})
            sample["factual_data"]
            ['2021']
        """

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
