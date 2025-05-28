from typing import Dict, Any

from truthbench.pipeline import Step, LLM


class ParaphraseStep(Step):
    prompt = (
        "Rewrite the provided sentence to express the same idea in slightly different words while preserving "
        "full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes "
        "all key details. Do not add any note.\n\n"
        "Original:\n{ground_truth}\n\n"
        "Paraphrased version:"
    )

    def __init__(self, llm: LLM):
        self._llm = llm
        super().__init__(required_fields=frozenset({"ground_truth"}))

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if not sample["ground_truth"]:
            sample["answers"] = None
            return

        prompt = self.prompt.format(ground_truth=sample["ground_truth"])
        paraphrased = self._llm.query([{"role": "user", "content": prompt}])
        sample["answers"] = {}
        sample["answers"]["A0"] = paraphrased
