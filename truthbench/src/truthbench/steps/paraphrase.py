from typing import Protocol, Optional, Dict, TypeVar, Generic

from truthbench.pipeline import Step, LLM


class SupportsParaphrase(Protocol):
    ground_truth: Optional[str] = None
    answers: Dict[str, str] = {}


T = TypeVar('T', bound=SupportsParaphrase)


class ParaphraseStep(Step, Generic[T]):
    prompt = (
        "Rewrite the provided sentence to express the same idea in slightly different words while preserving "
        "full accuracy, completeness, and meaning. Ensure the content remains faithful to the original and includes "
        "all key details. Do not add any note.\n\n"
        "Original:\n{ground_truth}\n\n"
        "Paraphrased version:"
    )

    def __init__(self, llm: LLM):
        self._llm = llm
        super().__init__(required_fields=("ground_truth", "answers"))

    def step(self, sample: SupportsParaphrase, tracker: Tracker) -> None:
        assert sample.ground_truth is not None
        prompt = self.prompt.format(ground_truth=sample.ground_truth)
        paraphrased = self._llm.query([{"role": "user", "content": prompt}])
        sample.answers["A0"] = paraphrased
