import pathlib
import random
import re
from typing import List, Tuple, Protocol, Optional, Dict, Generic, TypeVar

from truthbench.pipeline import Step, LLM


def process_terms(text: str, allowed_terms: List[str]) -> str:
    allowed_terms = [t.lower() for t in allowed_terms]

    # Function to determine replacement for each matched bracketed term
    def replacer(match):
        term = match.group(1)
        if term.lower() in allowed_terms:
            allowed_terms.remove(term.lower())  # break repetition by taking the first occurrence
            return f'[{term}]'
        return f'{{{{{term}}}}}'

    # Use regex to find all bracketed terms and apply the replacer function
    processed_text = re.sub(r'\[([^]]+)]', replacer, text)
    return processed_text


class SupportsNoise(Protocol):
    factual_data: Optional[List[str]] = None
    with_brackets: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    thinking: Dict[str, str] = {}

    def is_ranked(self) -> bool:
        return False


T = TypeVar('T', bound=SupportsNoise)


class CreateNoiseExamplesStep(Step, Generic[T]):

    def __init__(self, llm: LLM, levels: int = 4):
        self._llm = llm
        self._prompt = pathlib.Path("prompt.txt").read_text()
        self._levels = levels

        super().__init__(
            required_fields=("factual_data", "with_brackets", "answers"),
            required_validators=("is_ranked",)
        )

    @classmethod
    def batch(cls, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    @classmethod
    def split_groups(cls, num_terms: int, num_groups: int) -> List[List[int]]:
        batches = list(CreateNoiseExamplesStep.batch(list(sorted(range(num_terms), reverse=True)), num_groups))
        groups = [[None, ] * len(batches) for _ in range(num_groups)]
        for j, idx in enumerate(batches):
            if j % 2 == 0:
                for i in range(num_groups):
                    groups[i][j] = idx[i] if i < len(idx) else None
            else:
                for _i, i in enumerate(reversed(range(num_groups))):
                    groups[i][j] = idx[_i] if _i < len(idx) else None

        for group in groups:
            while None in group:
                group.remove(None)

        return groups

    @classmethod
    def parse_response(cls, text: str) -> Tuple[str, str]:
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
        output_match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)

        return (
            thinking_match.group(1).strip() if thinking_match else None,
            output_match.group(1).strip() if output_match else None
        )

    def step(self, sample: SupportsNoise, tracker: Tracker) -> None:
        if not sample.is_ranked():
            return

        groups = CreateNoiseExamplesStep.split_groups(len(sample.factual_data), self._levels)
        random.shuffle(groups)
        a0 = sample.with_brackets["A0"]
        noised_sample = a0
        for i, group in enumerate(groups, start=1):
            selected = [sample.factual_data[j] for j in group]
            input_sample = process_terms(noised_sample, selected)
            prompt = f"```\n{input_sample}\n```"

            output_sample = self._llm.query(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            thinking, output = CreateNoiseExamplesStep.parse_response(output_sample)

            if thinking:
                sample.thinking[f"A{i}"] = thinking

            if output:
                noised_sample = re.sub(r'\{\{(.*?)}}', r'[\1]', output)
                sample.with_brackets[f"A{i}"] = noised_sample
                cleaned = re.sub(r'\{\{(.*?)}}', r'\1', output)
                sample.answers[f"A{i}"] = cleaned
