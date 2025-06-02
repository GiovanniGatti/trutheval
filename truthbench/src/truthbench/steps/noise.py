import random
import re
from typing import List, Tuple, Dict, Any, Optional

from truthbench.pipeline import Step, LLM


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class CreateNoiseExamplesStep(Step):
    """
    Iteratively generates controlled factual perturbations from an input sentence containing factual spans.

    This step uses a prompt-driven LLM to simulate plausible but incorrect or misleading edits to selected
    factual spans, preserving surface fluency while subtly altering semantics. Factual items to modify are
    marked with square brackets `[ ]`, while protected spans are enclosed in double curly braces `{{ }}`.
    Each round introduces a new level of perturbation by editing a different subset of factual spans.

    The prompt encourages the LLM to first brainstorm alternatives using `<thinking>` tags and then return
    the final rewritten version inside `<output>` tags. The number of perturbation levels is user-configurable.
    The final answer variants are added under `"with_brackets"` and `"answers"` fields for downstream use.

    Attributes:
        - prompt (str): Full prompt template used to instruct the LLM for perturbation. Uses
                        CreateNoiseExamplesStep.PROMPT if none is provided.
        - llm (LLM): An LLM interface capable of structured prompting and response parsing.
        - levels (int): Number of perturbation rounds to perform (A_1, A_2, ..., A_{N-1}).

    Expected Sample Fields:
        - "factual_data" (List[str]): List of factual spans to selectively perturb.
        - "with_brackets" (Dict[str, str]): A0 must contain the original bracketed sentence.
        - "answers" (Dict[str, str]): A0 must contain the original unbracketed response.

    Modifies:
        - "with_brackets" (Dict[str, str]): Adds perturbed variants as A1, A2, ..., An.
        - "answers" (Dict[str, str]): Adds cleaned (unbracketed) perturbed variants as A1, A2, ..., An.
        - "thinking" (Dict[str, str]): Stores LLM’s planning output for each perturbation level.

    Counter:
        - None

    Notes:
        - Perturbations are groupwise: each level changes a unique subset of factual spans.
        - Uses a zig-zag round-robin batching scheme to assign spans to levels.
        - Only makes minimal edits to maintain linguistic plausibility.
        - Double curly brace terms `{{term}}` are never altered and used to anchor unmodified content.
        - Designed for evaluation tasks like factual robustness, misinformation detection, or model probing.

    Example:
        Input sample:
            {
                "answers": {"A0": "The ozone layer protects the Earth by absorbing harmful radiation."},
                "with_brackets": {"A0": "The ozone layer protects [the Earth] by absorbing [harmful radiation]."},
                "factual_data": ["the Earth", "harmful radiation"]
            }

        After step:
            sample["with_brackets"]["A1"] → perturbed version with 1st group edited
            sample["answers"]["A1"] → cleaned, bracket-free version of A1
            sample["thinking"]["A1"] → model’s reasoning about replacements
    """

    PROMPT = """\
# Instructions
You are given a text with terms marked between square brackets [ ] and double curly braces {{ }}. Your goal is to modify the terms marked between square brackets [ ] to make the text incorrect, misleading, or omit critical information.

Each change must be semantically credible, contextually plausible, and linguistically natural to a non-expert reader. For example, a gas must be replaced with another gas, or a country must be replaced with another country. Avoid over-generalizing substitutions that dilute meaning (e.g., “gases” instead of “greenhouse gases”), unless vagueness is the intended form of misinformation. Also avoid replacing terms with obvious synonyms. Do not invent non-standard terminology or introduce substitutions that would appear absurd, obviously false, grammatically broken, or conceptually incoherent.

You can make small adaptations of nearby words ONLY for grammatical correctness. However, all terms between double curly braces {{ }} MUST remain identical as the input. Remove square brackets [ ] from changed terms. Retain the original sentence structure and style wherever possible.

You are given a free space for planning your strategy. For each replacement word, try to list two to three alternatives and why they are good choices before coming up with a final decision. Output this planning between the marks <thinking></thinking>.

Finally, produce the final raw output without any further notes, explanations, or formatting between he marks <output></output>.

# Example
```
The ozone layer protects [the Earth] by absorbing [harmful ultraviolet radiation] from [the Sun]. It is {{primarily}} found in [the stratosphere], a layer of the atmosphere. Concerns about ozone depletion rose in [the 1980s] after [the discovery] of [a hole] over {{Antarctica}}.
```
<thinking>
1. [the Earth]
    * Options: “living organisms,” “the biosphere”
    * Chosen: the biosphere — plausible and often used in environmental contexts, but shifts focus away from the planet itself to just living systems, subtly distorting the scope of the ozone layer’s protective effect.
2. [harmful ultraviolet radiation]
    * Options: “unharmful ultraviolet radiation,” “harmful infrared radiation,” “heat energy”
    * Chosen: harmful infrared radiation — sounds technical and solar-related, but it's not what the ozone layer blocks. Misleading but plausible.
3. [the Sun]
    * Options: “deep space,” “solar flares”
    * Chosen: deep space — vague and misleading; implies that source of radiation is a general space phenomenon rather than solar-specific.
4. [the stratosphere]
    * Options: “mesosphere,” “troposphere,” “ionosphere”
    * Chosen: troposphere — the lowest layer, where weather happens, not where ozone is concentrated. Still sounds reasonable to a non-expert.
5. [the 1980s]
     * Options: “the late 1990s” “the 1970s,” “the early 1990s”
     * Chosen: the late 1990s — shifts timeline by a bit, particularly when the problem became of concern for the general public.
6. [the discovery]
     * Options: “a theory,” “an assumption,” “a hypothesis”
     * Chosen: a theory — undermines scientific certainty subtly without being absurd.
7. [a hole]
     * Options: “an irregularity,” “a gap,” “a reduction”
     * Chosen: an irregularity — Very neutral, sounds like a small change rather than a serious issue, minimizing severity.
</thinking>

<output>The ozone layer protects the biosphere by absorbing harmful infrared radiation from deep space. It is {{primarily}} found in the troposphere, a layer of the atmosphere. Concerns about ozone depletion rose in the late 1990s after a theory of an irregularity over {{Antarctica}}.</output>
"""

    def __init__(self, llm: LLM, levels: int = 5, prompt: Optional[str] = None):
        if levels < 2:
            raise ValueError("Number of noisy levels must be larger than 2.")

        self._llm = llm
        self._prompt = prompt or CreateNoiseExamplesStep.PROMPT
        self._noise_levels = levels - 1

        super().__init__(
            required_fields=frozenset({"factual_data", "with_brackets", "answers"}),
        )

    def process_terms(self, text: str, allowed_terms: List[str]) -> str:
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

    def split_groups(self, num_terms: int, num_groups: int) -> List[List[int]]:
        batches = list(batch(list(sorted(range(num_terms), reverse=True)), num_groups))
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

    def parse_response(self, text: str) -> Tuple[str, str]:
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
        output_match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)

        return (
            thinking_match.group(1).strip() if thinking_match else None,
            output_match.group(1).strip() if output_match else None
        )

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        if (not sample["with_brackets"] or
                "A0" not in sample["with_brackets"] or
                not sample["factual_data"] or
                not sample["answers"] or
                "A0" not in sample["answers"]):
            sample["thinking"] = None
            return

        sample["thinking"] = {}
        groups = self.split_groups(len(sample["factual_data"]), self._noise_levels)
        random.shuffle(groups)
        a0 = sample["with_brackets"]["A0"]
        noised_sample = a0
        for i, group in enumerate(groups, start=1):
            selected = [sample["factual_data"][j] for j in group]
            input_sample = self.process_terms(noised_sample, selected)
            prompt = f"```\n{input_sample}\n```"

            output_sample = self._llm.query(
                [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            thinking, output = self.parse_response(output_sample)

            if thinking:
                sample["thinking"][f"A{i}"] = thinking

            if output:
                noised_sample = re.sub(r'\{\{(.*?)}}', r'[\1]', output)
                sample["with_brackets"][f"A{i}"] = noised_sample
                cleaned = re.sub(r'\{\{(.*?)}}', r'\1', output)
                sample["answers"][f"A{i}"] = cleaned
