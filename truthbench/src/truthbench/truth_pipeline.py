from typing import Optional, List, Dict

import pydantic
import spacy
from spacy import Language

try:
    from truthbench.llms.openai import GPT
except ImportError:
    GPT = None
    OpenAI = None

from truthbench.pipeline import Pipeline, LLM
from truthbench.steps.blacklist import BlacklistItemsFromQuestionStep
from truthbench.steps.factual import FactualDataStep
from truthbench.steps.filter import FilterFactualDataStep
from truthbench.steps.noise import CreateNoiseExamplesStep
from truthbench.steps.paraphrase import ParaphraseStep
from truthbench.steps.rank import RankFactualDataStep


class Sample(pydantic.BaseModel):
    question: Optional[str] = None
    ground_truth: Optional[str] = None
    raw_factual_data: Optional[List[str]] = []
    with_brackets: Dict[str, str] = {}
    thinking: Dict[str, str] = {}
    blacklisted: Optional[List[str]] = None
    factual_data: Optional[List[str]] = None
    ranked_factual_data: Optional[List[str]] = None
    answers: Dict[str, str] = {}

    def is_initialized(self) -> bool:
        return (
                self.question is not None and
                self.ground_truth is not None and
                "A0" in self.answers.keys() and
                self.with_brackets and
                self.raw_factual_data is not None and
                len(self.raw_factual_data) > 0
        )

    def is_ranked(self) -> bool:
        return (
                self.is_initialized() and
                self.blacklisted is not None and
                self.ranked_factual_data is not None and
                len(self.ranked_factual_data) == len(self.raw_factual_data)
        )

    def is_filtered(self) -> bool:
        return (
                self.is_ranked() and
                self.factual_data is not None and
                len(self.factual_data) <= len(self.ranked_factual_data)
        )

    def is_valid(self) -> bool:
        return (
                self.is_ranked() and
                self.answers.keys() == {"A0", "A1", "A2", "A3", "A4"} and
                self.thinking.keys() == {"A1", "A2", "A3", "A4"}
        )


def truth_pipeline(
        llm: Optional[LLM] = None,
        nlp: Optional[Language] = None,
        stop_words: Optional[str] = None,
        with_progress: bool = True
) -> Pipeline[Sample]:
    if nlp is None:
        try:
            nlp: Language = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("Install EN spacy language with python -m spacy download en_core_web_sm")

    if stop_words is None:
        from spacy.lang.en import stop_words as _stop_words
        stop_words = _stop_words

    if llm is None:
        if GPT is None or OpenAI is None:
            raise ImportError("Install with: pip install truthbench[openai]")
        llm = GPT(OpenAI())

    return (
        Pipeline[Sample](with_progress)
        .with_step(ParaphraseStep(llm))
        .with_step(FactualDataStep(nlp))
        .with_step(BlacklistItemsFromQuestionStep(stop_words))
        .with_step(RankFactualDataStep(llm))
        .with_step(FilterFactualDataStep())
        .with_step(CreateNoiseExamplesStep(llm))
    )
