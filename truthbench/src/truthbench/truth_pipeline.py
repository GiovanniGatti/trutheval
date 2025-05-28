from typing import Optional

import spacy
from spacy import Language

from truthbench.steps.counter import CounterStep

try:
    from truthbench.llms.openai import GPT
    from openai import OpenAI
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


def truth_pipeline(
        llm: Optional[LLM] = None,
        nlp: Optional[Language] = None,
        stop_words: Optional[str] = None,
        with_progress: bool = True
) -> Pipeline:
    if nlp is None:
        try:
            nlp: Language = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("Install EN spacy language with python -m spacy download en_core_web_sm")

    if stop_words is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        stop_words = STOP_WORDS

    if llm is None:
        if GPT is None or OpenAI is None:
            raise ImportError("Install with: pip install truthbench[openai]")
        llm = GPT(OpenAI())

    return (
        Pipeline(with_progress)
        .with_step(ParaphraseStep(llm))
        .with_step(FactualDataStep(nlp))
        .with_step(BlacklistItemsFromQuestionStep(stop_words))
        .with_step(RankFactualDataStep(llm))
        .with_step(FilterFactualDataStep())
        .with_step(CreateNoiseExamplesStep(llm))
        .with_step(CounterStep())
    )
