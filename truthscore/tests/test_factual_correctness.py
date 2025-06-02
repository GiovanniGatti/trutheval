from unittest.mock import MagicMock, AsyncMock

import numpy as np
from langchain_core.outputs import LLMResult, Generation
from ragas import SingleTurnSample

from truthscore import OpenFactualCorrectness
from truthscore.factual_correctness import ClaimDecompositionOutput


def test_sanity_check():
    # Create realistic sample
    test_data = {
        "user_input": "What happened in Q3 2024?",
        "reference": "The company saw an 8% rise in Q3 2024, primarily driven by successful marketing efforts and "
                     "product adjustments, and is anticipated to keep growing in the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies "
                    "and product adaptation, with expectations of continued growth in the coming quarter.",
    }
    sample = SingleTurnSample(**test_data)

    llm = AsyncMock()
    nli_model = MagicMock()

    llm.generate.return_value = LLMResult(generations=[[
        Generation(
            text=ClaimDecompositionOutput(claims=[
                "The company experienced an 8% increase in Q3 2024.",
                "This was due to effective marketing strategies and product adaptation.",
                "The company is expected to continue growing next quarter."
            ]).model_dump_json()
        )
    ]])
    nli_model.predict.return_value = np.array([[0.1, 0.8, 0.1]] * 3)

    # Initialize metric with mocks
    metric = OpenFactualCorrectness(llm=llm, nli_model=nli_model)

    score = metric.single_turn_score(sample)

    assert isinstance(score, float)
    assert 0.99 <= score <= 1.0
