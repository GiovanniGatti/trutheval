# TruthScore

``truthscore` is a **fast, modular reimplementation of RAGAS's FactualCorrectness metric**, supporting both open-weight
and hosted LLMs. It evaluates factual consistency between a user response and a reference passage by breaking down
answers into claims and verifying them using **Natural Language Inference (NLI)**.

It is a component of the [`trutheval`](../README.md) framework and is intended for **scalable, cost-efficient**
factuality evaluation.

---

## 🔍 What it does

1. **Claim Decomposition**: The LLM-generated response is split into atomic factual claims using a lightweight LLM.
2. **Entailment Scoring**: Each claim is passed to an NLI model with the reference passage as context.
3. **Final Score**: The score reflects how many claims are entailed by the context, in the range `[0.0, 1.0]`.

For more details,
see [FactualCorrectness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/).

---

## ✨ Key Features

- 🔁 **RAGAS-compatible**: Faithfully reimplements the `FactualCorrectness` metric logic from RAGAS
- ✅ **Open-weight LLM support**: Works with open-weight models (e.g., Gemma, LLaMA, Mistral via Ollama)
- 🧠 **Plug-and-play**: Swap in custom NLI models
- ⚙️ **GPU-accelerated**: Recommended for claim decomposition + NLI
- 🧪 **Evaluated**: Competitive benchmark results (see [`trutheval`](../README.md))

---

## 📦 Installation

For full open-weight support (LLM hosted with Ollama + CrossEncoders NLI):

```bash
pip install truthscore[open]
```

Otherwise, use the lightweight dependency pick the dependencies that best work for you with:

```bash
pip install truthscore
```

Regarding ollama installation, please check [Ollama](https://github.com/ollama/ollama?tab=readme-ov-file#ollama).

## 🚀 Quick Start

### 💡 Open-weight (fully local)

```python
from langchain_community.llms import OllamaLLM
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from truthscore import OpenFactualCorrectness

test_data = {
    "user_input": "What happened in Q3 2024?",
    "reference": "The company saw an 8% rise in Q3 2024, driven by strong marketing and product efforts.",
    "response": "The company experienced an 8% increase in Q3 2024 due to effective marketing strategies and product efforts."
}
sample = SingleTurnSample(**test_data)

evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434"))
metric = OpenFactualCorrectness(llm=evaluator_llm)
score = metric.single_turn_score(sample)

print(score)  # e.g. 1.0
```

### ☁️ Hosted LLM (e.g., OpenAI)

```python
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from truthscore import OpenFactualCorrectness

evaluator_llm = LangchainLLMWrapper(OpenAI())
metric = OpenFactualCorrectness(llm=evaluator_llm)

# test_data same as above
score = metric.single_turn_score(SingleTurnSample(**test_data))
```

### ⚙️ Custom NLI Models

```python
import torch
from langchain_community.llms import OllamaLLM
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from sentence_transformers import CrossEncoder

from truthscore import OpenFactualCorrectness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
nli_model.model.to(device)

evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="gemma3:27b", base_url="http://localhost:11434"))
metric = OpenFactualCorrectness(llm=evaluator_llm, nli_model=nli_model)

# test_data same as above
score = metric.single_turn_score(SingleTurnSample(**test_data))
```

## 📊 Background

This metric was evaluated across a 500-example benchmark using perturbation levels A0–A4 on top of the Google Natural
Questions dataset using [truthbench](../truthbench/README.md).

See full results in the [trutheval](../README.md#empirical-validation-of-factuality-metrics-using-trutheval) project.
