# TruthScore

`truthscore` is a **fast, modular reimplementation of RAGAS's FactualCorrectness metric**, supporting both open-weight
and hosted LLMs. It evaluates factual consistency between a user response and a reference passage by breaking down
answers into claims and verifying them using **Natural Language Inference (NLI)**.

It is a metric component of the TruthBench framework and is intended for **scalable, cost-efficient** factuality
evaluation. [TruthBench](https://pypi.org/project/truthbench/) is the meta-evaluation framework: it applies controlled,
graded perturbations to ground-truth answers so that factuality metrics can be scored on how well their judgements track
the injected error severity. `truthscore` is an LLM+NLI factual-correctness metric that can be evaluated with it, and is
shipped as its own installable package. Both are described in
[our EvalLLM 2025 paper](https://aclanthology.org/2025.jeptalnrecital-evalllm.19/).

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
- 🧪 **Evaluated**: Competitive benchmark results (see [TruthBench](https://github.com/GiovanniGatti/truthbench/blob/main/README.md))

---

## 📦 Installation

For full open-weight support (LLM hosted with Ollama + CrossEncoders NLI):

```bash
pip install truthscore[open]
```

Otherwise, install the lightweight version and pick the dependencies that best suit your setup:

```bash
pip install truthscore
```

Regarding ollama installation, please check [Ollama](https://github.com/ollama/ollama?tab=readme-ov-file#ollama).

## 🚀 Quick Start

### 💡 Open-weight (fully local)

```python
from langchain_ollama import OllamaLLM
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
from langchain_ollama import OllamaLLM
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
Questions dataset using [truthbench](https://github.com/GiovanniGatti/truthbench/blob/main/truthbench/README.md).

See full results in the [project overview](https://github.com/GiovanniGatti/truthbench/blob/main/README.md#empirical-validation-of-factuality-metrics).

## Citation

If you use TruthScore in your research, please cite our EvalLLM 2025 paper:

```bibtex
@inproceedings{gharsallah-etal-2025-peut,
    title = "Peut-on faire confiance aux juges ? Validation de m{\'e}thodes d'{\'e}valuation de la factualit{\'e} par perturbation des r{\'e}ponses",
    author = {Gharsallah, Sarra  and
      Robaldo, Ad{\`e}le  and
      Tokareva, Mariia  and
      Gatti Pinheiro, Giovanni  and
      Guendouz, Ilyana  and
      Troncy, Rapha{\"e}l  and
      Papotti, Paolo  and
      Michiardi, Pietro},
    editor = "Bechet, Fr{\'e}d{\'e}ric  and
      Chifu, Adrian-Gabriel  and
      Pinel-Sauvagnat, Karen  and
      Favre, Benoit  and
      Maes, Eliot  and
      Nurbakova, Diana",
    booktitle = "Actes de l'atelier {\'E}valuation des mod{\`e}les g{\'e}n{\'e}ratifs (LLM) et challenge 2025 (EvalLLM)",
    month = "6",
    year = "2025",
    address = "Marseille, France",
    publisher = "ATALA {\&} ARIA",
    url = "https://aclanthology.org/2025.jeptalnrecital-evalllm.19/",
    pages = "228--252",
    language = "fra"
}
```
