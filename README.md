![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12%20|%203.13-4a006a)
![Coverage](https://img.shields.io/badge/Coverage-87%25-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
<img src="https://github.com/user-attachments/assets/8d30499c-ab58-4e92-8e11-089c01515c63" alt="drawing" style="height:340px"/>
</p>

# TruthEval

`trutheval` is a modular framework for generating datasets with graded factual perturbations. These datasets are
designed to test and validate the effectiveness of factuality evaluation pipelines.

It provides:

* 📚 Tools to generate Q&A datasets with controlled factual perturbations (A0–A4 levels)
* 🧪 Evaluation scripts and interfaces for comparing human vs. model-based assessments (see [UI](#ui))
* 📊 Data and metrics to support validation of factual scoring algorithms (see [datasets](#datasets))

Core components:

* [truthbench](./truthbench/README.md): Generates factual QA datasets with controlled perturbations. It's intended to
  support the development, tuning, and validation of factuality metrics and hallucination detection systems
* [truthscore](./truthscore/README.md): A fast, open-weight alternative to RAGAS using NLI models for factual
  scoring—cheaper, faster, and comparably effective. It's intended to evaluate LLMs directly.

Our framework includes extensive experimental validation, where we generate datasets with graduated factual
perturbations and benchmark diverse evaluation techniques — including open-weight LLMs and specialized
pipelines — demonstrating strong correlation between perturbation severity and factuality scores.

This work is described in detail in an accepted paper at the EvalLLM 2025 workshop (CORIA-TALN).

# Empirical validation of factuality metrics using `trutheval`

We evaluated how well different factuality scoring methods track increasing degrees of factual perturbation using 500
perturbed examples generated from 100 Q&A pairs
from the [Google Natural Questions](https://ai.google.com/research/NaturalQuestions/) dataset. The table below
summarizes the correlation between the intended perturbation levels (A0 to A4) and the factuality scores assigned by
each method.

| Method           | LLM                   | Pearson (95% CI)          | Kendall (Tau) | Kendall (95% CI) |
|------------------|-----------------------|---------------------------|---------------|------------------|
| **LLM-as-judge** | gemma3: 4b            | -0.63 \[-0.69, -0.58]     | -0.79         | \[-0.82, -0.77]  |
|                  | llama3.3: 70b         | -0.74 \[-0.78, -0.70]     | -0.86         | \[-0.88, -0.84]  |
|                  | mistral-small3.1: 24b | -0.71 \[-0.75, -0.66]     | -0.76         | \[-0.79, -0.72]  |
|                  | phi4: 14b             | -0.74 \[-0.78, -0.70]     | -0.81         | \[-0.83, -0.78]  |
|                  | prometheus-v2: 7b     | -0.62 \[-0.67, -0.56]     | -0.70         | \[-0.75, -0.66]  |
|                  | qwen2.5: 7b           | -0.63 \[-0.68, -0.57]     | -0.72         | \[-0.76, -0.67]  |
| **RAGAS**        | gpt-4o-mini           | **-0.87 \[-0.90, -0.85]** | -0.95         | \[-0.97, -0.93]  |
| **LLM + NLI**    | gemma3: 12b           | -0.82 \[-0.85, -0.79]     | **-0.96**     | \[-0.98, -0.94]  |
|                  | llama3.3: 70b         | -0.83 \[-0.86, -0.80]     | -0.94         | \[-0.96, -0.92]  |

Key takeaways:

* Pipeline methods (RAGAS and LLM + NLI) outperform standalone LLM-as-judge models, showing stronger negative
  correlations that indicate better detection of factual errors.
* The RAGAS pipeline with GPT-4o-mini achieves the highest Pearson correlation (-0.87) and near-perfect Kendall’s tau (
  -0.95), reflecting both linear and rank-order accuracy.
* The LLM + NLI (i.e., `truthscore`) approach offers a strong open-weight alternative with competitive performance,
  enabling efficient and cost-effective factuality evaluation.
* Standalone LLM-as-judge methods exhibit weaker correlations (Pearson between -0.62 and -0.74), suggesting lower
  reliability in capturing factual degradation.

These results demonstrate how TruthBench’s perturbed datasets enable effective benchmarking and comparison of factuality
evaluation pipelines, promoting development of accurate and scalable factual robustness assessment algorithms.

# Datasets

We are also open-sourcing the [datasets](./datasets) we used to access the quality of our pipeline. In short, we asked
annotators to compare the perceived quality of answers between experts and pipeline generated. The annotators needed to
decide which option aligned best to a specific set of guidelines (which can be found at our paper; see Appendix C). The
annotators have the alternative of accepting both options (if they had perceived similar quality) or rejecting them
both (if they both didn't comply with guidelines).

```text
├── datasets
│   ├── evaluation                         # datasets used for evaluating LLMs and other techniques (Section 5) 
│   │   ├── dataset.json                   # the pipeline generated dataset (with A0 -> A4)
│   │   ├── factual_correctness_eval.jsonl # evaluation for fast-fc (our cost efficient implementation and ragas (default)
│   │   ├── gold-dataset.json              # set of Question and ground truths sampled from Google's Natural Questions dataset
│   │   ├── llm_as_judge_eval.jsonl        # evaluation of several LLMs for factual correctness
│   │   ├── report.json                    # detailed report of the question transformations 
│   ├── human-assessment                   # datasets used for validating the quality of the pipeline (Section 4)
│   │   ├── assessment-dataset.json        # set of Q&As manually fabricated by experts (including A0 -> A4) with alternative versions produced by our pipeline
│   │   ├── report.json                    # the pipeline report with details about the incremental changes when producing the "ai" responses in assessment-dataset.json
│   │   ├── results-evaluator-1.json       # assessment from evaluator 1 (preferences)
│   │   ├── results-evaluator-2.json       # assessment from evaluator 2 (preferences) 
```

# UI

We provide a user-friendly webapp to facilitate comparing A0-A4 responses generated by different sources. The UI
provides a side-by-side visualization with diff capabilities. This tool was used by annotators to produce evaluate the
quality of our pipeline.



https://github.com/user-attachments/assets/fc98e124-1996-4592-bfa9-150b6d0e7942



One must provide an input dataset with the following schema:

```jsonc
{
  "questions": [
    {
      "id": 0,
      "question": "What are the main causes of climate change?",
      "ground_truth": "Climate change is primarily ...",
      "answers": {
        "A0": {
          "ai": "Human activities are the main drivers of climate change...",
          "human": "The primary driver of climate change is human activity..."
        },
        "A1": {
          "ai": "...",
          "human": "..."
        },
        "A2": {
          "ai": "...",
          "human": "..."
        },
        "A3": {
          "ai": "...",
          "human": "..."
        },
        "A4": {
          "ai": "...",
          "human": "..."
        }
      }
    },
    // ...
  ]
}
```

After evaluation, the results are exported with the following format (dictionary keys are the ids from the previous
file).

```jsonc
{
  "0": {
    "A0": "Both are bad",
    "A1": "AI",
    "A2": "Both are good",
    "A3": "AI",
    "A4": "Expert"
  },
  "1": {
    "A0": "...",
    "A1": "...",
    "A2": "...",
    "A3": "...",
    "A4": "..."
  },
  // ...
}
```

To launch the tool, first install the dependencies with `pip install -r ui/requirements.txt`. Then, you can run

```bash
python ui/evaluation_interface.py ./datasets/human-assessment/assessment-dataset.json ./datasets/results-evaluator-x.json
```

The application will start at [http://127.0.0.1:7860](http://127.0.0.1:7860) which can be access with your browser.

# Cite this work
