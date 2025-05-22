# TruthEval

# Using the perturbation pipeline

# Using the cost-efficient Factual Correctness metric

# [Datasets](./datasets)

We are also open-sourcing the datasets we used to access the quality of our pipeline. In short, we asked annotators to
compare the perceived quality of answers between experts and pipeline generated. The annotators needed to decide which
option aligned best to a specific set of guidelines (which can be found at our paper; see Appendix C). The annotators
have the alternative of accepting both options (if they had perceived similar quality) or rejecting them both (if they
both didn't comply with guidelines).

```text
├── datasets
│   ├── evaluation                         # datasets used for evaluating LLMs and other techniques (Section 5) 
│   │   ├── dataset.json                   # the pipeline generated dataset (with A0 -> A4)
│   │   ├── factual_correctness_eval.jsonl # evaluation for fast-fc (our cost efficient implementation and ragas (default)
│   │   ├── gold-dataset.json              # set of Question and groundtruths sampled from Google's Natural Questions dataset
│   │   ├── llm_as_judge_eval.jsonl        # evaluation of several LLMs for factual correctness
│   │   ├── report.json                    # detailed report of the question transformations 
│   ├── human-assessment                   # datasets used for validating the quality of the pipeline (Section 4)
│   │   ├── assessment-dataset.json        # set of Q&As manually fabricated by experts (including A0 -> A4) with alternative versions produced by our pipeline
│   │   ├── report.json                    # the pipeline report with details about the incremental changes when producing the "ai" responses in assessment-dataset.json
│   │   ├── results-evaluator-1.json       # assessment from evaluator 1 (preferences)
│   │   ├── results-evaluator-2.json       # assessment from evaluator 2 (preferences) 
```

# UI

# Prompts

# Cite this work
