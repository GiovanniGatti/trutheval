---
layout: page
title: Can We Trust the Judges? Validation of Factuality Evaluation Methods via Answer Perturbation
---

Everyone is racing to measure the truthfulness of LLMs — but how trustworthy are those measurements?

As large language models become tools for research, advice, and decision-making, evaluating their factual accuracy is
crucial. And yet, many of the metrics we rely on, from generative judges to retrieval-based scores, have never been
rigorously tested themselves. We’re using them to benchmark LLMs, fine-tune systems, and publish papers, often without
knowing whether they can really spot factual errors.

In this project, we flip the script: Instead of evaluating LLMs, we evaluate the evaluators.
We introduce a factual perturbation pipeline: A method that gradually injects controlled errors (like wrong dates,
swapped names, or subtle distortions) into correct answers. Then, we test whether popular factuality metrics respond as
they should.

Some do. Others don’t.

This blog post walks you through the pipeline, shares what we learned from benchmarking top metrics, and introduces the
tools we built to help others stress-test factuality evaluations — including our new framework, TruthEval.

## TruthBench

![_config.yml]({{ site.baseurl }}/images/pipeline.svg){:style="width: 675px; display:block; margin-left: auto;
margin-right: auto;"}

To evaluate how well factuality metrics can detect subtle and severe errors, we design a multi-step Factual Perturbation
Pipeline that generates answer variants with controlled factual degradation.

Starting from a ground-truth answer, the pipeline creates five alternative responses (A0–A4). A0 is a faithful
paraphrase, while A1 to A4 introduce increasingly serious factual mistakes. These variants are crafted to remain fluent
and plausible—challenging even for advanced LLM-based evaluators.

Our pipeline follows a structured process:

* It paraphrases the reference while preserving correctness (A0).
* It extracts and ranks key factual components using syntactic parsing and LLM reasoning.
* It filters out overlaps with the question, ensuring perturbations target information not obvious from the query.
* It groups facts by importance and perturbation level, balancing severity across A1–A4.

Finally, it prompts an LLM to rewrite facts in plausible but incorrect ways, producing subtle to extreme factual
deviations.

This framework is useful because it enables fine-grained, realistic stress-testing of factuality metrics. Rather than
using synthetic or arbitrary errors, our method introduces factual mistakes in a way that mimics real-world LLM failure
modes. This makes it an ideal testbed for evaluating robustness in factual evaluation.

## Validation

To ensure our pipeline generates factually perturbed answers as intended, we ran a blind test comparing its output to
that of human experts. Here's how it worked:

1. **Setup:** For 20 Q&A pairs, both a human expert and our AI pipeline created five versions of the answer (from
   completely correct to increasingly incorrect—A0 to A4).
2. **Blind Evaluation:** Two independent evaluators were shown the AI and human versions side-by-side—without knowing
   which was which—and asked to pick which better fit the intended level of correctness (or say both/neither worked).

👀 Results:

* In 85% of cases, evaluators found no meaningful difference between AI and human responses.
* For the most incorrect answers (A1–A4), the AI was rated as non-inferior to human experts, even under conservative
  statistical assumptions.
* Direct disagreement between AI and expert preferences was extremely rare (just 2.5% of cases).

While inter-rater agreement was modest—highlighting how hard this task is even for humans—the overall takeaway is clear:
our AI performs on par with experts when generating nuanced factual perturbations.

![_config.yml]({{ site.baseurl }}/images/by_expert_preference.svg){:style="height: 265px; vertical-align: middle;"}
![_config.yml]({{ site.baseurl }}/images/by_level.svg){:style="height: 270px; vertical-align: middle;"}

## TruthScore

To automatically assess how factually consistent an answer is with its source passage, we use TruthScore—a fast, modular
alternative to RAGAS’s FactualCorrectness metric.

How it works:

1. It breaks down an answer into atomic factual claims using a lightweight LLM.
2. Each claim is verified via Natural Language Inference (NLI) against the reference passage.
3. The final score reflects how many claims are actually entailed by the source (0 = none, 1 = all).

TruthScore is open-weight–friendly (supports models like Mistral and Gemma), runs efficiently on GPU, and scales
well—making it perfect for our evaluation needs.

In short: it helps us quantify how "true" a generated answer is, with minimal overhead.

## Evaluation of LLM Assessment Methods

To understand how well different techniques capture factuality, we ran a systematic evaluation using our noising
pipeline. The idea is simple: if you start with a high-quality answer and gradually inject factual errors, a good
evaluator should assign lower and lower factuality scores. But do they?

We applied this logic to compare three major approaches:

* **LLM-as-judge:** Ask the LLM directly to rate factuality.
* **RAGAS:** Use the structured factual correctness pipeline (defaulting to GPT-4o-mini).
* **LLM + NLI (TruthScore):** Our open-weight alternative using claim decomposition and a Natural Language Inference
  model.

We took 100 Q&A pairs from the Google Natural Questions dataset, and for each, created five versions — from completely
faithful (A0) to highly perturbed (A4) where 80% of facts were changed. This resulted in 500 examples, each with
increasing levels of factual degradation.

To evaluate performance, we correlated the true perturbation level with the factuality score produced by each method,
using:

* Pearson correlation to assess linearity,
* Kendall’s tau to check if models get the relative order right.

Here’s what we found:

![_config.yml]({{ site.baseurl }}/images/score_vs_noise.svg){:style="height: 220px; vertical-align: middle;"}
![_config.yml]({{ site.baseurl }}/images/pearson_vs_llm_size.svg){:style="height: 220px; vertical-align: middle;"}

| Method        | LLM                   | Pearson (95% CI)          | Kendall Tau | Kendall (95% CI) |
|---------------|-----------------------|---------------------------|-------------|------------------|
| LLM-as-judge  | gemma3: 4b            | -0.63 \[-0.69, -0.58]     | -0.79       | \[-0.82, -0.77]  |
|               | llama3.3: 70b         | -0.74 \[-0.78, -0.70]     | -0.86       | \[-0.88, -0.84]  |
|               | mistral-small3.1: 24b | -0.71 \[-0.75, -0.66]     | -0.76       | \[-0.79, -0.72]  |
|               | phi4: 14b             | -0.74 \[-0.78, -0.70]     | -0.81       | \[-0.83, -0.78]  |
|               | prometheus-v2: 7b     | -0.62 \[-0.67, -0.56]     | -0.70       | \[-0.75, -0.66]  |
|               | qwen2.5: 7b           | -0.63 \[-0.68, -0.57]     | -0.72       | \[-0.76, -0.67]  |
| **RAGAS**     | gpt-4o-mini           | **-0.87** \[-0.90, -0.85] | -0.95       | \[-0.97, -0.93]  |
| **LLM + NLI** | gemma3: 12b           | -0.82 \[-0.85, -0.79]     | **-0.96**   | \[-0.98, -0.94]  |
|               | llama3.3: 70b         | -0.83 \[-0.86, -0.80]     | -0.94       | \[-0.96, -0.92]  |

> 🔍 TL;DR: LLM-as-judge struggles. Pipelines shine—especially when combining open-weight models with NLI.

1. **LLM-as-judge is inconsistent**:
   Despite their size and training, judge-based LLMs (e.g., Prometheus, Mistral) fail to align strongly with the
   intended perturbation levels. Their Pearson and Kendall values are significantly lower, and they often produce noisy
   or flat score distributions. As shown in the figure below, Prometheus-v2 barely reacts to increasing noise levels—bad
   news for fine-grained factuality checks. Prometheus-v2 (blue) is almost flat, while GPT-4o-mini (red) tracks the drop
   perfectly. The open-weight LLM + NLI variant follows closely, too.

2. **Factual correctness pipelines are far more reliable**:
   Both the GPT-powered RAGAS pipeline and the open-weight LLM + NLI setup show strong, consistent drops in score as
   perturbations increase. Kendall’s tau values of 0.94–0.96 mean they nearly always get the answer ranking right. This
   makes them trustworthy tools for evaluating factuality—especially when factual degradation is subtle or cumulative.

3. **TruthScore (LLM + NLI) performs competitively—and affordably**:
   TruthScore, our open-weight implementation, doesn’t just hold its own against GPT-4o—it almost matches or even beats
   it in ranking accuracy. It uses much smaller models, runs on consumer GPUs, and avoids vendor lock-in—all without
   compromising factuality detection.

4. **Size isn’t everything**:
   Surprisingly, larger models don’t always do better. Figure below shows that some mid-sized models outperform their
   larger cousins. Bigger doesn’t mean better — smart pipelines win. Even smaller models in a well-designed pipeline can
   outperform massive judges. Factuality evaluation isn’t just about
   scale—it’s about structure.

Whether you're building a factual QA system, benchmarking your RAG pipeline, or just trying to trust your LLM outputs,
you need a reliable way to detect factual errors. This study shows that factual correctness pipelines—especially with
open models—are not only more aligned with truth degradation, but also more scalable and interpretable than LLM-as-judge
methods.

In short: don’t just ask the LLM to grade itself. Equip it with tools to reason about truth.

## Known Limitations

While our method provides a structured way to test factuality evaluation, it has a few important limitations.

1. **Inconsistent Perturbation Strength**: Not all answers respond the same way to our edits. Short answers may be
   highly sensitive to minor changes, while longer, more detailed ones might require stronger perturbations to alter
   meaning. This means the degree of factual noise isn’t uniform across all examples.

2. **Meaning May Be Preserved**:    Some changes affect surface wording without altering the core meaning. This can make
   it harder to evaluate how well models detect factual errors, especially when the fact is embedded within longer
   responses.

3. **Risk of Contradictions**:   Some perturbations may unintentionally introduce contradictions or inconsistencies,
   which can confuse both human and model evaluations. Our current guidelines don’t fully resolve how such cases should
   be handled.

4. **Language Limitations**: Although the method is language-agnostic in principle, it relies on high-quality
   language tools. In low-resource languages or those with complex grammar, the quality of perturbations may drop.

5. **Prompting Strategies Not Fully Explored**: We have not yet tested more advanced prompting techniques like
   chain-of-thought or instruction tags, which could improve factuality assessments in future work.

## Using TruthEval

You can use TruthEval to both generate noised examples and evaluate model responses using factuality metrics. It’s split
into two modules: TruthBench and TruthScore.

### TruthBench

Use TruthBench to create perturbed versions of your dataset for evaluation. It works via CLI and supports OpenAI and
local LLMs.

```bash
pip install truthbench[openai]
python -m spacy download en_core_web_sm
truthbench --input-file data.json --output-dir results/
```

[See full TruthBench docs →](https://github.com/GiovanniGatti/trutheval/tree/main/truthbench#readme)

### TruthScore

Use TruthScore to score the factual consistency of responses. It supports OpenAI, local models via Ollama, and optional
custom NLI models for deeper semantic checks.

```
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from truthscore import OpenFactualCorrectness

evaluator_llm = LangchainLLMWrapper(OpenAI())
metric = OpenFactualCorrectness(llm=evaluator_llm)

score = metric.single_turn_score(SingleTurnSample(**your_data))
```

[See full TruthScore docs →](https://github.com/GiovanniGatti/trutheval/tree/main/truthscore#readme)

## Conclusions

TruthEval shows that pipeline-based approaches like RAGAS outperform judge-style LLM methods when it comes to detecting
factual errors—especially as responses degrade. Interestingly, bigger models aren’t always better: mid-sized open-source
LLMs combined with NLI can deliver strong results with lower cost and compute.

More than a benchmark, TruthEval is a practical toolkit. It lets you stress-test factuality metrics, compare them under
controlled perturbations, and choose what works best for your task. We hope it helps raise the bar for evaluating
truthfulness in NLP — and supports building more trustworthy educational tools.

[👉 Get started on GitHub](https://github.com/GiovanniGatti/trutheval/)