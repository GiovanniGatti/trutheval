# TruthBench

`truthbench` is a modular pipeline designed to generate controlled factual perturbations of ground-truth answers. These
perturbations enable fine-grained meta-evaluation of factuality metrics used to assess large language model (LLM)
outputs.

While many tools exist to judge whether LLM-generated answers are "factual," their own sensitivity, reliability, and
robustness remain underexplored. `truthbench` provides a way to systematically test these tools using corrupted versions
of correct answers, ranging from semantically faithful paraphrases to subtly or severely inaccurate alternatives.

# Key Features

* 🧠 LLM-based Paraphrasing and Corruption: Produces answer variants (A0–A4) that span a factuality spectrum.
* 🏗️ Step-by-Step Pipeline Architecture: Modular components for paraphrasing, information extraction, perturbation, and
  grouping.
* 🎯 Controlled Evaluation Levels: Supports reproducible degradation of factual content while preserving fluency and
  answer
  structure.
* 🔍 Built for Evaluating Evaluators: Enables validation of popular factuality metrics like RAGAS, FactScore, and
  LLM-as-judge models.

# Use Cases

* Meta-evaluating factuality metrics in open-ended QA settings.
* Building datasets with graded factual errors.
* Benchmarking the sensitivity of evaluation tools to fine-grained truth degradation.

# How It Works

The pipeline takes a question and ground-truth answer, and produces 5 graded answers:

| Answer | Description                               |
|--------|-------------------------------------------|
| `A0`   | Faithful paraphrase of the ground truth   |
| `A1`   | Mild factual perturbation                 |
| `A2`   | Moderate factual error                    |
| `A3`   | High factual degradation                  |
| `A4`   | Severely incorrect or misleading response |

Internally, the pipeline follows these stages:

1. Paraphrase Ground Truth (A0)
2. Extract Key Factual Components
3. Filter Overlap with Question
4. Rank Factual Importance
5. Group by Perturbation Level
6. Generate Perturbed Answers (A1–A4)

Each step is implemented as a modular Step class, enabling customization and extension.

# Example

<details>
<summary><strong>Example</strong>: <em>Who did the United States win its independence from?</em></summary>

**A0 (Reference)**  
Independence Day, commonly known as the Fourth of July or July Fourth, is **a federal holiday** in the United States
celebrating **the adoption of the Declaration of Independence** **on July 4, 1776**. **On this day**, the Continental
Congress announced that the thirteen American colonies considered themselves a new nation, called **the United States of
America**, and were no longer under **British rule**. Interestingly, the Congress had voted to declare independence *
*two days** earlier, **on July 2**.

**A1 (Low perturbation)**  
... celebrating **the adoption of the Declaration of Independence** ~~on July 4, 1776~~ **on August 5, 1776** ...

**A2 (Medium perturbation)**  
... celebrating the Declaration of Independence **on August 5, 1781**. ~~On this day~~ **On that moment**, ...

**A3 (High perturbation)**  
... is **an unofficial event** ... celebrating **a proposal of the Declaration of Independence** **on August 5, 1781
** ...

**A4 (Extreme perturbation)**  
... celebrating **a proposal of the drafting of Independence** **on August 5, 1781** ... called **the United States of
the Colonies**, and were no longer under **Spanish rule**.

</details>

# Using the perturbation pipeline

## CLI Usage

You can run the TruthBench pipeline directly from the command line.

### Installation

Install the package with optional OpenAI dependencies:

```bash
pip install truthbench[openai]
```

### Download required spaCy model

TruthBench relies on the spaCy English model. Download it once with:

```bash
python -m spacy download en_core_web_sm
```

### Set your OpenAI API key

Export your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Run the pipeline

```bash
truthbench --input-file path/to/input.json --output-dir path/to/output_dir
```

This will create `report.json` and `dataset.json` inside `output_dir`.

### Output File Formats

After running the pipeline, two main output files are generated in the output directory:

#### 1. `dataset.json`

This file contains the input questions along with multiple generated answer variants.

- **Structure:**

```jsonc
  {
    "questions": [
      {
        "id": 0,                                                   // Unique identifier for the question
        "question": "why is the sky blue?",                        // The original question text.
        "ground_truth": "The sky appears to be blue because...",   // The correct answer text.
        "answers": {                                               // A dictionary of answer variants with increased perturbation levels
          "A0": "The sky looks blue because...",
          "A1": "...",
          "A2": "...",
          "A3": "...",
          "A4": "..."
        }
      },
      // ...
    ]
  }
```

#### 2. `report.json`

This file contains all the processing details.

```jsonc
{
  "report": {                                           // Summary metrics about the evaluation (counts of samples, errors, etc.)
    "input_samples": 100,
    "find_factual_data_error": 0,
    "json_parse_ranking_error": 3,
    "index_ranking_error": 52,
    "ranking_factual_data_error": 2,
    "output_samples": 100
  },
  "questions": [                                       // The complete processing trace for every dataset sample
    {
      "question": "what do the 3 dots mean in math?",
      "ground_truth": "In logical argument...",
      "raw_factual_data": [
        "logical reasoning",
        "...",
      ],
      "with_brackets": {
        "A0": "In [logical reasoning] and [mathematics] ..."
        // ...
      },
      // ...
    },
    // ...
  ]
}
```

## Creating a Custom Reader, Step, and Using an Open-Source LLM in the Pipeline

You can customize the pipeline to your needs. You may combine your custom implementations with available code or
override any blocks.

The `Pipeline` runs on three abstractions:

* `Reader`: fetches data;
* `Step`: provides the processing logic;
* `Pipeline`: holds a sequence of steps and execute them.

You can declare a pipeline by chaining a sequence of `Step`s and run it like this...

```python
from truthbench import Pipeline
from truthbench.steps.counter import CounterStep
from truthbench.steps.paraphrase import ParaphraseStep

llm = ...
reader = ...

p = (
    Pipeline()
    .with_step(ParaphraseStep(llm))
    .with_step(CounterStep(expected_levels=5))
)

samples, tracker = p.run(reader)
```

The `samples` contain the list with the processing traces for each sample, while `tracker` has general stats about the
processing.

Adding a custom step requires you to implement a `Step` abstract class.

```python
from typing import Dict, Any
from truthbench import Step


class WordCountStep(Step):
    def __init__(self):
        super().__init__(required_fields={"paraphrased_question"}, counters=frozenset({"word_counted"}))

    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        question = sample["paraphrased_question"]
        sample["word_count"] = len(question.split())
        tracker["word_counted"] += 1
```

Each step may have a dependency on previous processing. In the above example, it requires that a previous step has
computed `paraphrased_question`. If that's not the case, you likely have a dependency issue or a bug worth
investigating. A step can also declare a set of `counters` it needs to keep track of stats. In the above example, it
declares it may increment `word_counted`.

The following steps are available:

| **Step Name**                                                                     | **Description**                                                                                 | **Updated Counters**                                                                                   | **Required Fields**                        |
|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [`ParaphraseStep`](truthbench/src/truthbench/steps/paraphrase.py)                 | Generates a faithful paraphrase of the ground-truth answer using the LLM.                       | *(none)*                                                                                               | `ground_truth`                             |
| [`FactualDataStep`](truthbench/src/truthbench/steps/factual.py)                   | Identifies factual spans in a sentence using spaCy and brackets them.                           | `find_factual_data_error`                                                                              | `answers`                                  |
| [`BlacklistItemsFromQuestionStep`](truthbench/src/truthbench/steps/blacklist.py)) | Removes factual items from `raw_factual_data` if they appear in the question (minus stopwords). | *(none)*                                                                                               | `question`, `raw_factual_data`             |
| [`RankFactualDataStep`](truthbench/src/truthbench/steps/rank.py)                  | Uses an LLM to assign an importance ranking to factual terms based on a bracketed sentence.     | `ranked_factual_data`, `index_ranking_error`, `ranking_factual_data_error`, `json_parse_ranking_error` | `with_brackets`, `raw_factual_data`        |
| [`FilterFactualDataStep`](truthbench/src/truthbench/steps/filter.py)              | Keeps top-ranked factual items and removes those blacklisted (present in the question).         | *(none)*                                                                                               | `ranked_factual_data`, `blacklisted`       |
| [`CreateNoiseExamplesStep`](truthbench/src/truthbench/steps/noise.py)             | Generates noisy paraphrases with varying levels of factual degradation using factual spans.     | *(none)*                                                                                               | `factual_data`, `with_brackets`, `answers` |
| [`CounterStep`](truthbench/src/truthbench/steps/counter.py)                       | Verifies if the expected number of answer levels are present and increments a counter.          | `output_samples`                                                                                       | `answers`                                  |

A pipeline also needs a datasource to fetch data. You can declare your own data fetching mechanism by subclassing
a `Reader`.

```python
from typing import List, Dict, Any
from truthbench import Reader


class StaticReader(Reader):
    def samples(self) -> List[Dict[str, Any]]:
        return [
            {
                "question": "why is the sky blue?",
                "ground_truth": "The sky appears blue because of Rayleigh scattering..."
            }
        ]
```

Generally, Readers expect to output at least two fields: `question` and `ground_truth`.

Right now, we made available a [`JsonReader`](truthbench/src/truthbench/readers/json_reader.py) that expects a `json`
file with the following structure:

```jsonc
[
    {
        "question": "who is playing the halftime show at super bowl 2016?",
        "ground_truth": "The Super Bowl 50 Halftime Show took place on..."
    },
    // ...
]
```

Lastly, some steps may need access to a running large language model (LLM). We provide support to OpenAI's ChatGPT with
`[GPT](truthbench/src/truthbench/llms/openai.py)` (it requires installing `pip install truthbench[openai]`), but you can
implement your own LLM access by subclassing:

```python
from typing import List, Dict
from truthbench import LLM


class OpenSourceLLM(LLM):
    def __init__(self, model):
        self.model = model  # e.g., from HuggingFace or llama-cpp

    def query(self, messages: List[Dict[str, str]]) -> str:
        prompt = ...  # Convert messages if needed
        response = self.model.generate(prompt)  # Use the appropriate method
        return response
```

# Pipeline validation

To ensure the quality of the factual perturbations, we conducted a human evaluation comparing outputs from the
truthbench pipeline with those created by experts.

Two evaluators were shown factual Q&A pairs with five answer variants (A0–A4) and asked to blindly choose which
version (AI- or expert-generated) better fit the intended level of factuality — or indicate a tie.

Key results:

* 🟰 82.5% of evaluations resulted in ties, indicating that AI and human answers were often perceptually
  indistinguishable.
* ✅ The AI pipeline was statistically non-inferior to human performance.
* ❗ Only 2.5% of examples showed conflicting preferences between evaluators.

# Known limitations

Our perturbation pipeline systematically applies linguistic and semantic modifications using dependency parsers and
predefined operators. However, the effectiveness of these perturbations can vary depending on the properties of the
target text:

* 🧩 **Variation in sensitivity:** Verbose or highly detailed answers (e.g., those generated by large language models)
  may
  require more targeted or intensive perturbations to induce meaningful semantic changes. In contrast, shorter, more
  concise answers tend to be more sensitive to even minor modifications. Consequently, the uniformity of perturbation
  strength across different questions and answers is not guaranteed.

* 🛡️ **Core content preservation:** Some perturbations might alter surface-level phrasing without affecting the core
  factual content. For example, for the question “Who breaks a tie in the US Senate?,” `truthbench` will fail to modify
  “the Vice President” in “The Vice President serves as the ex officio President of the Senate but is only permitted to
  vote to resolve a tie.” Although we currently lack quantitative evidence on how widespread these cases are, this
  limitation is especially relevant for verbose answers where the main fact constitutes only a small fraction of the
  text. Our evaluators were not specifically instructed on handling these borderline cases, indicating a need for
  further analysis and possibly alternative perturbation strategies.

* ⚠️ **Semantic inconsistencies:** Certain perturbations may introduce contradictions or inconsistencies. For instance,
  for the question “Who wrote the text for Jeanie with the Light Brown Hair?,”  `truthbench` can produce “Jeanie with
  the Light Brown Hair is a folk song created by Henry Bishop \[...]. **Foster** composed the song thinking of \[...].
  Such examples fail our semantic guidelines and should be marked as rejected — either accepting a valid human
  alternative or rejecting both answers.

* 🌐 **Language dependency:** Although the approach is designed to be language-agnostic in principle, it relies heavily
  on
  the availability and quality of dependency parsers and language models for the target language. Languages with complex
  morphology or syntax, or those that are low-resource, may experience reduced perturbation accuracy and coverage.

