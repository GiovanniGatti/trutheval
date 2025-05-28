<p align="center">
<img src="https://github.com/user-attachments/assets/8d30499c-ab58-4e92-8e11-089c01515c63" alt="drawing" style="height:340px"/>
</p>

# TruthEval

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

# Using the cost-efficient Factual Correctness metric

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
