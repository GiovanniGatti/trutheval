import json
import pathlib
from typing import List, Dict, Any

from truthbench.pipeline import Reader


class JsonReader(Reader):
    """
    A reader that loads question-answer samples from a JSON file.

    The JSON file must contain a list of dictionaries. Each dictionary must include
    the following keys:
        - "question" (str)
        - "ground_truth" (str)

    Example input JSON:
        [
            {"question": "What is Python?", "ground_truth": "A programming language."},
            {"question": "What is 2+2?", "ground_truth": "4"}
        ]

    Parameters:
        input_file (pathlib.Path): Path to the input JSON file.

    Raises:
        ValueError: If the JSON is invalid, not a list of objects, or missing required keys.
    """

    def __init__(self, input_file: pathlib.Path):
        self._input_file = input_file

    def samples(self) -> List[Dict[str, Any]]:
        with open(self._input_file, "r") as f:
            content = f.read()

        gold_dataset = json.loads(content)

        if not isinstance(gold_dataset, list):
            raise ValueError("Expected top-level JSON array (list of samples)")

        samples = []
        for d in gold_dataset:
            if not isinstance(d, dict):
                raise ValueError(f"Samples must be JSON objects")
            if "question" not in d or "ground_truth" not in d:
                raise ValueError(
                    f"Missing required keys: 'question' and 'ground_truth'"
                )
            samples.append({"question": d["question"], "ground_truth": d["ground_truth"]})

        return samples
