import json
import pathlib
from typing import List, Dict, Any

from truthbench.pipeline import Reader


class JsonReader(Reader):

    def __init__(self, input_file: pathlib.Path):
        self._input_file = input_file

    def samples(self) -> List[Dict[str, Any]]:
        with open(self._input_file, "r") as f:
            content = f.read()

        gold_dataset = json.loads(content)
        return [
            {"question": d["question"], "ground_truth": d["ground_truth"]} for d in gold_dataset
        ]
