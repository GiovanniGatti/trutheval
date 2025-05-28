import abc
from typing import List, Dict, Tuple, Any, Set

from tqdm import tqdm


class StrictTracker(dict):
    def __init__(self, allowed_keys: set[str]):
        super().__init__()
        self._allowed_keys = allowed_keys
        self.update({k: 0 for k in self._allowed_keys})

    def __getitem__(self, key):
        if key not in self._allowed_keys:
            raise KeyError(
                f"Tracker counter '{key}' is being set but was not declared. "
                f"Declare it at the Step constructor with: "
                f"    super().__init__(..., counters=frozenset({{{repr(key)}}}))"
            )
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key not in self._allowed_keys:
            raise KeyError(
                f"Tracker counter '{key}' is being set but was not declared. "
                f"Declare it at the Step constructor with: "
                f"    super().__init__(..., counters=frozenset({{{repr(key)}}}))"
            )
        return super().__setitem__(key, value)


class LLM(abc.ABC):

    @abc.abstractmethod
    def query(self, messages: List[Dict[str, str]]) -> str:
        ...


class Step(abc.ABC):

    def __init__(self, required_fields: Set[str] = frozenset(), counters: Set[str] = frozenset()):
        self.required_fields = required_fields
        self.counters = counters

    def validate(self, sample: Dict[str, Any]) -> None:
        current_fields = set(sample.keys())
        if not self.required_fields.issubset(current_fields):
            missing = self.required_fields.difference(current_fields)
            raise ValueError(
                f"{type(self).__name__} requires {self.required_fields}, but some are missing from the sample: "
                f"{missing}. Check pipeline dependencies before proceeding."
            )

    @abc.abstractmethod
    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        ...


class Reader(abc.ABC):
    @abc.abstractmethod
    def samples(self) -> List[Dict[str, Any]]:
        ...


class Pipeline:

    def __init__(self, with_progress: bool = True):
        self._steps: List[Step] = []
        self._with_progress = with_progress

    def with_step(self, step: Step) -> 'Pipeline':
        self._steps.append(step)
        return self

    def run(self, reader: Reader) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        allowed_keys = {"input_samples"} | frozenset.union(*(step.counters for step in self._steps))

        tracker = StrictTracker(allowed_keys)

        samples = reader.samples()

        collected = []
        for sample in tqdm(samples, desc="Samples:", disable=not self._with_progress):
            tracker["input_samples"] += 1
            for step in self._steps:
                step.validate(sample)
                step.step(sample, tracker)
            collected.append(sample)

        return collected, tracker
