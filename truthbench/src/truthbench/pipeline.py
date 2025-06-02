import abc
from typing import List, Dict, Tuple, Any, Set

from tqdm import tqdm


class StrictTracker(dict):
    """
    A dictionary subclass that enforces allowed keys and initializes counters.

    This tracker only allows keys declared in `allowed_keys`. Accessing or setting
    a key not in `allowed_keys` raises a KeyError with a helpful message.
    """

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
    """
    Abstract base class for Language Models.

    Subclasses must implement the `query` method to send messages to the LLM
    and return the response as a string.
    """

    @abc.abstractmethod
    def query(self, messages: List[Dict[str, str]]) -> str:
        """
        Query the language model with a list of messages and get the output string.

        Args:
            messages (List[Dict[str, str]]): A list of message dicts with keys like 'role' and 'content'.

        Returns:
            str: The LLM's response as a string.
        """
        ...


class Step(abc.ABC):
    """
    Abstract base class representing a single processing step in the pipeline.

    Args:
        required_fields (Set[str]): Set of keys that must be present in each sample before running this step.
        counters (Set[str]): Set of counter names that this step may increment in the tracker.
    """

    def __init__(self, required_fields: Set[str] = frozenset(), counters: Set[str] = frozenset()):
        self.required_fields = required_fields
        self.counters = counters

    def validate(self, sample: Dict[str, Any]) -> None:
        """
        Validate that the sample contains all required fields for this step.

        Raises:
            ValueError: If any required field is missing in the sample.
        """

        current_fields = set(sample.keys())
        if not self.required_fields.issubset(current_fields):
            missing = self.required_fields.difference(current_fields)
            raise ValueError(
                f"{type(self).__name__} requires {sorted(self.required_fields)}, but some are missing from the sample: "
                f"{sorted(missing)}. Check pipeline dependencies before proceeding."
            )

    @abc.abstractmethod
    def step(self, sample: Dict[str, Any], tracker: Dict[str, int]) -> None:
        """
        Execute the step logic on the sample, possibly updating the tracker.

        Args:
            sample (Dict[str, Any]): The data sample to process.
            tracker (Dict[str, int]): A dictionary tracking counters/errors during processing.
        """
        ...


class Reader(abc.ABC):
    """
    Abstract base class for data readers that provide samples to the pipeline.

    Subclasses must implement the `samples` method that returns a list of validated samples.
    """

    @abc.abstractmethod
    def samples(self) -> List[Dict[str, Any]]:
        """
        Load and return validated samples.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing 'question' and 'ground_truth' keys.

        Raises:
            ValueError: If the source could not be read or has an invalid format.
        """
        ...


class Pipeline:
    """
    Orchestrates a sequence of Steps to process data samples.

    Args:
        with_progress (bool): Whether to display a progress bar during execution (tqdm).
    """

    def __init__(self, with_progress: bool = True):
        self._steps: List[Step] = []
        self._with_progress = with_progress

    def with_step(self, step: Step) -> 'Pipeline':
        """
        Add a processing step to the pipeline.

        Args:
            step (Step): A step instance to add.

        Returns:
            Pipeline: Self, to allow method chaining.
        """
        self._steps.append(step)
        return self

    def run(self, reader: Reader) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Execute all steps in sequence on each sample provided by the reader.

        Args:
            reader (Reader): Data reader yielding samples.

        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, int]]:
                - List of processed samples.
                - Tracker dictionary with counters collected during processing.
        """
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
