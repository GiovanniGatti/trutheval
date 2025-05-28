import abc
from typing import List, Dict, Tuple, Any, Set

from tqdm import tqdm


class LLM(abc.ABC):

    @abc.abstractmethod
    def query(self, messages: List[Dict[str, str]]) -> str:
        ...


class Step(abc.ABC):

    def __init__(self, required_fields: Set[str] = frozenset()):
        self.required_fields = required_fields

    @abc.abstractmethod
    def step(self, sample: Dict[str, Any], tracker: Tracker) -> None:
        current_fields = set(sample.keys())
        if not self.required_fields.issubset(current_fields):
            missing = self.required_fields.difference(current_fields)
            raise ValueError(
                f"Step requires {self.required_fields}, but some are missing from the sample: {missing}. "
                f"Check pipeline dependencies before proceeding."
            )


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

    def run(self, reader: Reader) -> Tuple[List[Dict[str, Any]], Tracker]:
        tracker = Tracker()
        collected = []

        samples = reader.samples()

        for sample in tqdm(samples, desc="Samples:", disable=self._with_progress):
            tracker.input_samples += 1
            for step in self._steps:
                step.step(sample, tracker)
            collected.append(sample)
            if sample.is_valid():
                tracker.output_samples += 1

        return collected, tracker

    # def ensure_fields(self, sample: T) -> None:
    #     for step in self._steps:
    #         for f in step.required_fields:
    #             if not hasattr(sample, f):
    #                 raise RuntimeError(f"Sample missing required field {f} for step {type(step).__name__}")
    #             a = getattr(sample, f)
    #             if callable(a):
    #                 raise RuntimeError(
    #                     f"Expected required field {f} for step {type(step).__name__}, "
    #                     f"but found a method with the same name."
    #                 )
    #         for v in step.required_validators:
    #             if not hasattr(sample, v):
    #                 raise RuntimeError(f"Sample missing required validator {v} for step {type(step).__name__}")
    #             a = getattr(sample, v)
    #             if not callable(a):
    #                 raise RuntimeError(
    #                     f"Expected required validator {v} for step {type(step).__name__}, "
    #                     f"but found an attribute with the same name."
    #                 )

# TODO
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--output-dir", "-o", required=True, type=pathlib.Path)
#     parser.add_argument("--input-file", "-i", required=True, type=pathlib.Path)
#
#     args = parser.parse_args()
#
#     client = OpenAI()
#     llm = LLM(client)
#
#     samples, tracker = pipeline.run(JsonReader(args.input_file))
#
#     report = Report(report=tracker, questions=samples)
#     dataset = report.to_dataset()
#
#     with open(args.output_dir / "report.json", "w", encoding="utf-8") as f:
#         f.write(report.model_dump_json(indent=4))
#
#     with open(args.output_dir / "dataset.json", "w", encoding="utf-8") as f:
#         f.write(dataset.model_dump_json(indent=4))
#
#     print()
#     print("Stats")
#     print(tracker.model_dump_json(indent=4))
