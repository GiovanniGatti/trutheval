import argparse
import pathlib
from typing import List, Optional, Dict

import pydantic

import truthbench
from truthbench.readers.json_reader import JsonReader


class Tracker(pydantic.BaseModel):
    input_samples: int = 0
    find_factual_data_error: int = 0
    json_parse_ranking_error: int = 0
    index_ranking_error: int = 0
    ranking_factual_data_error: int = 0
    output_samples: int = 0


class Sample(pydantic.BaseModel):
    question: Optional[str] = None
    ground_truth: Optional[str] = None
    raw_factual_data: Optional[List[str]] = None
    with_brackets: Optional[Dict[str, str]] = None
    thinking: Optional[Dict[str, str]] = None
    blacklisted: Optional[List[str]] = None
    factual_data: Optional[List[str]] = None
    ranked_factual_data: Optional[List[str]] = None
    answers: Optional[Dict[str, str]] = None

    def is_valid(self) -> bool:
        return len(self.answers.keys()) > 1


class Item(pydantic.BaseModel):
    id: int
    question: str
    ground_truth: str
    answers: Dict[str, str]

    @classmethod
    def from_sample(cls, id_: int, sample: Sample) -> 'Item':
        return Item(id=id_, question=sample.question, ground_truth=sample.ground_truth, answers=sample.answers)


class Dataset(pydantic.BaseModel):
    questions: List[Item]


class Report(pydantic.BaseModel):
    report: Tracker
    questions: List[Sample]

    def to_dataset(self) -> Dataset:
        items = [Item.from_sample(id_=i, sample=s) for i, s in enumerate(self.questions) if s.is_valid()]
        return Dataset(questions=items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run truthbench pipeline")
    parser.add_argument(
        "--output-dir", "-o", required=True, type=pathlib.Path,
        help="Directory where to place the output dataset and the execution report"
    )
    parser.add_argument(
        "--input-file", "-i", required=True, type=pathlib.Path,
        help="Input json dataset containing questions and ground truths"
    )
    parser.add_argument(
        "--keep", "-k", default=.8, type=float,
        help="Percentage of factual data to preserve"
    )
    parser.add_argument(
        "--num-levels", "-l", default=5, type=int,
        help="Number of perturbation levels to produce A0-AX"
    )

    args = parser.parse_args()

    pipeline = truthbench.truth_pipeline(keep=args.keep, num_levels=args.num_levels)
    samples, tracker = pipeline.run(JsonReader(args.input_file))

    report = Report(report=Tracker(**tracker), questions=[Sample(**s) for s in samples])
    dataset = report.to_dataset()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / "report.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=4))

    with open(args.output_dir / "dataset.json", "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
