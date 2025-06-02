from typing import Optional, List, Dict

import pydantic


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
