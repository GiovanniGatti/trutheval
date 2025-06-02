import pathlib
import unittest

from pydantic import ValidationError

from truthbench.models import Tracker, Report, Dataset


def test_report_to_dataset():
    from truthbench.models import Report, Sample
    samples = [
        Sample(question="q1?", ground_truth="gt1", answers={"A0": "a"}),
        Sample(question="q2?", ground_truth="gt2", answers={"A0": "a", "A1": "b"})
    ]
    report = Report(report=Tracker(), questions=samples)
    dataset = report.to_dataset()
    assert len(dataset.questions) == 1


def test_report_backward_compatibility():
    REPORT_PATH = pathlib.Path(__file__).parent.parent.parent / "datasets" / "evaluation" / "report.json"

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = f.read()

    # Basic validation: no error means it passed Pydantic validation
    try:
        Report.model_validate_json(data)
    except ValidationError as e:
        assert False, f"The current Report schema is not compatible with old datasets. Error {e}"


def test_dataset_backward_compatibility():
    DATASET_PATH = pathlib.Path(__file__).parent.parent.parent / "datasets" / "evaluation" / "dataset.json"

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = f.read()

    # Basic validation: no error means it passed Pydantic validation
    try:
        Dataset.model_validate_json(data)
    except ValidationError as e:
        assert False, f"The current Dataset schema is not compatible with old datasets. Error {e}"


if __name__ == "__main__":
    unittest.main()
