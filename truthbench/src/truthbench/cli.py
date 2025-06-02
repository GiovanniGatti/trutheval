import argparse
import pathlib

import truthbench
from truthbench.models import Report, Tracker, Sample
from truthbench.readers.json_reader import JsonReader


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
