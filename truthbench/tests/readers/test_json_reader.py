import json
import unittest

import pytest

from truthbench.readers.json_reader import JsonReader


@pytest.fixture
def save_json(tmp_path):
    def _write_json(content):
        file_path = tmp_path / "data.json"
        with open(file_path, "w") as f:
            json.dump(content, f)
        return file_path

    return _write_json


def test_samples_reads_json_correctly(save_json):
    test_data = [
        {"question": "What is Python?", "ground_truth": "A programming language."},
        {"question": "What is 2+2?", "ground_truth": "4"}
    ]
    file_path = save_json(test_data)

    reader = JsonReader(file_path)
    result = reader.samples()

    assert result == test_data


@pytest.mark.parametrize("test_input,error_message", [
    ({"question": "What is 2+2?", "ground_truth": "4"}, r"Expected top-level JSON array \(list of samples\)"),
    ([["What is 2+2?", "4"]], r"Samples must be JSON objects"),
    ([{"q": "Where?", "a": "There"}], r"Missing required keys: 'question' and 'ground_truth'"),
])
def test_invalid_json(save_json, test_input, error_message):
    file_path = save_json(test_input)

    reader = JsonReader(file_path)

    with pytest.raises(ValueError, match=error_message) as e:
        reader.samples()


def test_samples_empty_file(save_json):
    file_path = save_json([])

    reader = JsonReader(file_path)
    result = reader.samples()

    assert result == []

    file_path.unlink()


if __name__ == "__main__":
    unittest.main()
