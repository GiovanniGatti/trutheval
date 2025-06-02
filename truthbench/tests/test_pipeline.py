import unittest

import pytest

from truthbench.pipeline import StrictTracker, Step, Reader, Pipeline


def test_stricttracker_initialization_and_access():
    allowed = {"a", "b"}
    tracker = StrictTracker(allowed)

    # All allowed keys initialized to zero
    assert tracker["a"] == 0
    assert tracker["b"] == 0

    # Setting allowed keys works
    tracker["a"] = 5
    assert tracker["a"] == 5

    # Accessing a disallowed key raises KeyError
    with pytest.raises(KeyError):
        _ = tracker["c"]

    # Setting a disallowed key raises KeyError
    with pytest.raises(KeyError):
        tracker["c"] = 10


class DummyStep(Step):
    def __init__(self, required_fields=None, counters=None):
        super().__init__(required_fields or frozenset(), counters or frozenset())

    def step(self, sample, tracker):
        # Simple step: increment a counter if present, add a field
        if "count" in tracker:
            tracker["count"] += 1
        sample["processed"] = True


class DummyReader(Reader):
    def __init__(self, samples):
        self._samples = samples

    def samples(self):
        return self._samples


def test_pipeline_run_basic():
    # Create a pipeline with one step requiring 'foo'
    step = DummyStep(required_fields=frozenset(("foo",)), counters=frozenset(("count",)))
    pipeline = Pipeline(with_progress=False).with_step(step)

    # Reader with two samples, both have 'foo'
    samples = [{"foo": 1}, {"foo": 2}]
    reader = DummyReader(samples)

    processed_samples, tracker = pipeline.run(reader)

    # Samples processed with step's effect
    assert all(s.get("processed") for s in processed_samples)
    assert len(processed_samples) == 2

    # Tracker counters
    assert tracker["input_samples"] == 2
    assert tracker["count"] == 2


def test_pipeline_run_missing_required_field():
    # Step requires 'foo'
    step = DummyStep(required_fields={"foo"})
    pipeline = Pipeline(with_progress=False).with_step(step)

    # Sample missing 'foo'
    samples = [{"bar": 1}]
    reader = DummyReader(samples)

    with pytest.raises(ValueError) as excinfo:
        pipeline.run(reader)
    assert "requires ['foo']" in str(excinfo.value)


def test_pipeline_run_multiple_steps_and_counters():
    class IncStep(Step):
        def __init__(self):
            super().__init__(required_fields=frozenset({"foo"}), counters=frozenset({"inc"}))

        def step(self, sample, tracker):
            tracker["inc"] = tracker.get("inc", 0) + 1

    class TagStep(Step):
        def __init__(self):
            super().__init__(required_fields=frozenset({"foo"}), counters=frozenset({"tagged"}))

        def step(self, sample, tracker):
            sample["tagged"] = True
            tracker["tagged"] = tracker.get("tagged", 0) + 1

    pipeline = Pipeline(with_progress=False).with_step(IncStep()).with_step(TagStep())
    samples = [{"foo": "x"} for _ in range(3)]
    reader = DummyReader(samples)

    processed_samples, tracker = pipeline.run(reader)

    assert all(s.get("tagged") for s in processed_samples)
    assert tracker["input_samples"] == 3
    assert tracker["inc"] == 3
    assert tracker["tagged"] == 3


if __name__ == "__main__":
    unittest.main()
