"""Tests for evaluation pipeline."""

import json

from src.evaluate import FAITHFULNESS_THRESHOLD, RELEVANCY_THRESHOLD


def test_golden_dataset_loads():
    with open("eval/golden_dataset.json") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1
    for item in data:
        assert "question" in item
        assert "expected_answer" in item


def test_thresholds_are_reasonable():
    assert 0 < FAITHFULNESS_THRESHOLD <= 1.0
    assert 0 <= RELEVANCY_THRESHOLD <= 1.0
