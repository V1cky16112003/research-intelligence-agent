from __future__ import annotations
"""Tests for the RAGAS evaluation runner."""
import json
from pathlib import Path
from eval.run_ragas import check_thresholds, THRESHOLDS


def test_golden_set_loads():
    """Golden set file exists and has 20 valid questions."""
    path = Path("eval/golden_set.json")
    assert path.exists(), "eval/golden_set.json not found"
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 20
    for item in data:
        assert "question" in item
        assert "ground_truth" in item
        assert "id" in item


def test_check_thresholds_all_pass():
    """All metrics above threshold -> no failures."""
    metrics = {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8}
    failures = check_thresholds(metrics)
    assert failures == []


def test_check_thresholds_faithfulness_fails():
    """Faithfulness below threshold -> failure reported."""
    metrics = {"faithfulness": 0.5, "answer_relevancy": 0.85, "context_precision": 0.8}
    failures = check_thresholds(metrics)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_check_thresholds_all_fail():
    """All metrics below threshold -> three failures."""
    metrics = {"faithfulness": 0.1, "answer_relevancy": 0.2, "context_precision": 0.3}
    failures = check_thresholds(metrics)
    assert len(failures) == 3


def test_thresholds_values():
    """Thresholds are set to the expected values."""
    assert THRESHOLDS["faithfulness"] == 0.8
    assert THRESHOLDS["answer_relevancy"] == 0.75
    assert THRESHOLDS["context_precision"] == 0.7
