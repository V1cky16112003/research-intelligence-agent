from __future__ import annotations
"""Tests for the RAGAS evaluation runner."""
import json
import math
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.run_ragas import _apply_rate_limit, _mean, check_thresholds, THRESHOLDS


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


def test_check_thresholds_nan_is_a_failure():
    """A NaN metric must fail the gate, not silently pass it.

    `nan < threshold` is False in Python, so a naive `<` check alone would
    let a broken pipeline that produces NaN scores pass CI undetected.
    """
    metrics = {"faithfulness": float("nan"), "answer_relevancy": 0.85, "context_precision": 0.8}
    failures = check_thresholds(metrics)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_mean_excludes_nan_and_none():
    """_mean must average only real values, ignoring NaN/None entries."""
    assert _mean([0.8, 0.9, float("nan"), None]) == pytest.approx(0.85)


def test_mean_all_nan_returns_nan():
    """_mean must return NaN (not 0.0) when every sample is unscoreable."""
    result = _mean([float("nan"), None])
    assert math.isnan(result)


def test_rate_limit_allows_calls_under_the_cap():
    """Calls within the per-minute cap must not be delayed."""
    mock_client = MagicMock()
    original_chat_create = mock_client.chat.completions.create
    original_chat_create.return_value = "chat-response"
    mock_client.embeddings.create.return_value = "embed-response"

    limited = _apply_rate_limit(mock_client, max_calls_per_minute=5)

    start = time.monotonic()
    for _ in range(5):
        assert limited.chat.completions.create() == "chat-response"
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert original_chat_create.call_count == 5


def test_rate_limit_blocks_once_cap_is_exceeded():
    """The (max_calls + 1)th call within the window must block until it clears."""
    from eval.run_ragas import _SlidingWindowRateLimiter

    # Short window so the test runs fast: 2 calls per 0.3s, instead of waiting 60s.
    limiter = _SlidingWindowRateLimiter(max_calls=2, period_seconds=0.3)
    limiter.acquire()
    limiter.acquire()
    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.25  # had to wait for the window to clear, not instant
