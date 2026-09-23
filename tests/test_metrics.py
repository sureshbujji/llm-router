"""Tests for the metrics math: percentiles, means, cost aggregation."""

import pytest

from src.metrics import mean, p95, percentile, total_cost_usd


def test_percentile_known_values():
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(data, 50) == 50  # nearest-rank: ceil(0.5*10)=5th -> 50
    assert percentile(data, 95) == 100  # ceil(0.95*10)=10th -> 100
    assert percentile(data, 0) == 10
    assert percentile(data, 100) == 100


def test_p95_single_element():
    assert p95([42.0]) == 42.0


def test_p95_unsorted_input():
    assert p95([90, 10, 50, 30, 70]) == 90  # ceil(0.95*5)=5th of sorted


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


def test_total_cost_usd_sums_records():
    records = [{"cost_usd": 0.001}, {"cost_usd": 0.0025}, {"cost_usd": 0.0}]
    assert total_cost_usd(records) == pytest.approx(0.0035)


def test_total_cost_usd_empty():
    assert total_cost_usd([]) == 0.0
