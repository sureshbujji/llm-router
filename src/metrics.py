"""Aggregation math for simulation metrics: percentiles, means, cost rollups."""

import math


def percentile(data: list, p: float) -> float:
    """Nearest-rank percentile. Raises ValueError on empty input."""
    if not data:
        raise ValueError("percentile of empty data")
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be 0..100, got {p}")
    ordered = sorted(data)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(0, min(rank - 1, len(ordered) - 1))]


def p95(data: list) -> float:
    return percentile(data, 95)


def mean(data: list) -> float:
    if not data:
        raise ValueError("mean of empty data")
    return sum(data) / len(data)


def total_cost_usd(records: list) -> float:
    return sum(r["cost_usd"] for r in records)
