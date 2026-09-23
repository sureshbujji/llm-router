"""Make the repo root importable so tests can use `from src... import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.config import RoutingPolicy, TIERS  # noqa: E402
from src.router import Task  # noqa: E402


@pytest.fixture
def policy():
    return RoutingPolicy(
        complexity_threshold=4.0,
        latency_slo_ms=4000.0,
        cost_budget_per_task_usd=0.05,
        session_budget_usd=None,
    )


@pytest.fixture
def tiers():
    return TIERS


def make_task(task_id="t1", prompt="do the thing", complexity=2.0,
              min_quality=0.7, category="qa"):
    return Task(id=task_id, prompt=prompt, complexity=complexity,
                min_quality=min_quality, category=category)
