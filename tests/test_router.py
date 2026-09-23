"""Tests for routing policy decisions, including edge cases."""

from src.config import RoutingPolicy
from src.router import Router, Task


def make_task(task_id="t1", prompt="do the thing", complexity=2.0,
              min_quality=0.7, category="qa"):
    return Task(id=task_id, prompt=prompt, complexity=complexity,
                min_quality=min_quality, category=category)


def test_below_threshold_routes_flash(policy):
    d = Router(policy).route(make_task(complexity=2.0))
    assert d.tier == "flash"
    assert any("flash" in r for r in d.reasons)


def test_above_threshold_routes_pro(policy):
    d = Router(policy).route(make_task(complexity=7.0))
    assert d.tier == "pro"


def test_at_threshold_is_inclusive_flash(policy):
    # Complexity exactly at the threshold must go to the cheap tier.
    d = Router(policy).route(make_task(complexity=4.0))
    assert d.tier == "flash"


def test_just_above_threshold_routes_pro(policy):
    d = Router(policy).route(make_task(complexity=4.0001))
    assert d.tier == "pro"


def test_budget_exhausted_forces_flash():
    # Session budget already spent: even a pro-worthy task downgrades.
    policy = RoutingPolicy(complexity_threshold=4.0, session_budget_usd=0.0001)
    router = Router(policy)
    router.record_spend(0.0001)  # budget fully consumed
    d = router.route(make_task(complexity=9.0))
    assert d.tier == "flash"
    assert any("budget" in r for r in d.reasons)


def test_per_task_cap_forces_flash():
    policy = RoutingPolicy(complexity_threshold=4.0, cost_budget_per_task_usd=1e-9)
    d = Router(policy).route(make_task(complexity=9.0))
    assert d.tier == "flash"
    assert any("budget" in r for r in d.reasons)


def test_latency_slo_downgrades_pro_when_flash_fits():
    # SLO below any pro estimate: hard task downgrades to flash with a reason.
    policy = RoutingPolicy(complexity_threshold=4.0, latency_slo_ms=500.0)
    d = Router(policy).route(make_task(complexity=8.0))
    assert d.tier == "flash"
    assert any("SLO" in r for r in d.reasons)
    assert d.estimated_latency_ms <= 500.0


def test_latency_slo_keeps_pro_when_nothing_fits():
    # Absurdly tight SLO: no downgrade possible, pro stays (documented behavior).
    policy = RoutingPolicy(complexity_threshold=4.0, latency_slo_ms=1.0)
    d = Router(policy).route(make_task(complexity=8.0))
    assert d.tier == "pro"


def test_budget_never_upgrades_flash_task():
    # A cheap task stays cheap even with unlimited budget.
    policy = RoutingPolicy(complexity_threshold=4.0, session_budget_usd=1000.0)
    d = Router(policy).route(make_task(complexity=1.0))
    assert d.tier == "flash"


def test_decision_carries_audit_trail(policy):
    d = Router(policy).route(make_task(task_id="t9", complexity=8.0))
    assert d.task_id == "t9"
    assert d.reasons, "every decision must explain itself"
    assert d.estimated_cost_usd > 0
    assert d.estimated_latency_ms > 0


def test_record_spend_accumulates(policy):
    router = Router(policy)
    router.record_spend(0.01)
    router.record_spend(0.02)
    assert router.spent_usd == 0.03
