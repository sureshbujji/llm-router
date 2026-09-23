"""Routing policy configuration: tier price/latency/quality table + policy knobs.

Swap guide: the tier table below is the single place that describes each
model tier. To point a tier at a real provider, keep the cost/latency profile
and replace the matching backend in backends.py with an API client that
implements the ModelBackend protocol.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tier:
    """Cost/latency/quality profile of one model tier.

    price_per_1k_tokens_usd: blended in+out price per 1k tokens (mock scale).
    base_latency_ms / latency_per_complexity_ms:
        modeled latency = base + per_complexity * task_complexity.
    capability: complexity (0-10) the tier serves at full quality; beyond it
        quality decays as capability / complexity (see backends.py).
    max_quality: quality ceiling of the tier (1.0 = perfect).
    """

    name: str
    price_per_1k_tokens_usd: float
    base_latency_ms: float
    latency_per_complexity_ms: float
    capability: float
    max_quality: float


# --- Tier table ------------------------------------------------------------
# Priced like a cheap/fast "flash" tier vs a strong/slow "pro" tier. The
# order-of-magnitude gaps mirror real provider pricing (a budget-class model
# vs a frontier reasoning-class model); absolute numbers are mock scale.
TIERS = {
    "flash": Tier(
        name="flash",
        price_per_1k_tokens_usd=0.00025,
        base_latency_ms=120.0,
        latency_per_complexity_ms=45.0,
        capability=4.0,
        max_quality=0.95,
    ),
    "pro": Tier(
        name="pro",
        price_per_1k_tokens_usd=0.0040,
        base_latency_ms=850.0,
        latency_per_complexity_ms=260.0,
        capability=9.5,
        max_quality=1.0,
    ),
}


@dataclass
class RoutingPolicy:
    """Routing policy knobs.

    complexity_threshold: task complexity <= threshold routes to flash,
        otherwise to pro. Tasks AT the threshold go to flash (inclusive).
    latency_slo_ms: if the chosen tier's estimated latency exceeds the SLO,
        fall back to flash when flash fits inside the SLO.
    cost_budget_per_task_usd: hard per-call cap; forces flash when exceeded.
    session_budget_usd: optional cumulative spend cap for a run; when the
        next call would exceed it, force flash (budget-exhausted path).
    """

    complexity_threshold: float = 4.0
    latency_slo_ms: float = 4000.0
    cost_budget_per_task_usd: float = 0.05
    session_budget_usd: Optional[float] = None
