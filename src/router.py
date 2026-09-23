"""Cost/latency-aware router: picks a model tier per task from policy.

Decision order:
  1. Base tier from the complexity threshold (at-threshold -> flash).
  2. Latency SLO: pro estimate above the SLO falls back to flash if flash fits.
  3. Budgets: per-task cap or exhausted session budget forces flash.
Every decision carries ``reasons`` so routing is auditable.
"""

from dataclasses import dataclass, field

from src.backends import estimate_cost_usd, estimate_latency_ms
from src.config import TIERS, RoutingPolicy


@dataclass
class Task:
    id: str
    prompt: str
    complexity: float  # 0..10 estimated difficulty
    min_quality: float  # quality floor; below this counts as a regression
    category: str = ""


@dataclass
class RoutingDecision:
    task_id: str
    tier: str
    reasons: list = field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_latency_ms: float = 0.0


class Router:
    def __init__(self, policy: RoutingPolicy = None, tiers=None):
        self.policy = policy or RoutingPolicy()
        self.tiers = tiers or TIERS
        self.spent_usd = 0.0

    def route(self, task: Task) -> RoutingDecision:
        p = self.policy
        reasons: list = []

        # 1. Base tier from the complexity threshold (inclusive -> flash).
        if task.complexity <= p.complexity_threshold:
            tier = "flash"
            reasons.append(
                f"complexity {task.complexity} <= threshold {p.complexity_threshold}: flash"
            )
        else:
            tier = "pro"
            reasons.append(
                f"complexity {task.complexity} > threshold {p.complexity_threshold}: pro"
            )

        cost = estimate_cost_usd(self.tiers[tier], task.prompt)
        latency = estimate_latency_ms(self.tiers[tier], task.complexity, task.prompt)

        # 2. Latency SLO: fall back to flash only if flash actually fits.
        if tier == "pro" and latency > p.latency_slo_ms:
            flash_latency = estimate_latency_ms(self.tiers["flash"], task.complexity, task.prompt)
            if flash_latency <= p.latency_slo_ms:
                tier = "flash"
                reasons.append(
                    f"latency SLO {p.latency_slo_ms:.0f}ms: pro est {latency:.0f}ms, "
                    f"flash est {flash_latency:.0f}ms -> downgrade to flash"
                )
                cost = estimate_cost_usd(self.tiers[tier], task.prompt)
                latency = flash_latency

        # 3. Budgets: per-task cap, or session budget exhausted.
        over_task_cap = cost > p.cost_budget_per_task_usd
        budget_exhausted = (
            p.session_budget_usd is not None
            and self.spent_usd + cost > p.session_budget_usd
        )
        if tier != "flash" and (over_task_cap or budget_exhausted):
            why = (
                "session budget exhausted"
                if budget_exhausted
                else f"per-task cap ${p.cost_budget_per_task_usd:.4f} exceeded"
            )
            tier = "flash"
            reasons.append(f"budget: {why} -> downgrade to flash")
            cost = estimate_cost_usd(self.tiers[tier], task.prompt)
            latency = estimate_latency_ms(self.tiers[tier], task.complexity, task.prompt)

        return RoutingDecision(
            task_id=task.id,
            tier=tier,
            reasons=reasons,
            estimated_cost_usd=cost,
            estimated_latency_ms=latency,
        )

    def record_spend(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
