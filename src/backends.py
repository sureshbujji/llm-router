"""Mock model backends: deterministic stand-ins behind a shared protocol.

Each backend models cost, latency, and quality purely from the tier table
(config.py) and the task, so routing decisions have measurable, repeatable
consequences with no API key and no network.

Production swap: implement ``ModelBackend`` with a real API client per tier
(e.g. a FlashClient and a ProClient reading keys from .env), measure real
latency/cost per call, and plug a real quality signal (human labels or an
LLM judge) into ``Generation.quality``. The router, cache, and simulation
do not change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from src.config import TIERS, Tier

TOKENS_PER_WORD = 1.3  # rough blended in/out token estimate


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))


def estimate_cost_usd(tier: Tier, prompt: str, est_response_words: int = 60) -> float:
    tokens = estimate_tokens(prompt) + int(est_response_words * TOKENS_PER_WORD)
    return tier.price_per_1k_tokens_usd * tokens / 1000.0


def estimate_latency_ms(tier: Tier, complexity: float, prompt: str) -> float:
    # Small deterministic jitter from the prompt hash so identical-complexity
    # tasks don't all report byte-identical latencies. Stable across runs.
    jitter = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 41 - 20
    return tier.base_latency_ms + tier.latency_per_complexity_ms * complexity + jitter


class ModelBackend(Protocol):
    name: str

    def generate(self, prompt: str, complexity: float) -> "Generation":
        ...


@dataclass
class Generation:
    text: str
    tier: str
    quality: float  # 0..1, deterministic from complexity vs tier capability
    latency_ms: float  # modeled, deterministic
    cost_usd: float
    tokens: int


class MockBackend:
    """Deterministic mock backend for one tier."""

    def __init__(self, tier_name: str):
        self.tier = TIERS[tier_name]
        self.name = tier_name

    def quality(self, complexity: float) -> float:
        """Full quality up to the tier's capability, then capability/complexity decay."""
        t = self.tier
        if complexity <= t.capability:
            return t.max_quality
        return round(t.max_quality * t.capability / complexity, 4)

    def generate(self, prompt: str, complexity: float) -> Generation:
        digest = hashlib.md5(f"{self.name}:{prompt}".encode()).hexdigest()[:8]
        text = f"[{self.name}:{digest}] " + " ".join(prompt.split()[:24])
        tokens = estimate_tokens(prompt) + estimate_tokens(text)
        return Generation(
            text=text,
            tier=self.name,
            quality=self.quality(complexity),
            latency_ms=estimate_latency_ms(self.tier, complexity, prompt),
            cost_usd=self.tier.price_per_1k_tokens_usd * tokens / 1000.0,
            tokens=tokens,
        )


class MockFlash(MockBackend):
    def __init__(self):
        super().__init__("flash")


class MockPro(MockBackend):
    def __init__(self):
        super().__init__("pro")
