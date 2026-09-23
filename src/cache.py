"""Embedding-free semantic cache: TF-IDF vectors + cosine similarity.

Stdlib only (re, math). Deterministic: identical prompts always hit, and
similarity scores are stable across runs. Tracks hit rate plus estimated
cost/latency saved per hit.
"""

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")
CACHE_LOOKUP_MS = 8.0  # modeled cost of a cache lookup


def tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CacheEntry:
    prompt: str
    response: str
    tier: str
    quality: float
    cost_usd: float
    latency_ms: float


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    cost_saved_usd: float = 0.0
    latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self._entries: list = []
        self._doc_freq: dict = {}
        self.stats = CacheStats()

    def _vector(self, text: str) -> dict:
        tokens = tokenize(text)
        if not tokens:
            return {}
        counts: dict = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        n_docs = len(self._entries)
        vec = {}
        for tok, count in counts.items():
            tf = count / len(tokens)
            # Smoothed IDF over the stored corpus; unseen terms get df=0.
            idf = math.log((1 + n_docs) / (1 + self._doc_freq.get(tok, 0))) + 1.0
            vec[tok] = tf * idf
        return vec

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in a if t in b)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def similarity(self, prompt_a: str, prompt_b: str) -> float:
        """Cosine similarity of two prompts under the current corpus IDF."""
        return self._cosine(self._vector(prompt_a), self._vector(prompt_b))

    def lookup(self, prompt: str, est_cost_usd: float = 0.0,
               est_latency_ms: float = 0.0):
        """Return (entry, similarity) on hit, else (None, best_similarity).

        A hit records the estimated cost/latency the backend call would have
        incurred as saved. Boundary is inclusive: similarity == threshold hits.
        """
        if not self._entries:
            self.stats.misses += 1
            return None, 0.0
        query_vec = self._vector(prompt)
        best, best_sim = None, -1.0
        for entry in self._entries:
            sim = self._cosine(query_vec, self._vector(entry.prompt))
            if sim > best_sim:
                best, best_sim = entry, sim
        if best_sim >= self.threshold:
            self.stats.hits += 1
            self.stats.cost_saved_usd += est_cost_usd
            self.stats.latency_saved_ms += est_latency_ms
            return best, best_sim
        self.stats.misses += 1
        return None, best_sim

    def store(self, entry: CacheEntry) -> None:
        self._entries.append(entry)
        for tok in set(tokenize(entry.prompt)):
            self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1

    def __len__(self) -> int:
        return len(self._entries)
