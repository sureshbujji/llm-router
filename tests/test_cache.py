"""Tests for semantic cache hit/miss logic and similarity boundaries."""

import pytest

from src.cache import CacheEntry, SemanticCache


def make_entry(prompt, tier="flash", quality=0.9, cost=0.0001, latency=200.0):
    return CacheEntry(prompt=prompt, response=f"resp:{prompt[:10]}", tier=tier,
                      quality=quality, cost_usd=cost, latency_ms=latency)


def test_empty_cache_always_misses():
    cache = SemanticCache()
    entry, sim = cache.lookup("anything at all")
    assert entry is None
    assert sim == 0.0
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.0


def test_exact_prompt_hits():
    cache = SemanticCache()
    cache.store(make_entry("Translate 'order confirmation' to Spanish."))
    entry, sim = cache.lookup("Translate 'order confirmation' to Spanish.")
    assert entry is not None
    assert sim == pytest.approx(1.0)
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 1.0


def test_dissimilar_prompt_misses():
    cache = SemanticCache()
    cache.store(make_entry("Translate 'order confirmation' to Spanish."))
    entry, sim = cache.lookup("Derive backpropagation for a transformer attention layer.")
    assert entry is None
    assert sim < cache.threshold
    assert cache.stats.misses == 1


def test_paraphrase_hits():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.store(make_entry(
        "Summarize the Q3 sales report highlights in three bullet points."))
    entry, sim = cache.lookup(
        "Summarize the highlights of the Q3 sales report in three bullet points.")
    assert entry is not None, f"paraphrase should hit (sim={sim:.4f})"
    assert sim >= 0.85


def test_threshold_boundary_is_inclusive():
    # Pin the threshold exactly at a measured similarity: must hit.
    cache = SemanticCache(similarity_threshold=0.85)
    cache.store(make_entry("Draft a polite follow-up email about the overdue invoice."))
    probe = "Draft a polite follow-up email regarding the overdue invoice."
    measured = cache.similarity(
        "Draft a polite follow-up email about the overdue invoice.", probe)

    at_boundary = SemanticCache(similarity_threshold=measured)
    at_boundary.store(make_entry("Draft a polite follow-up email about the overdue invoice."))
    entry, _ = at_boundary.lookup(probe)
    assert entry is not None, "similarity == threshold must hit (inclusive)"

    just_above = SemanticCache(similarity_threshold=measured + 1e-6)
    just_above.store(make_entry("Draft a polite follow-up email about the overdue invoice."))
    entry, _ = just_above.lookup(probe)
    assert entry is None, "similarity < threshold must miss"


def test_hit_records_cost_and_latency_saved():
    cache = SemanticCache()
    cache.store(make_entry("What is the refund policy?"))
    cache.lookup("What is the refund policy?", est_cost_usd=0.004,
                 est_latency_ms=1500.0)
    assert cache.stats.cost_saved_usd == pytest.approx(0.004)
    assert cache.stats.latency_saved_ms == pytest.approx(1500.0)


def test_miss_records_no_savings():
    cache = SemanticCache()
    cache.store(make_entry("What is the refund policy?"))
    cache.lookup("Explain quantum entanglement.", est_cost_usd=0.004,
                 est_latency_ms=1500.0)
    assert cache.stats.cost_saved_usd == 0.0
    assert cache.stats.latency_saved_ms == 0.0


def test_best_match_wins():
    cache = SemanticCache()
    cache.store(make_entry("completely unrelated gibberish about zeppelins"))
    near = make_entry("Summarize the Q3 sales report highlights in three bullet points.")
    cache.store(near)
    entry, _ = cache.lookup(
        "Summarize the highlights of the Q3 sales report in three bullet points.")
    assert entry is near


def test_store_updates_corpus_size():
    cache = SemanticCache()
    assert len(cache) == 0
    cache.store(make_entry("one"))
    cache.store(make_entry("two"))
    assert len(cache) == 2
