"""Tests for the deterministic mock backends' quality model."""

from src.backends import MockFlash, MockPro


def test_flash_full_quality_within_capability():
    assert MockFlash().quality(4.0) == 0.95
    assert MockFlash().quality(1.0) == 0.95


def test_flash_quality_decays_beyond_capability():
    # capability 4.0, max 0.95 -> 0.95 * 4/8 at complexity 8
    assert MockFlash().quality(8.0) == 0.475


def test_pro_full_quality_within_capability():
    assert MockPro().quality(9.5) == 1.0


def test_pro_quality_decays_beyond_capability():
    assert MockPro().quality(10.0) == 0.95


def test_generate_is_deterministic():
    backend = MockFlash()
    g1 = backend.generate("hello world", 2.0)
    g2 = backend.generate("hello world", 2.0)
    assert g1 == g2


def test_generate_cost_scales_with_tier():
    flash = MockFlash().generate("some prompt here", 2.0)
    pro = MockPro().generate("some prompt here", 2.0)
    assert pro.cost_usd > flash.cost_usd
    assert pro.latency_ms > flash.latency_ms
