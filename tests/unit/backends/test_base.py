"""Unit tests for the shared backend protocol surface (P3-BACK-07)."""

from __future__ import annotations

import pytest

from transduce.backends.base import TokenPricing

pytestmark = pytest.mark.unit


def test_token_pricing_estimate_combines_in_and_out_rates() -> None:
    pricing = TokenPricing(in_per_million_usd=3.00, out_per_million_usd=15.00)

    estimate = pricing.estimate(tokens_in=1_000_000, tokens_out=200_000)

    assert estimate == pytest.approx(3.00 + 3.00)


def test_token_pricing_estimate_with_zero_tokens_returns_zero() -> None:
    pricing = TokenPricing(in_per_million_usd=3.00, out_per_million_usd=15.00)

    estimate = pricing.estimate(tokens_in=0, tokens_out=0)

    assert estimate == pytest.approx(0.0)


def test_token_pricing_estimate_handles_small_call() -> None:
    pricing = TokenPricing(in_per_million_usd=0.80, out_per_million_usd=4.00)

    estimate = pricing.estimate(tokens_in=2_500, tokens_out=500)

    assert estimate == pytest.approx(2_500 * 0.80 / 1_000_000 + 500 * 4.00 / 1_000_000)


def test_token_pricing_estimate_negative_tokens_raises_value_error() -> None:
    pricing = TokenPricing(in_per_million_usd=3.00, out_per_million_usd=15.00)

    with pytest.raises(ValueError, match="non-negative"):
        pricing.estimate(tokens_in=-1, tokens_out=10)


def test_token_pricing_negative_rate_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="in_per_million_usd"):
        TokenPricing(in_per_million_usd=-1.0, out_per_million_usd=15.0)
