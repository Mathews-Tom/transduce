"""Unit tests for the per-request cost guard (P3-BUDG-01..04)."""

from __future__ import annotations

import pytest

from transduce.budget.budgeter import (
    BudgetExceededError,
    Budgeter,
)

pytestmark = pytest.mark.unit


def test_budgeter_charge_accumulates_total_cost_and_attempts() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)

    budgeter.charge(cost=0.01)
    budgeter.charge(cost=0.005)

    assert budgeter.total_cost_usd == pytest.approx(0.015)
    assert budgeter.attempts == 2


def test_budgeter_charge_with_none_cost_treats_as_zero() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)

    budgeter.charge(cost=None)
    budgeter.charge(cost=None)

    assert budgeter.total_cost_usd == pytest.approx(0.0)
    assert budgeter.attempts == 2


def test_budgeter_can_retry_returns_true_when_under_budget() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)
    budgeter.charge(cost=0.01)
    budgeter.record_score(score=0.4)

    allowed, reason = budgeter.can_retry()

    assert allowed is True
    assert reason is None


def test_budgeter_can_retry_returns_false_when_budget_reached() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)
    budgeter.charge(cost=0.05)

    allowed, reason = budgeter.can_retry()

    assert allowed is False
    assert reason == "budget_exceeded"


def test_budgeter_can_retry_aborts_on_flat_trend_at_window() -> None:
    budgeter = Budgeter(
        max_cost_usd=10.0,
        non_improving_window=3,
    )
    for value in (0.4, 0.4, 0.4):
        budgeter.charge(cost=0.001)
        budgeter.record_score(score=value)

    allowed, reason = budgeter.can_retry()

    assert allowed is False
    assert reason == "non_improving_trend"


def test_budgeter_can_retry_aborts_on_strictly_decreasing_trend() -> None:
    budgeter = Budgeter(max_cost_usd=10.0, non_improving_window=3)
    for value in (0.6, 0.5, 0.4):
        budgeter.charge(cost=0.001)
        budgeter.record_score(score=value)

    allowed, reason = budgeter.can_retry()

    assert allowed is False
    assert reason == "non_improving_trend"


def test_budgeter_can_retry_allows_when_trend_is_improving() -> None:
    budgeter = Budgeter(max_cost_usd=10.0, non_improving_window=3)
    for value in (0.3, 0.4, 0.5):
        budgeter.charge(cost=0.001)
        budgeter.record_score(score=value)

    allowed, reason = budgeter.can_retry()

    assert allowed is True
    assert reason is None


def test_budgeter_can_retry_allows_when_window_not_yet_full() -> None:
    budgeter = Budgeter(max_cost_usd=10.0, non_improving_window=3)
    budgeter.charge(cost=0.001)
    budgeter.record_score(score=0.3)
    budgeter.charge(cost=0.001)
    budgeter.record_score(score=0.3)

    allowed, reason = budgeter.can_retry()

    assert allowed is True
    assert reason is None


def test_budgeter_trend_abort_disabled_when_flag_false() -> None:
    budgeter = Budgeter(
        max_cost_usd=10.0,
        abort_on_non_improving_trend=False,
        non_improving_window=3,
    )
    for value in (0.4, 0.4, 0.4):
        budgeter.charge(cost=0.001)
        budgeter.record_score(score=value)

    allowed, reason = budgeter.can_retry()

    assert allowed is True
    assert reason is None


def test_budgeter_charge_negative_cost_raises_value_error() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)

    with pytest.raises(ValueError, match="non-negative"):
        budgeter.charge(cost=-0.01)


def test_budgeter_record_score_out_of_range_raises_value_error() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)

    with pytest.raises(ValueError, match="score"):
        budgeter.record_score(score=1.5)


def test_budgeter_construction_rejects_negative_max_cost() -> None:
    with pytest.raises(ValueError, match="max_cost_usd"):
        Budgeter(max_cost_usd=-0.01)


def test_budgeter_construction_rejects_window_below_two() -> None:
    with pytest.raises(ValueError, match="non_improving_window"):
        Budgeter(max_cost_usd=0.05, non_improving_window=1)


def test_budget_exceeded_error_includes_state_snapshot() -> None:
    budgeter = Budgeter(max_cost_usd=0.05)
    budgeter.charge(cost=0.05)
    state = budgeter.state

    err = BudgetExceededError(reason="budget_exceeded", state=state, limit=0.05)

    assert err.reason == "budget_exceeded"
    assert err.state.total_cost_usd == pytest.approx(0.05)
    assert err.state.attempts == 1
    assert err.limit == pytest.approx(0.05)
    assert "0.050000 USD exceeds limit 0.050000 USD" in str(err)
