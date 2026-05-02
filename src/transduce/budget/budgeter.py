"""Per-request cost guard with non-improving-trend abort (P3-BUDG-01..04).

The :class:`Budgeter` tracks two things across the retry loop:

1. Cumulative cost in USD. Backends without pricing return ``None`` from
   ``cost_estimate``; the budgeter records ``0.0`` for those attempts so
   local deployments still see attempt counts but pay no money for them.
2. Verifier-score trend over a trailing window. When the trend is
   monotonically non-improving (each of the last ``non_improving_window``
   attempts failed to beat the previous one), the budgeter signals that
   further retries will not converge — the model has settled on a wrong
   answer and additional attempts only burn cost without improving the
   verdict.

The orchestrator instantiates one :class:`Budgeter` per request and
calls ``charge`` / ``record_score`` after every attempt. ``can_retry``
returns ``(True, None)`` when retries are still allowed and
``(False, reason)`` when the loop must exit; the orchestrator raises
:class:`BudgetExceededError` with the structured diagnostic in the
latter case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

_NonImprovingReason = Literal["budget_exceeded", "non_improving_trend"]
"""Possible reasons :meth:`Budgeter.can_retry` returns ``(False, reason)``."""


_TREND_FLOOR_ATTEMPTS: Final[int] = 2
"""Window of two attempts is the minimum that can express a trend."""


@dataclass(frozen=True)
class BudgetState:
    """Snapshot of the budgeter at the moment a decision was made."""

    total_cost_usd: float
    attempts: int
    scores: tuple[float, ...]


class BudgetExceededError(RuntimeError):
    """Raised when the cumulative cost exceeds ``max_cost_usd`` or the trend stalls."""

    def __init__(
        self,
        *,
        reason: _NonImprovingReason,
        state: BudgetState,
        limit: float,
    ) -> None:
        if reason == "budget_exceeded":
            message = (
                f"cumulative cost {state.total_cost_usd:.6f} USD exceeds limit "
                f"{limit:.6f} USD after {state.attempts} attempts"
            )
        else:
            message = (
                f"verifier scores non-improving over last {len(state.scores)} attempts "
                f"({list(state.scores)}); aborting retry loop"
            )
        super().__init__(message)
        self.reason: _NonImprovingReason = reason
        self.state = state
        self.limit = limit


class Budgeter:
    """Per-request budget guard for cost and verifier-trend checks."""

    def __init__(
        self,
        *,
        max_cost_usd: float,
        abort_on_non_improving_trend: bool = True,
        non_improving_window: int = 3,
    ) -> None:
        if max_cost_usd < 0.0:
            raise ValueError(f"max_cost_usd must be non-negative, got {max_cost_usd}")
        if non_improving_window < _TREND_FLOOR_ATTEMPTS:
            raise ValueError(
                f"non_improving_window must be >= {_TREND_FLOOR_ATTEMPTS}, "
                f"got {non_improving_window}"
            )
        self._max_cost_usd = max_cost_usd
        self._abort_on_non_improving_trend = abort_on_non_improving_trend
        self._window = non_improving_window
        self._total_cost = 0.0
        self._attempts = 0
        self._scores: list[float] = []

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    @property
    def state(self) -> BudgetState:
        return BudgetState(
            total_cost_usd=self._total_cost,
            attempts=self._attempts,
            scores=tuple(self._scores),
        )

    def charge(self, *, cost: float | None) -> None:
        """Add the cost of one attempt. ``None`` is treated as ``0.0``."""
        amount = 0.0 if cost is None else float(cost)
        if amount < 0.0:
            raise ValueError(f"attempt cost must be non-negative, got {amount}")
        self._total_cost += amount
        self._attempts += 1

    def record_score(self, *, score: float) -> None:
        """Record the candidate's verifier score (mean of per-scorer values)."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be within [0.0, 1.0], got {score}")
        self._scores.append(score)

    def can_retry(self) -> tuple[bool, _NonImprovingReason | None]:
        """Return whether another attempt is permitted."""
        if self._total_cost >= self._max_cost_usd:
            return False, "budget_exceeded"
        if self._abort_on_non_improving_trend and _is_non_improving(self._scores, self._window):
            return False, "non_improving_trend"
        return True, None


def _is_non_improving(scores: Sequence[float], window: int) -> bool:
    """Return True when the last ``window`` scores are monotonically non-increasing."""
    if len(scores) < window:
        return False
    tail = scores[-window:]
    return all(tail[i] <= tail[i - 1] for i in range(1, window))


__all__ = [
    "BudgetExceededError",
    "BudgetState",
    "Budgeter",
]
