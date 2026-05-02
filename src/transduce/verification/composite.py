"""Composite verifier for compose chains (P3-COMP-02, P3-COMP-05).

Per-stage verifiers ensure each link in the chain produces a candidate
that meets that stage's profile. They cannot detect drift that
accumulates across stages — three small drifts can stack into a
material drift while every per-stage check passes. The composite
verifier closes that gap by re-running the same scorer ensemble on the
``(original, final)`` pair after the chain finishes, gated by a
slacker threshold that accommodates the accumulated transformation.

The dev plan defines the default ``composite_threshold`` as
``min_step_threshold - 0.05`` (P3-COMP-05). The orchestrator computes
that delta from the chain's stage profiles and passes it explicitly so
this module owns no policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from transduce.verification.base import Scorer, ScoreResult


@dataclass(frozen=True)
class CompositeOutcome:
    """Aggregate outcome of a composite verifier run."""

    verdict: Literal["accept", "reject"]
    aggregate_score: float
    results: Sequence[ScoreResult]
    failed_scorer: str | None = None
    rejection_reason: str | None = None


class CompositeVerificationFailedError(RuntimeError):
    """Raised when the composite verifier rejects the (original, final) pair (P3-COMP-06)."""

    def __init__(
        self,
        *,
        last_candidate: str,
        outcome: CompositeOutcome,
        which_stage: int,
    ) -> None:
        super().__init__(
            f"composite verifier rejected the chain output after stage {which_stage}: "
            f"{outcome.rejection_reason or outcome.failed_scorer or 'unspecified'}"
        )
        self.last_candidate = last_candidate
        self.outcome = outcome
        self.which_stage = which_stage


class CompositeVerifier:
    """Run a fixed scorer ensemble on the ``(original, final)`` pair after a compose chain."""

    def __init__(self, *, scorers: Sequence[Scorer], threshold: float) -> None:
        if not scorers:
            raise ValueError("CompositeVerifier requires at least one scorer")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold}")
        self._scorers = tuple(scorers)
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def run(self, original: str, final: str) -> CompositeOutcome:
        results: list[ScoreResult] = []
        for scorer in self._scorers:
            result = scorer.score(original, final)
            results.append(result)
            if result.verdict == "reject":
                return CompositeOutcome(
                    verdict="reject",
                    aggregate_score=result.value,
                    results=tuple(results),
                    failed_scorer=result.name,
                    rejection_reason=result.rejection_reason,
                )
        aggregate = _aggregate(results)
        if aggregate < self._threshold:
            return CompositeOutcome(
                verdict="reject",
                aggregate_score=aggregate,
                results=tuple(results),
                failed_scorer="composite_threshold",
                rejection_reason=(
                    f"aggregate composite score {aggregate:.3f} below threshold "
                    f"{self._threshold:.3f}"
                ),
            )
        return CompositeOutcome(
            verdict="accept",
            aggregate_score=aggregate,
            results=tuple(results),
        )


def _aggregate(results: Sequence[ScoreResult]) -> float:
    """Mean of per-scorer values; matches the per-attempt aggregate in the budgeter."""
    if not results:
        return 0.0
    return sum(result.value for result in results) / len(results)


__all__ = [
    "CompositeOutcome",
    "CompositeVerificationFailedError",
    "CompositeVerifier",
]
