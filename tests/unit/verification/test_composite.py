"""Unit tests for the composite verifier (P3-COMP-02, P3-COMP-05..06)."""

from __future__ import annotations

import pytest

from transduce.verification.base import ScoreResult
from transduce.verification.composite import (
    CompositeVerificationFailedError,
    CompositeVerifier,
)

pytestmark = pytest.mark.unit


class _StubScorer:
    """Scorer test double that returns scripted verdicts."""

    def __init__(self, *, name: str, value: float, verdict: str = "accept") -> None:
        self.name = name
        self._value = value
        self._verdict = verdict

    def score(self, original: str, candidate: str) -> ScoreResult:
        del original, candidate
        return ScoreResult(
            name=self.name,
            value=self._value,
            verdict=self._verdict,  # type: ignore[arg-type]
            rejection_reason=None if self._verdict == "accept" else f"{self.name} rejected",
        )


def test_composite_verifier_all_accept_returns_accept_with_aggregate() -> None:
    scorers = [
        _StubScorer(name="cosine", value=0.92),
        _StubScorer(name="nli", value=0.88),
    ]
    verifier = CompositeVerifier(scorers=scorers, threshold=0.80)

    outcome = verifier.run("original", "final")

    assert outcome.verdict == "accept"
    assert outcome.aggregate_score == pytest.approx(0.90)
    assert outcome.failed_scorer is None


def test_composite_verifier_first_reject_short_circuits() -> None:
    scorers = [
        _StubScorer(name="cosine", value=0.20, verdict="reject"),
        _StubScorer(name="nli", value=0.88),
    ]
    verifier = CompositeVerifier(scorers=scorers, threshold=0.80)

    outcome = verifier.run("original", "final")

    assert outcome.verdict == "reject"
    assert outcome.failed_scorer == "cosine"
    assert outcome.aggregate_score == pytest.approx(0.20)


def test_composite_verifier_aggregate_below_threshold_rejects() -> None:
    scorers = [
        _StubScorer(name="cosine", value=0.70),
        _StubScorer(name="nli", value=0.65),
    ]
    verifier = CompositeVerifier(scorers=scorers, threshold=0.80)

    outcome = verifier.run("original", "final")

    assert outcome.verdict == "reject"
    assert outcome.failed_scorer == "composite_threshold"
    assert "below threshold" in (outcome.rejection_reason or "")


def test_composite_verifier_construction_rejects_empty_scorers() -> None:
    with pytest.raises(ValueError, match="at least one scorer"):
        CompositeVerifier(scorers=[], threshold=0.80)


def test_composite_verifier_construction_rejects_threshold_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold"):
        CompositeVerifier(scorers=[_StubScorer(name="x", value=0.5)], threshold=1.5)


def test_composite_verification_failed_error_carries_outcome_and_stage() -> None:
    scorers = [_StubScorer(name="cosine", value=0.20, verdict="reject")]
    verifier = CompositeVerifier(scorers=scorers, threshold=0.80)
    outcome = verifier.run("o", "f")

    err = CompositeVerificationFailedError(
        last_candidate="final-text",
        outcome=outcome,
        which_stage=3,
    )

    assert err.last_candidate == "final-text"
    assert err.which_stage == 3
    assert err.outcome.failed_scorer == "cosine"
    assert "stage 3" in str(err)
