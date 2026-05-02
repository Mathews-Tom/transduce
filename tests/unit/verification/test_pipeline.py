"""Unit tests for the sequential verifier pipeline (P1-VER-05)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from transduce.verification.base import ScoreResult
from transduce.verification.pipeline import VerifierPipeline

pytestmark = pytest.mark.unit


class _RecordingScorer:
    """Test double scoring against a fixed verdict and incrementing a call counter."""

    invocations: ClassVar[list[str]] = []

    def __init__(self, *, name: str, verdict: str, span: str | None = None) -> None:
        self.name = name
        self._verdict = verdict
        self._span = span

    def score(self, original: str, candidate: str) -> ScoreResult:
        _RecordingScorer.invocations.append(self.name)
        return ScoreResult(
            name=self.name,
            value=1.0 if self._verdict == "accept" else 0.0,
            verdict=self._verdict,  # type: ignore[arg-type]
            rejection_reason=None if self._verdict == "accept" else f"{self.name} rejected",
            span=self._span,
        )


@pytest.fixture(autouse=True)
def _reset_invocations() -> None:
    _RecordingScorer.invocations.clear()


def test_pipeline_first_fail_short_circuits() -> None:
    pipeline = VerifierPipeline(
        [
            _RecordingScorer(name="cosine", verdict="reject", span="cosine span"),
            _RecordingScorer(name="entity", verdict="accept"),
        ]
    )

    outcome = pipeline.run("orig", "cand")

    assert outcome.verdict == "reject"
    assert outcome.failed_scorer == "cosine"
    assert outcome.span == "cosine span"
    assert _RecordingScorer.invocations == ["cosine"]


def test_pipeline_all_pass_returns_accept() -> None:
    pipeline = VerifierPipeline(
        [
            _RecordingScorer(name="cosine", verdict="accept"),
            _RecordingScorer(name="entity", verdict="accept"),
            _RecordingScorer(name="number", verdict="accept"),
        ]
    )

    outcome = pipeline.run("orig", "cand")

    assert outcome.verdict == "accept"
    assert outcome.failed_scorer is None
    assert _RecordingScorer.invocations == ["cosine", "entity", "number"]
    assert [r.name for r in outcome.results] == ["cosine", "entity", "number"]


def test_pipeline_runs_remaining_scorers_after_accept() -> None:
    pipeline = VerifierPipeline(
        [
            _RecordingScorer(name="cosine", verdict="accept"),
            _RecordingScorer(name="entity", verdict="reject"),
            _RecordingScorer(name="number", verdict="accept"),
        ]
    )

    outcome = pipeline.run("orig", "cand")

    assert outcome.verdict == "reject"
    assert outcome.failed_scorer == "entity"
    assert _RecordingScorer.invocations == ["cosine", "entity"]


def test_pipeline_empty_scorer_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one scorer"):
        VerifierPipeline([])


def test_pipeline_exposes_scorer_tuple() -> None:
    scorers = [
        _RecordingScorer(name="a", verdict="accept"),
        _RecordingScorer(name="b", verdict="accept"),
    ]
    pipeline = VerifierPipeline(scorers)

    assert tuple(s.name for s in pipeline.scorers) == ("a", "b")
