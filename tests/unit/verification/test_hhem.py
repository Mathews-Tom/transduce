"""Unit tests for ``HHEMScorer`` (P2-VER-03)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from transduce.verification.hhem import HHEMScorer


def _stub_scorer(probability: float) -> Callable[[str, str], float]:
    def score_pair(_original: str, _candidate: str) -> float:
        return probability

    return score_pair


@pytest.mark.unit
def test_hhem_grounded_summary_passes() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(0.92), threshold=0.50)

    result = scorer.score(
        "The deployment finished on schedule.",
        "Deployment completed on time.",
    )

    assert result.verdict == "accept"
    assert result.value == pytest.approx(0.92)


@pytest.mark.unit
def test_hhem_hallucination_fails() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(0.18), threshold=0.50)

    result = scorer.score(
        "The product launched last quarter.",
        "The product launched last quarter to record-breaking demand.",
    )

    assert result.verdict == "reject"
    assert "below threshold" in (result.rejection_reason or "")


@pytest.mark.unit
def test_hhem_threshold_respected() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(0.60), threshold=0.70)

    result = scorer.score("original", "candidate")

    assert result.verdict == "reject"


@pytest.mark.unit
def test_hhem_at_threshold_accepts() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(0.50), threshold=0.50)

    result = scorer.score("original", "candidate")

    assert result.verdict == "accept"


@pytest.mark.unit
def test_hhem_rejects_out_of_range_probability() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(1.5))

    with pytest.raises(ValueError, match="out-of-range"):
        scorer.score("original", "candidate")


@pytest.mark.unit
def test_hhem_constructor_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        HHEMScorer(scorer=_stub_scorer(0.5), threshold=-0.1)


@pytest.mark.unit
def test_hhem_empty_input_raises_value_error() -> None:
    scorer = HHEMScorer(scorer=_stub_scorer(0.9))

    with pytest.raises(ValueError, match="non-empty"):
        scorer.score("", "candidate")
