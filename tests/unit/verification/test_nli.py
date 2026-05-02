"""Unit tests for ``BidirectionalNLIScorer`` (P2-VER-02)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from transduce.verification.nli import BidirectionalNLIScorer


def _stub_entailer(scores: dict[tuple[str, str], float]) -> Callable[[str, str], float]:
    """Build an entailer that looks up exact ``(premise, hypothesis)`` pairs."""

    def entail(premise: str, hypothesis: str) -> float:
        return scores[(premise, hypothesis)]

    return entail


@pytest.mark.unit
def test_nli_paraphrase_passes_both_directions() -> None:
    original = "The deployment finished on schedule."
    candidate = "The rollout completed on time."
    entail = _stub_entailer(
        {
            (original, candidate): 0.92,
            (candidate, original): 0.88,
        }
    )
    scorer = BidirectionalNLIScorer(entail=entail, threshold=0.70)

    result = scorer.score(original, candidate)

    assert result.verdict == "accept"
    assert result.value == pytest.approx(0.88)


@pytest.mark.unit
def test_nli_negation_fails_one_direction() -> None:
    original = "The deployment finished on schedule."
    candidate = "The deployment did not finish on schedule."
    entail = _stub_entailer(
        {
            (original, candidate): 0.05,
            (candidate, original): 0.04,
        }
    )
    scorer = BidirectionalNLIScorer(entail=entail, threshold=0.70)

    result = scorer.score(original, candidate)

    assert result.verdict == "reject"
    assert "forward" in (result.rejection_reason or "")


@pytest.mark.unit
def test_nli_hallucinated_qualifier_fails() -> None:
    original = "The product launched last quarter."
    candidate = "The product launched last quarter to record-breaking demand."
    entail = _stub_entailer(
        {
            (original, candidate): 0.55,
            (candidate, original): 0.95,
        }
    )
    scorer = BidirectionalNLIScorer(entail=entail, threshold=0.70)

    result = scorer.score(original, candidate)

    assert result.verdict == "reject"
    assert "forward" in (result.rejection_reason or "")
    assert result.value == pytest.approx(0.55)


@pytest.mark.unit
def test_nli_handles_long_inputs_with_chunking() -> None:
    base = "The system processed twelve requests. "
    original = base * 50
    candidate = base * 50
    call_log: list[tuple[str, str]] = []

    def entail(premise: str, hypothesis: str) -> float:
        call_log.append((premise, hypothesis))
        return 0.9

    scorer = BidirectionalNLIScorer(entail=entail, threshold=0.70, max_chunk_chars=200)

    result = scorer.score(original, candidate)

    assert result.verdict == "accept"
    assert len(call_log) > 2, "long input must produce more than two scoring calls"


@pytest.mark.unit
def test_nli_rejects_out_of_range_entailer_score() -> None:
    original = "Original."
    candidate = "Candidate."
    entail = _stub_entailer({(original, candidate): 1.5})
    scorer = BidirectionalNLIScorer(entail=entail)

    with pytest.raises(ValueError, match="out-of-range"):
        scorer.score(original, candidate)


@pytest.mark.unit
def test_nli_constructor_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        BidirectionalNLIScorer(entail=lambda *_: 1.0, threshold=1.5)


@pytest.mark.unit
def test_nli_constructor_rejects_bad_chunk_size() -> None:
    with pytest.raises(ValueError, match="max_chunk_chars"):
        BidirectionalNLIScorer(entail=lambda *_: 1.0, max_chunk_chars=0)


@pytest.mark.unit
def test_nli_empty_input_raises_value_error() -> None:
    scorer = BidirectionalNLIScorer(entail=lambda *_: 1.0)

    with pytest.raises(ValueError, match="non-empty"):
        scorer.score("", "candidate")
