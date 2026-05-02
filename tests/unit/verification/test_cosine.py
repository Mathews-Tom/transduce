"""Unit tests for the cosine similarity scorer (P1-VER-01)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from transduce.verification.cosine import CosineSimilarityScorer

pytestmark = pytest.mark.unit


def _stub_embedder(table: dict[str, Sequence[float]]) -> CosineSimilarityScorer:
    def embed(text: str) -> Sequence[float]:
        return table[text]

    return CosineSimilarityScorer(embed=embed, threshold=0.85)


def test_cosine_identical_inputs_returns_value_one() -> None:
    scorer = _stub_embedder({"hello": [1.0, 0.0], "world": [0.0, 1.0]})

    result = scorer.score("hello", "hello")

    assert result.verdict == "accept"
    assert result.value == pytest.approx(1.0)


def test_cosine_below_threshold_returns_reject() -> None:
    scorer = _stub_embedder(
        {
            "left": [1.0, 0.0, 0.0],
            "right": [0.0, 1.0, 0.0],
        }
    )

    result = scorer.score("left", "right")

    assert result.verdict == "reject"
    assert result.value == pytest.approx(0.0)
    assert result.rejection_reason is not None


def test_cosine_above_threshold_returns_accept() -> None:
    scorer = _stub_embedder(
        {
            "primary": [1.0, 0.1, 0.0],
            "paraphrase": [0.99, 0.0, 0.0],
        }
    )

    result = scorer.score("primary", "paraphrase")

    assert result.verdict == "accept"


def test_cosine_handles_empty_input_raises_value_error() -> None:
    scorer = CosineSimilarityScorer(embed=lambda _text: [1.0, 0.0])

    with pytest.raises(ValueError, match="non-empty"):
        scorer.score("", "anything")


def test_cosine_threshold_outside_unit_interval_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        CosineSimilarityScorer(embed=lambda _text: [1.0], threshold=1.2)


def test_cosine_dimension_mismatch_raises_value_error() -> None:
    scorer = CosineSimilarityScorer(
        embed=lambda text: [1.0, 0.0] if text == "a" else [1.0, 0.0, 0.0]
    )

    with pytest.raises(ValueError, match="embedding dimensions"):
        scorer.score("a", "b")


def test_cosine_zero_norm_embedding_raises_value_error() -> None:
    scorer = CosineSimilarityScorer(embed=lambda _text: [0.0, 0.0])

    with pytest.raises(ValueError, match="zero-norm"):
        scorer.score("a", "b")


def test_cosine_clamps_negative_similarity_to_zero() -> None:
    scorer = CosineSimilarityScorer(
        embed=lambda text: [1.0, 0.0] if text == "a" else [-1.0, 0.0],
        threshold=0.5,
    )

    result = scorer.score("a", "b")

    assert result.verdict == "reject"
    assert result.value == 0.0
