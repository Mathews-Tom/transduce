"""Unit tests for ``NegationDiffScorer`` (P2-VER-01)."""

from __future__ import annotations

from typing import Any

import pytest

from transduce.verification.negation import (
    NegationDiffScorer,
    diff_negation_cues,
    extract_negation_cues,
)


@pytest.mark.unit
def test_negation_did_to_did_not_returns_reject() -> None:
    scorer = NegationDiffScorer()

    result = scorer.score(
        "The deployment succeeded on the first attempt.",
        "The deployment did not succeed on the first attempt.",
    )

    assert result.verdict == "reject"
    assert result.span == "did not"


@pytest.mark.unit
def test_negation_double_negation_handling() -> None:
    scorer = NegationDiffScorer()

    result = scorer.score(
        "We had no problems with the rollout.",
        "We did not have no problems with the rollout.",
    )

    assert result.verdict == "reject"
    assert result.span == "did not"


@pytest.mark.unit
def test_negation_within_quoted_speech_ignored() -> None:
    scorer = NegationDiffScorer()

    result = scorer.score(
        'The CEO said "we cannot ship without tests".',
        'The chief said "we cannot ship without tests".',
    )

    assert result.verdict == "accept"


@pytest.mark.unit
def test_negation_no_change_returns_pass() -> None:
    scorer = NegationDiffScorer()

    result = scorer.score(
        "The deployment finished on time.",
        "Deployment finished on schedule.",
    )

    assert result.verdict == "accept"


@pytest.mark.unit
def test_negation_contraction_recognised_as_cue() -> None:
    scorer = NegationDiffScorer()

    result = scorer.score(
        "The build finished cleanly.",
        "The build didn't finish cleanly.",
    )

    assert result.verdict == "reject"
    assert result.span == "didn't"


@pytest.mark.unit
def test_negation_paraphrase_corpus_all_pass(negation_pairs: list[dict[str, Any]]) -> None:
    """Every accept-labelled fixture pair must score accept."""
    scorer = NegationDiffScorer()

    for pair in negation_pairs:
        if pair.get("label") != "accept":
            continue
        result = scorer.score(pair["original"], pair["transformed"])
        assert result.verdict == "accept", (
            f"accept fixture rejected: {pair['original']!r} -> {pair['transformed']!r}"
        )


@pytest.mark.unit
def test_negation_reject_corpus_all_caught(negation_pairs: list[dict[str, Any]]) -> None:
    """Every reject-labelled fixture pair (insertion or removal) must reject."""
    scorer = NegationDiffScorer()

    for pair in negation_pairs:
        if pair.get("label") != "reject":
            continue
        result = scorer.score(pair["original"], pair["transformed"])
        assert result.verdict == "reject", (
            f"reject fixture missed: {pair['original']!r} -> {pair['transformed']!r}"
        )


@pytest.mark.unit
def test_extract_negation_cues_matches_multiword_pair() -> None:
    cues = extract_negation_cues("The vendor failed to deliver the equipment.")

    assert "failed to" in cues
    assert "to" not in cues


@pytest.mark.unit
def test_diff_negation_cues_separates_added_and_removed() -> None:
    diff = diff_negation_cues(
        "The team did not ship the feature.",
        "The team failed to ship the feature.",
    )

    assert diff.added == ("failed to",)
    assert diff.removed == ("did not",)
