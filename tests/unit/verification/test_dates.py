"""Unit tests for ``DatePreservationScorer`` (P2-VER-04)."""

from __future__ import annotations

from typing import Any

import pytest

from transduce.verification.dates import DatePreservationScorer, extract_date_tokens


@pytest.mark.unit
def test_date_q3_2025_to_recently_returns_reject() -> None:
    scorer = DatePreservationScorer()

    result = scorer.score(
        "Earnings are reported in Q3 2025.",
        "Earnings are reported recently.",
    )

    assert result.verdict == "reject"
    assert "q3 2025" in (result.span or "")


@pytest.mark.unit
def test_date_iso_format_preserved() -> None:
    scorer = DatePreservationScorer()

    result = scorer.score(
        "Migration begins on 2026-01-15 at 02:00 UTC.",
        "Migration begins on 2026-01-15 at 02:00 UTC.",
    )

    assert result.verdict == "accept"


@pytest.mark.unit
def test_date_relative_marker_preserved() -> None:
    scorer = DatePreservationScorer()

    result = scorer.score(
        "We expect a decision within two weeks.",
        "A decision is expected within two weeks.",
    )

    assert result.verdict == "accept"


@pytest.mark.unit
def test_date_changed_returns_reject() -> None:
    scorer = DatePreservationScorer()

    result = scorer.score(
        "The deadline is 2025-12-31.",
        "The deadline is 2026-01-31.",
    )

    assert result.verdict == "reject"


@pytest.mark.unit
def test_date_corpus_accept_pairs_pass(date_pairs: list[dict[str, Any]]) -> None:
    scorer = DatePreservationScorer()

    for pair in date_pairs:
        if pair.get("label") != "accept":
            continue
        result = scorer.score(pair["original"], pair["transformed"])
        assert result.verdict == "accept", (
            f"accept fixture rejected: {pair['original']!r} -> {pair['transformed']!r}"
        )


@pytest.mark.unit
def test_date_corpus_reject_pairs_caught(date_pairs: list[dict[str, Any]]) -> None:
    scorer = DatePreservationScorer()

    for pair in date_pairs:
        if pair.get("label") != "reject":
            continue
        result = scorer.score(pair["original"], pair["transformed"])
        assert result.verdict == "reject", (
            f"reject fixture missed: {pair['original']!r} -> {pair['transformed']!r}"
        )


@pytest.mark.unit
def test_extract_date_tokens_normalises_whitespace() -> None:
    tokens = extract_date_tokens("Earnings hit Q3   2025 last quarter.")

    assert "q3 2025" in tokens
    assert "last quarter" in tokens
