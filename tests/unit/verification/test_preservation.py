"""Unit tests for preservation scorers (P1-VER-02..04)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from transduce.verification.preservation import (
    EntityExtractor,
    EntityPreservationScorer,
    NumberPreservationScorer,
    UrlPreservationScorer,
    extract_number_triples,
    extract_urls,
)

pytestmark = pytest.mark.unit


def _stub_extractor(entities: Sequence[str]) -> EntityExtractor:
    def extract(text: str) -> Sequence[str]:
        return [ent for ent in entities if ent in text]

    return extract


# ---- Entity preservation ---------------------------------------------------


def test_entity_preserved_returns_pass() -> None:
    scorer = EntityPreservationScorer(_stub_extractor(["Apple Inc."]))

    result = scorer.score(
        "Apple Inc. published the report.",
        "Apple Inc. released the document.",
    )

    assert result.verdict == "accept"
    assert result.value == 1.0


def test_entity_dropped_returns_reject() -> None:
    scorer = EntityPreservationScorer(_stub_extractor(["Acme Corp", "Globex Industries"]))

    result = scorer.score(
        "Acme Corp announced a partnership with Globex Industries.",
        "The company announced a partnership with another firm.",
    )

    assert result.verdict == "reject"
    assert result.rejection_reason is not None
    assert result.span == "Acme Corp"


def test_entity_substring_no_false_positive_apple_vs_apple_inc() -> None:
    scorer = EntityPreservationScorer(_stub_extractor(["Apple Inc."]))

    result = scorer.score(
        "Apple Inc. introduced new features.",
        "Apple introduced new features.",
    )

    assert result.verdict == "reject"
    assert result.span == "Apple Inc."


def test_entity_normalized_corp_to_corporation_returns_accept() -> None:
    scorer = EntityPreservationScorer(_stub_extractor(["Acme Corp"]))

    result = scorer.score(
        "Acme Corp announced a partnership.",
        "Acme Corporation announced a partnership.",
    )

    assert result.verdict == "accept"


def test_entity_extractor_returning_empty_passes_with_no_entities() -> None:
    scorer = EntityPreservationScorer(_stub_extractor([]))

    result = scorer.score("anything", "anything else")

    assert result.verdict == "accept"


def test_entity_preservation_scorer_runs_against_fixture_corpus_at_full_recall(
    entity_pairs: list[dict[str, Any]],
) -> None:
    rejects = [pair for pair in entity_pairs if pair["label"] == "reject"]

    detected = 0
    for pair in rejects:
        scorer = EntityPreservationScorer(_stub_extractor(_seed_entities(pair["original"])))
        result = scorer.score(pair["original"], pair["transformed"])
        if result.verdict == "reject":
            detected += 1

    assert detected == len(rejects), "100% recall on entity-drop reject set required"


# Curated entity list used to exercise EntityPreservationScorer against the
# fixture corpus without depending on spaCy in unit tests. The set covers
# every reject case in tests/fixtures/entity_pairs.json.
_FIXTURE_ENTITIES = (
    "Acme Corp",
    "Globex Industries",
    "Initech",
    "Sarah Mitchell",
    "James Liu",
    "Apex Foundation",
    "Helios Dynamics",
    "Northwind Trading",
    "Contoso Ltd.",
    "Fabrikam",
    "Tailspin Toys",
    "Wide World Importers",
    "Microsoft Research",
    "Adobe Systems",
    "Apple Inc.",
    "International Business Machines",
    "The Linux Foundation",
    "OpenAPI Initiative",
)


def _seed_entities(text: str) -> Sequence[str]:
    return [ent for ent in _FIXTURE_ENTITIES if ent in text]


# ---- Number preservation ---------------------------------------------------


def test_number_decimal_distinguished_0012_vs_012() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score(
        "The error rate dropped to 0.012 percent.",
        "The error rate dropped to 0.12 percent.",
    )

    assert result.verdict == "reject"
    assert "0.012" in (result.span or "")


def test_number_with_unit_preserves_unit() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score(
        "The contract is worth $5 million.",
        "The contract is worth €5 million.",
    )

    assert result.verdict == "reject"


def test_number_magnitude_word_preserved_94b_equals_94_billion() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score(
        "The acquisition was 94B dollars.",
        "The acquisition was 94 billion dollars.",
    )

    assert result.verdict == "accept"


def test_number_value_preserved_returns_accept() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score(
        "Revenue grew by 12.5% year over year.",
        "Year over year revenue grew 12.5%.",
    )

    assert result.verdict == "accept"


def test_number_value_changed_returns_reject() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score("100 events processed.", "200 events processed.")

    assert result.verdict == "reject"


def test_number_unit_dropped_for_unrecognised_tokens_accepts_paraphrase() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score(
        "The team grew from 12 to 30 engineers.",
        "The engineering team expanded from 12 to 30.",
    )

    assert result.verdict == "accept"


def test_number_data_unit_change_returns_reject() -> None:
    scorer = NumberPreservationScorer()

    result = scorer.score("Storage capacity is 256 GB.", "Storage capacity is 256 MB.")

    assert result.verdict == "reject"


def test_extract_number_triples_handles_currency_and_magnitude() -> None:
    triples = extract_number_triples("$5 million and 94B dollars")

    assert ("5", "$", "million") in triples
    assert ("94", "", "billion") in triples


def test_number_scorer_runs_against_fixture_corpus_at_full_recall(
    number_pairs: list[dict[str, Any]],
) -> None:
    rejects = [pair for pair in number_pairs if pair["label"] == "reject"]
    accepts = [pair for pair in number_pairs if pair["label"] == "accept"]
    scorer = NumberPreservationScorer()

    reject_hits = sum(
        1
        for pair in rejects
        if scorer.score(pair["original"], pair["transformed"]).verdict == "reject"
    )
    accept_hits = sum(
        1
        for pair in accepts
        if scorer.score(pair["original"], pair["transformed"]).verdict == "accept"
    )

    assert reject_hits == len(rejects), "100% recall on number-drift reject set required"
    assert accept_hits == len(accepts), "no false positives on number-preserved accept set"


# ---- URL preservation ------------------------------------------------------


def test_url_dropped_returns_reject() -> None:
    scorer = UrlPreservationScorer()

    result = scorer.score(
        "Documentation is at https://docs.example.com/api.",
        "Documentation is at the documentation site.",
    )

    assert result.verdict == "reject"
    assert "docs.example.com" in (result.span or "")


def test_url_with_query_string_preserved() -> None:
    scorer = UrlPreservationScorer()

    result = scorer.score(
        "See https://example.com/search?q=transduce&lang=en for details.",
        "Refer to https://example.com/search?q=transduce&lang=en for details.",
    )

    assert result.verdict == "accept"


def test_url_idn_handling_preserves_unicode_host() -> None:
    scorer = UrlPreservationScorer()

    result = scorer.score(
        "Visit https://例え.jp/page for the Japanese site.",
        "Visit https://例え.jp/page for the Japanese site.",
    )

    assert result.verdict == "accept"


def test_url_extract_strips_trailing_period() -> None:
    urls = extract_urls("See https://example.com/page.")

    assert urls == ["https://example.com/page"]


def test_url_scorer_runs_against_fixture_corpus_at_full_recall(
    url_pairs: list[dict[str, Any]],
) -> None:
    rejects = [pair for pair in url_pairs if pair["label"] == "reject"]
    scorer = UrlPreservationScorer()

    detected = sum(
        1
        for pair in rejects
        if scorer.score(pair["original"], pair["transformed"]).verdict == "reject"
    )

    assert detected == len(rejects), "100% recall on url-drop reject set required"
