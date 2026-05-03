"""Structural tests for the v0.1 eval corpora.

Full per-mode AUROC validation runs nightly under
``@pytest.mark.eval``; this file asserts the structural contract
(record counts, category coverage, label balance) so a malformed
corpus fails before the slow eval harness loads model weights.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.eval.loader import (
    load_faithfulness_corpus,
    load_faithfulness_v0_2_corpus,
    load_injection_corpus,
)

pytestmark = pytest.mark.unit


def test_faithfulness_corpus_loads_with_at_least_200_records() -> None:
    records = load_faithfulness_corpus()

    assert len(records) >= 200


def test_faithfulness_corpus_covers_all_six_categories() -> None:
    records = load_faithfulness_corpus()

    categories = {record["category"] for record in records}

    assert categories == {
        "negation",
        "antonym",
        "tense",
        "number",
        "entity",
        "fact_drift",
    }


def test_faithfulness_corpus_each_category_has_at_least_30_records() -> None:
    records = load_faithfulness_corpus()
    counts = Counter(record["category"] for record in records)

    for category, count in counts.items():
        assert count >= 30, f"category {category!r} has only {count} records"


def test_faithfulness_corpus_has_both_accept_and_reject_per_category() -> None:
    records = load_faithfulness_corpus()
    by_category: dict[str, Counter[str]] = {}
    for record in records:
        by_category.setdefault(record["category"], Counter())[record["label"]] += 1

    for category, label_counts in by_category.items():
        assert label_counts["accept"] > 0, f"{category} has no accept records"
        assert label_counts["reject"] > 0, f"{category} has no reject records"


def test_injection_corpus_loads_with_at_least_100_records() -> None:
    records = load_injection_corpus()

    assert len(records) >= 100


def test_injection_corpus_has_balanced_attack_and_benign_classes() -> None:
    records = load_injection_corpus()
    detection = Counter(record["expected_detection"] for record in records)

    assert detection[True] >= 50, "injection corpus needs at least 50 attack prompts"
    assert detection[False] >= 10, (
        "injection corpus needs at least 10 benign prompts to measure FP rate"
    )


def test_injection_corpus_covers_documented_attack_categories() -> None:
    records = load_injection_corpus()
    attack_categories = {record["category"] for record in records if record["expected_detection"]}

    assert {
        "ignore_previous_instructions",
        "role_flip",
        "system_prompt_leak",
        "fence_breakout",
        "exfiltration",
    }.issubset(attack_categories)


# ---------------------------------------------------------------------------
# transduce-faithfulness v0.2 (multilingual subset)
# ---------------------------------------------------------------------------


def test_faithfulness_v0_2_loads_with_at_least_300_records() -> None:
    records = load_faithfulness_v0_2_corpus()

    assert len(records) >= 300


def test_faithfulness_v0_2_languages_cover_en_de_fr() -> None:
    records = load_faithfulness_v0_2_corpus()
    languages = {record["language"] for record in records}

    assert {"en", "de", "fr"}.issubset(languages)


def test_faithfulness_v0_2_every_record_declares_language() -> None:
    records = load_faithfulness_v0_2_corpus()

    for index, record in enumerate(records):
        assert isinstance(record.get("language"), str), (
            f"record[{index}] is missing a language string"
        )
        assert record["language"], f"record[{index}] has empty language"


def test_faithfulness_v0_2_has_at_least_thirty_non_english_records() -> None:
    records = load_faithfulness_v0_2_corpus()
    non_english = [record for record in records if record["language"] != "en"]

    assert len(non_english) >= 30


def test_faithfulness_v0_2_balanced_accept_and_reject_per_language() -> None:
    records = load_faithfulness_v0_2_corpus()
    by_language: dict[str, Counter[str]] = {}
    for record in records:
        by_language.setdefault(record["language"], Counter())[record["label"]] += 1

    for language, label_counts in by_language.items():
        assert label_counts["accept"] > 0, f"language {language!r} has no accept records"
        assert label_counts["reject"] > 0, f"language {language!r} has no reject records"
