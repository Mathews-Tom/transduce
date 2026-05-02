"""Structural validation of the fixture corpora shipped with the harness.

These tests assert the contracts that downstream verifier tests rely on:
- each corpus file loads, has the minimum pair count required by the dev plan,
- every pair has the required string fields,
- every pair declares a valid label drawn from the accept/reject vocabulary,
- the per-corpus reject categories match the corpus's purpose so that the
  scorer-specific suites can filter on category without surprises,
- and the underlying ``load_corpus`` validator rejects each malformed input
  shape it advertises in its message contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.corpora import load_corpus

VALID_LABELS: frozenset[str] = frozenset({"accept", "reject"})
MIN_PAIRS_PER_CORPUS: int = 30


def _assert_corpus_shape(corpus: list[dict[str, Any]], name: str) -> None:
    assert len(corpus) >= MIN_PAIRS_PER_CORPUS, (
        f"corpus {name} has {len(corpus)} pairs, dev plan requires >= {MIN_PAIRS_PER_CORPUS}"
    )
    for index, pair in enumerate(corpus):
        original = pair["original"]
        transformed = pair["transformed"]
        label = pair.get("label")
        category = pair.get("category")
        assert isinstance(original, str), f"{name}[{index}].original must be a string"
        assert original, f"{name}[{index}].original must be non-empty"
        assert isinstance(transformed, str), f"{name}[{index}].transformed must be a string"
        assert transformed, f"{name}[{index}].transformed must be non-empty"
        assert label in VALID_LABELS, (
            f"{name}[{index}].label must be one of {sorted(VALID_LABELS)}, got {label!r}"
        )
        assert isinstance(category, str), f"{name}[{index}].category must be a string"
        assert category, f"{name}[{index}].category must be non-empty"


def _categories(corpus: Iterable[dict[str, Any]]) -> set[str]:
    return {item["category"] for item in corpus}


@pytest.mark.unit
def test_text_pairs_corpus_satisfies_minimum_size_and_shape(
    text_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(text_pairs, "text_pairs")


@pytest.mark.unit
def test_negation_pairs_corpus_covers_inserted_and_removed_categories(
    negation_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(negation_pairs, "negation_pairs")
    categories = _categories(negation_pairs)
    assert "negation_inserted" in categories
    assert "negation_removed" in categories


@pytest.mark.unit
def test_entity_pairs_corpus_covers_dropped_swapped_and_preserved(
    entity_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(entity_pairs, "entity_pairs")
    categories = _categories(entity_pairs)
    assert "entity_dropped" in categories
    assert "entity_swapped" in categories
    assert any(c.startswith("entity_preserved") or c == "entity_normalized" for c in categories)


@pytest.mark.unit
def test_number_pairs_corpus_covers_decimal_unit_and_magnitude(
    number_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(number_pairs, "number_pairs")
    categories = _categories(number_pairs)
    assert "decimal_shifted" in categories
    assert "unit_changed" in categories
    assert "magnitude_shifted" in categories or "magnitude_word_changed" in categories


@pytest.mark.unit
def test_url_pairs_corpus_covers_dropped_and_preserved(
    url_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(url_pairs, "url_pairs")
    categories = _categories(url_pairs)
    assert "url_dropped" in categories
    assert "url_preserved" in categories


@pytest.mark.unit
def test_date_pairs_corpus_covers_dropped_changed_and_preserved(
    date_pairs: list[dict[str, Any]],
) -> None:
    _assert_corpus_shape(date_pairs, "date_pairs")
    categories = _categories(date_pairs)
    assert "date_dropped" in categories
    assert "date_changed" in categories
    assert "date_preserved" in categories


@pytest.mark.unit
def test_total_corpus_pair_count_meets_foundation_target(
    text_pairs: list[dict[str, Any]],
    negation_pairs: list[dict[str, Any]],
    entity_pairs: list[dict[str, Any]],
    number_pairs: list[dict[str, Any]],
    url_pairs: list[dict[str, Any]],
    date_pairs: list[dict[str, Any]],
) -> None:
    """Dev plan F-05: 200+ labeled pairs across six categories."""
    total = (
        len(text_pairs)
        + len(negation_pairs)
        + len(entity_pairs)
        + len(number_pairs)
        + len(url_pairs)
        + len(date_pairs)
    )
    assert total >= 200, f"total corpus pairs {total} below dev-plan F-05 target of 200"


@pytest.mark.unit
def test_load_corpus_missing_file_raises_filenotfounderror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fixture corpus missing"):
        load_corpus("nonexistent", fixtures_dir=tmp_path)


@pytest.mark.unit
def test_load_corpus_non_list_root_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON array"):
        load_corpus("broken", fixtures_dir=tmp_path)


@pytest.mark.unit
def test_load_corpus_non_dict_item_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0\] must be an object"):
        load_corpus("broken", fixtures_dir=tmp_path)


@pytest.mark.unit
def test_load_corpus_missing_required_field_raises_valueerror(tmp_path: Path) -> None:
    payload = [{"original": "hi"}]  # missing "transformed"
    (tmp_path / "broken.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required string field 'transformed'"):
        load_corpus("broken", fixtures_dir=tmp_path)
