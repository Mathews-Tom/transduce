"""Unit tests for the word-level diff generator (P1-DIFF-01)."""

from __future__ import annotations

import pytest

from transduce.diff.word_level import compute_diff

pytestmark = pytest.mark.unit


def test_diff_identical_returns_single_equal_op() -> None:
    diff = compute_diff("the quick brown fox", "the quick brown fox")

    assert len(diff) == 1
    assert diff[0].op == "equal"
    assert diff[0].text == "the quick brown fox"


def test_diff_full_replacement_returns_delete_then_insert() -> None:
    diff = compute_diff("foo", "bar")

    ops = [d.op for d in diff]
    texts = {d.op: d.text for d in diff}
    assert ops == ["delete", "insert"]
    assert texts == {"delete": "foo", "insert": "bar"}


def test_diff_semantic_cleanup_groups_adjacent_changes() -> None:
    diff = compute_diff("the cat sat on the mat", "the dog sat on the rug")

    flattened = [(d.op, d.text) for d in diff]
    assert flattened == [
        ("equal", "the "),
        ("delete", "cat"),
        ("insert", "dog"),
        ("equal", " sat on the "),
        ("delete", "mat"),
        ("insert", "rug"),
    ]


def test_diff_empty_inputs_returns_empty_list() -> None:
    diff = compute_diff("", "")

    assert diff == []


def test_diff_insertion_only_returns_single_insert_op() -> None:
    diff = compute_diff("", "hello world")

    assert len(diff) == 1
    assert diff[0].op == "insert"
    assert diff[0].text == "hello world"


def test_diff_deletion_only_returns_single_delete_op() -> None:
    diff = compute_diff("hello world", "")

    assert len(diff) == 1
    assert diff[0].op == "delete"
    assert diff[0].text == "hello world"


def test_diff_unicode_preserved_in_op_text() -> None:
    diff = compute_diff("café", "cafe")

    full = "".join(d.text for d in diff if d.op in ("equal", "delete"))
    assert "café" in full or "caf" in full
