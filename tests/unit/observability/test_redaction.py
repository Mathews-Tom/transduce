"""Tests for the sha256_8 redaction helper (P3-OBS-03)."""

from __future__ import annotations

import hashlib

import pytest

from transduce.observability.redaction import sha256_8


@pytest.mark.unit
def test_sha256_8_returns_first_eight_hex_characters_of_sha256_digest() -> None:
    text = "the quick brown fox"
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    actual = sha256_8(text)

    assert actual == expected
    assert len(actual) == 8


@pytest.mark.unit
def test_sha256_8_is_deterministic_for_identical_inputs() -> None:
    text = "Acme reported $4.2M in Q3 revenue."

    first = sha256_8(text)
    second = sha256_8(text)

    assert first == second


@pytest.mark.unit
def test_sha256_8_distinguishes_minor_perturbations() -> None:
    original = "The launch succeeded across all regions."
    flipped = "The launch failed across all regions."

    assert sha256_8(original) != sha256_8(flipped)


@pytest.mark.unit
def test_sha256_8_handles_unicode_input_without_raising() -> None:
    text = "Q3 revenue grew — Größe der Belegschaft auf 47."

    digest = sha256_8(text)

    assert len(digest) == 8
    assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.unit
def test_sha256_8_empty_string_returns_known_prefix() -> None:
    expected = hashlib.sha256(b"").hexdigest()[:8]

    assert sha256_8("") == expected
