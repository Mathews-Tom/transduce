"""Shared pytest fixtures for the transduce test suite.

Fixture files live under ``tests/fixtures/`` as JSON arrays of objects. Each
object must declare ``original``, ``transformed``, ``label`` (one of ``accept``
or ``reject``) and ``category`` (a non-empty string). The structural contract is
enforced by ``tests.helpers.corpora.load_corpus`` at fixture-load time and
asserted by the corpus shape tests in ``tests/unit/test_fixtures.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.corpora import load_corpus


@pytest.fixture(scope="session")
def text_pairs() -> list[dict[str, Any]]:
    """General paraphrase pairs covering accept and reject cases."""
    return load_corpus("text_pairs")


@pytest.fixture(scope="session")
def negation_pairs() -> list[dict[str, Any]]:
    """Pairs that flip negation cues - canonical reject set for negation diff."""
    return load_corpus("negation_pairs")


@pytest.fixture(scope="session")
def entity_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb named entities - reject set for entity preservation."""
    return load_corpus("entity_pairs")


@pytest.fixture(scope="session")
def number_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb numerical values - reject set for number preservation."""
    return load_corpus("number_pairs")


@pytest.fixture(scope="session")
def url_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb URLs - reject set for URL preservation."""
    return load_corpus("url_pairs")


@pytest.fixture(scope="session")
def date_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb dates and temporal markers - reject set for date preservation."""
    return load_corpus("date_pairs")
