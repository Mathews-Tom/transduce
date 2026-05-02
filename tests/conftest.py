"""Shared pytest fixtures for the transduce test suite.

Fixture files live under ``tests/fixtures/`` as JSON arrays of objects with at
least ``original`` and ``transformed`` string fields. ``label`` and ``category``
are optional descriptors used by verifier tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_corpus(name: str) -> list[dict[str, Any]]:
    """Load and structurally validate a fixture corpus by basename."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"fixture corpus missing: {path} - corpora are required deliverables"
        )
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array, got {type(data).__name__}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object, got {type(item).__name__}")
        for required in ("original", "transformed"):
            if required not in item or not isinstance(item[required], str):
                raise ValueError(f"{path}[{index}] missing required string field {required!r}")
    return data


@pytest.fixture(scope="session")
def text_pairs() -> list[dict[str, Any]]:
    """General paraphrase pairs covering accept and reject cases."""
    return _load_corpus("text_pairs")


@pytest.fixture(scope="session")
def negation_pairs() -> list[dict[str, Any]]:
    """Pairs that flip negation cues - canonical reject set for negation diff."""
    return _load_corpus("negation_pairs")


@pytest.fixture(scope="session")
def entity_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb named entities - reject set for entity preservation."""
    return _load_corpus("entity_pairs")


@pytest.fixture(scope="session")
def number_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb numerical values - reject set for number preservation."""
    return _load_corpus("number_pairs")


@pytest.fixture(scope="session")
def url_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb URLs - reject set for URL preservation."""
    return _load_corpus("url_pairs")


@pytest.fixture(scope="session")
def date_pairs() -> list[dict[str, Any]]:
    """Pairs that perturb dates and temporal markers - reject set for date preservation."""
    return _load_corpus("date_pairs")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Return a temporary directory for transduce config files.

    Cleaned up automatically by pytest's ``tmp_path`` fixture.
    """
    config_dir = tmp_path / "transduce-config"
    config_dir.mkdir()
    return config_dir
