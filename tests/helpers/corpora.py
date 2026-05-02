"""Fixture-corpus loader shared by ``tests/conftest.py`` and the corpus suite.

Lives in a normal module rather than ``conftest.py`` so static analyzers can
follow the import; ``conftest.py`` is loaded by pytest's plugin mechanism and
is not always discoverable as an importable module by IDE tooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR: Path = Path(__file__).resolve().parent.parent / "fixtures"
REQUIRED_FIELDS: tuple[str, ...] = ("original", "transformed")


def load_corpus(name: str, *, fixtures_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load and structurally validate a fixture corpus by basename.

    Raises:
        FileNotFoundError: the named corpus file does not exist.
        ValueError: the file contents do not match the corpus contract.
    """
    base = fixtures_dir if fixtures_dir is not None else FIXTURES_DIR
    path = base / f"{name}.json"
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
        for required in REQUIRED_FIELDS:
            if required not in item or not isinstance(item[required], str):
                raise ValueError(f"{path}[{index}] missing required string field {required!r}")
    return data
