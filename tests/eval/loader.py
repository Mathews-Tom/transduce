"""Eval-corpus loader and structural validation helpers (P2 entry criterion).

The loader is the smallest possible reader the eval harness needs to
integrate with the runner that lands in v1.5 (P4-BENCH-02). It
deliberately does not run scorers; it only validates the corpus
structure so a malformed JSONL file fails fast before any model
inference is launched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CORPUS_ROOT = Path(__file__).resolve().parent

_FAITHFULNESS_REQUIRED: tuple[str, ...] = (
    "category",
    "original",
    "candidate",
    "label",
)
_INJECTION_REQUIRED: tuple[str, ...] = (
    "category",
    "prompt",
    "expected_detection",
)
_FAITHFULNESS_LABELS: frozenset[str] = frozenset({"accept", "reject"})
_FAITHFULNESS_CATEGORIES: frozenset[str] = frozenset(
    {"negation", "antonym", "tense", "number", "entity", "fact_drift"}
)


class CorpusError(RuntimeError):
    """Raised when a corpus file fails structural validation."""


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise CorpusError(f"corpus file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{line_number} invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise CorpusError(
                    f"{path}:{line_number} expected object, got {type(record).__name__}"
                )
            yield record


def load_faithfulness_corpus(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the transduce-faithfulness corpus."""
    target = path if path is not None else CORPUS_ROOT / "transduce_faithfulness_v0_1.jsonl"
    records = list(iter_jsonl(target))
    for index, record in enumerate(records):
        for field in _FAITHFULNESS_REQUIRED:
            if field not in record:
                raise CorpusError(f"{target}[{index}] missing required field {field!r}")
        if record["label"] not in _FAITHFULNESS_LABELS:
            raise CorpusError(f"{target}[{index}] label must be one of {_FAITHFULNESS_LABELS}")
        if record["category"] not in _FAITHFULNESS_CATEGORIES:
            raise CorpusError(
                f"{target}[{index}] category {record['category']!r} not in "
                f"{_FAITHFULNESS_CATEGORIES}"
            )
    return records


def load_injection_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the injection-attacks corpus."""
    target = path if path is not None else CORPUS_ROOT / "injection_attacks_v0_1.jsonl"
    records = list(iter_jsonl(target))
    for index, record in enumerate(records):
        for field in _INJECTION_REQUIRED:
            if field not in record:
                raise CorpusError(f"{target}[{index}] missing required field {field!r}")
        if not isinstance(record["expected_detection"], bool):
            raise CorpusError(f"{target}[{index}] expected_detection must be a boolean")
    return records


__all__ = [
    "CORPUS_ROOT",
    "CorpusError",
    "iter_jsonl",
    "load_faithfulness_corpus",
    "load_injection_corpus",
]
