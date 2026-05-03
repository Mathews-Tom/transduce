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
_COMPOSITION_REQUIRED: tuple[str, ...] = (
    "category",
    "original",
    "stage_1",
    "stage_2",
    "expected_composite_verdict",
)
_FAITHFULNESS_LABELS: frozenset[str] = frozenset({"accept", "reject"})
_FAITHFULNESS_CATEGORIES: frozenset[str] = frozenset(
    {"negation", "antonym", "tense", "number", "entity", "fact_drift"}
)
_COMPOSITION_CATEGORIES: frozenset[str] = frozenset(
    {"faithful_chain", "drift_accumulated", "intensity_overshoot"}
)
_COMPOSITION_VERDICTS: frozenset[str] = frozenset({"accept", "reject"})


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


def load_faithfulness_v0_2_corpus(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and validate the transduce-faithfulness v0.2 corpus.

    Adds a strict ``language`` field check on top of the v0.1 schema:
    every record must declare its source language so multilingual
    subsets can be filtered without inferring from text. v0.1 records
    are upgraded to ``language="en"`` by the v0.2 builder; native v0.2
    records carry their actual language code.
    """
    target = path if path is not None else CORPUS_ROOT / "transduce_faithfulness_v0_2.jsonl"
    records = list(iter_jsonl(target))
    for index, record in enumerate(records):
        for field in _FAITHFULNESS_REQUIRED:
            if field not in record:
                raise CorpusError(f"{target}[{index}] missing required field {field!r}")
        if "language" not in record:
            raise CorpusError(f"{target}[{index}] missing required field 'language'")
        language = record["language"]
        if not isinstance(language, str) or not language:
            raise CorpusError(f"{target}[{index}] language must be a non-empty string")
        if record["label"] not in _FAITHFULNESS_LABELS:
            raise CorpusError(f"{target}[{index}] label must be one of {_FAITHFULNESS_LABELS}")
        if record["category"] not in _FAITHFULNESS_CATEGORIES:
            raise CorpusError(
                f"{target}[{index}] category {record['category']!r} not in "
                f"{_FAITHFULNESS_CATEGORIES}"
            )
    return records


def load_composition_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the transduce-composition corpus.

    Each record is a (original, stage_1, stage_2, expected_composite_verdict)
    triple covering faithful chains, accumulated drift, and intensity
    overshoot — the three failure modes the composite verifier must
    distinguish per ``docs/system-design.md`` §Composite Verifier.
    """
    target = path if path is not None else CORPUS_ROOT / "transduce_composition_v0_1.jsonl"
    records = list(iter_jsonl(target))
    for index, record in enumerate(records):
        for field in _COMPOSITION_REQUIRED:
            if field not in record:
                raise CorpusError(f"{target}[{index}] missing required field {field!r}")
        if record["expected_composite_verdict"] not in _COMPOSITION_VERDICTS:
            raise CorpusError(
                f"{target}[{index}] expected_composite_verdict must be one of "
                f"{_COMPOSITION_VERDICTS}"
            )
        if record["category"] not in _COMPOSITION_CATEGORIES:
            raise CorpusError(
                f"{target}[{index}] category {record['category']!r} not in "
                f"{_COMPOSITION_CATEGORIES}"
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
    "load_composition_corpus",
    "load_faithfulness_corpus",
    "load_faithfulness_v0_2_corpus",
    "load_injection_corpus",
]
