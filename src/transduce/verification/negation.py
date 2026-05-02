"""Negation diff scorer (P2-VER-01).

Deterministic floor for the verifier ensemble: tokenises both texts, extracts
negation cues with simple lexical scope, and rejects when the multiset of
non-quoted cues differs between original and candidate. Catches the classic
``did → did not`` flip that cosine similarity reads as a near-paraphrase.

Cue list and scope rules are intentionally narrow. Sentences inside quoted
speech are excluded because a faithful paraphrase often re-attributes the
same quoted utterance, and rejecting on speech content fails the
paraphrase-no-negation accept set in ``tests/fixtures/negation_pairs.json``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from transduce.verification.base import ScoreResult

_NEGATION_CUES: Final[frozenset[str]] = frozenset(
    {
        "not",
        "never",
        "no",
        "nor",
        "neither",
        "without",
        "cannot",
        "hardly",
        "scarcely",
        "barely",
    }
)

_MULTI_WORD_CUES: Final[tuple[tuple[str, ...], ...]] = (
    ("fail", "to"),
    ("failed", "to"),
    ("fails", "to"),
    ("failing", "to"),
    ("unable", "to"),
    ("could", "not"),
    ("did", "not"),
    ("does", "not"),
    ("do", "not"),
    ("was", "not"),
    ("were", "not"),
    ("is", "not"),
    ("are", "not"),
    ("has", "not"),
    ("have", "not"),
    ("had", "not"),
    ("will", "not"),
    ("would", "not"),
    ("should", "not"),
    ("could", "not"),
    ("can", "not"),
    ("must", "not"),
)

_CONTRACTION_CUES: Final[frozenset[str]] = frozenset(
    {
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
        "can't",
        "cannot",
        "mustn't",
        "n't",
    }
)

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_LDQUO, _RDQUO = chr(0x201C), chr(0x201D)
_LSQUO, _RSQUO = chr(0x2018), chr(0x2019)
_QUOTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(
        (
            r'"[^"]*"',
            r"'[^']*'",
            f"{_LDQUO}[^{_RDQUO}]*{_RDQUO}",
            f"{_LSQUO}[^{_RSQUO}]*{_RSQUO}",
        )
    )
)


class NegationDiffResult(BaseModel):
    """Per-text negation cue inventories surfaced on the response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    added: tuple[str, ...] = Field(default_factory=tuple)
    removed: tuple[str, ...] = Field(default_factory=tuple)


def extract_negation_cues(text: str) -> list[str]:
    """Return the multiset of negation cues in ``text``, ignoring quoted spans.

    Tokenisation is whitespace + punctuation aware via a single regex; cues are
    matched case-insensitively. Multi-word cues (``did not``, ``fail to``) are
    matched as adjacent token pairs and consume both tokens so ``not`` is not
    double-counted on top of ``did not``.
    """
    stripped = _strip_quoted_spans(text)
    tokens = _TOKEN_PATTERN.findall(stripped.lower())
    cues: list[str] = []
    index = 0
    while index < len(tokens):
        pair = (tokens[index], tokens[index + 1]) if index + 1 < len(tokens) else None
        if pair is not None and pair in _MULTI_WORD_CUES:
            cues.append(f"{pair[0]} {pair[1]}")
            index += 2
            continue
        token = tokens[index]
        if token in _CONTRACTION_CUES or token in _NEGATION_CUES:
            cues.append(token)
        index += 1
    return cues


def diff_negation_cues(original: str, candidate: str) -> NegationDiffResult:
    """Return cues added and removed between ``original`` and ``candidate``."""
    original_counts = Counter(extract_negation_cues(original))
    candidate_counts = Counter(extract_negation_cues(candidate))
    added = sorted((candidate_counts - original_counts).elements())
    removed = sorted((original_counts - candidate_counts).elements())
    return NegationDiffResult(added=tuple(added), removed=tuple(removed))


class NegationDiffScorer:
    """Reject when negation cues are added or removed relative to the original."""

    name = "negation_diff"

    def score(self, original: str, candidate: str) -> ScoreResult:
        diff = diff_negation_cues(original, candidate)
        details: dict[str, list[str]] = {
            "added": list(diff.added),
            "removed": list(diff.removed),
        }
        if not diff.added and not diff.removed:
            return ScoreResult(name=self.name, value=1.0, verdict="accept", details=dict(details))
        if diff.added and diff.removed:
            reason = (
                f"negation cues changed: added {list(diff.added)}, removed {list(diff.removed)}"
            )
            span = diff.added[0]
        elif diff.added:
            reason = f"negation cue inserted: {list(diff.added)}"
            span = diff.added[0]
        else:
            reason = f"negation cue removed: {list(diff.removed)}"
            span = diff.removed[0]
        return ScoreResult(
            name=self.name,
            value=0.0,
            verdict="reject",
            rejection_reason=reason,
            span=span,
            details=dict(details),
        )


def _strip_quoted_spans(text: str) -> str:
    """Replace quoted spans with whitespace so cues inside them do not count."""
    return _QUOTE_PATTERN.sub(lambda match: " " * len(match.group(0)), text)


__all__ = [
    "NegationDiffResult",
    "NegationDiffScorer",
    "diff_negation_cues",
    "extract_negation_cues",
]
