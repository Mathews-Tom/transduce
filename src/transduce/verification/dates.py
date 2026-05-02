"""Date preservation scorer (P2-VER-04).

Opt-in scorer surfaced when the mode (or per-request preserve list) declares
``preserve.dates``. Extracts ISO dates, fiscal markers (``Q3 2025``,
``fiscal year 2026``), explicit month-year forms, and a small set of
relative time markers, then rejects when any token from the original is
missing from the candidate.

The extractor is deterministic and dependency-free. Matching is exact-string
on the normalised token (``2025-12-31``, ``Q3 2025``, ``fiscal year 2026``)
so a paraphrase that rewords ``March 15, 2024`` to ``some time ago`` rejects
loudly, while reorderings and tense shifts that keep the token intact pass.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Final

from transduce.verification.base import ScoreResult

_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FISCAL_QUARTER: Final[re.Pattern[str]] = re.compile(
    r"\b(?:fiscal\s+)?Q[1-4](?:\s+\d{4})?\b",
    re.IGNORECASE,
)
_FISCAL_YEAR: Final[re.Pattern[str]] = re.compile(
    r"\bfiscal\s+year\s+\d{4}\b",
    re.IGNORECASE,
)
_MONTH_YEAR: Final[re.Pattern[str]] = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)(?:\s+\d{1,2},?)?\s+\d{4}\b",
    re.IGNORECASE,
)
_BARE_MONTH: Final[re.Pattern[str]] = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b",
    re.IGNORECASE,
)
_NUMBER_WORD: Final[str] = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|fifty)"
)

_RELATIVE_MARKERS: Final[re.Pattern[str]] = re.compile(
    r"\b(?:next|last|this)\s+(?:quarter|month|year|week)\b"
    r"|\b(?:end|start|beginning)\s+of\s+(?:the\s+)?(?:fiscal\s+year"
    r"|quarter|month|year)\b"
    rf"|\bwithin\s+{_NUMBER_WORD}\s+(?:weeks?|months?|days?|years?|quarters?)\b"
    r"|\bevery\s+(?:quarter|month|week|year|day)\b",
    re.IGNORECASE,
)

_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    _ISO_DATE,
    _FISCAL_YEAR,
    _FISCAL_QUARTER,
    _MONTH_YEAR,
    _RELATIVE_MARKERS,
    _BARE_MONTH,
)


def extract_date_tokens(text: str) -> list[str]:
    """Extract date tokens from ``text``, normalised to lowercase and de-spaced.

    Patterns are tried in priority order: ISO dates, fiscal-year forms, fiscal
    quarters, month-year, relative markers, then bare month names. Each match
    consumes its character span so a single text region contributes one token
    even when multiple patterns could match.
    """
    spans: list[tuple[int, int, str]] = []
    consumed: list[bool] = [False] * len(text)
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(consumed[start:end]):
                continue
            spans.append((start, end, _normalise_token(match.group(0))))
            consumed[start:end] = [True] * (end - start)
    spans.sort()
    return [token for _, _, token in spans]


_DETERMINER_RE: Final[re.Pattern[str]] = re.compile(r"\bthe\s+", re.IGNORECASE)


def _normalise_token(raw: str) -> str:
    """Lowercase, collapse whitespace, and strip the leading determiner ``the``.

    ``end of fiscal year`` and ``end of the fiscal year`` normalise to the same
    token so paraphrases that add or drop the determiner do not falsely reject.
    """
    collapsed = re.sub(r"\s+", " ", raw.strip()).lower()
    return _DETERMINER_RE.sub("", collapsed)


class DatePreservationScorer:
    """Reject when any date token from the original is missing in the candidate."""

    name = "date_preservation"

    def score(self, original: str, candidate: str) -> ScoreResult:
        original_tokens = Counter(extract_date_tokens(original))
        candidate_tokens = Counter(extract_date_tokens(candidate))
        missing = original_tokens - candidate_tokens
        if not missing:
            return ScoreResult(name=self.name, value=1.0, verdict="accept")
        first_missing = next(iter(sorted(missing.elements())))
        return ScoreResult(
            name=self.name,
            value=0.0,
            verdict="reject",
            rejection_reason=f"date token not preserved: {first_missing!r}",
            span=first_missing,
        )


__all__ = ["DatePreservationScorer", "extract_date_tokens"]
