"""Preservation scorers: entity, number, URL (P1-VER-02..04).

Each scorer accepts an injected dependency (entity extractor for the
entity scorer, regex compilation for the URL scorer) so unit tests can
exercise the scorer logic without loading spaCy or hitting network. The
production wiring builds the spaCy-backed entity extractor at startup
via :func:`build_spacy_entity_extractor`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Final

from transduce.verification.base import ScoreResult

EntityExtractor = Callable[[str], Sequence[str]]


# ---- Entity preservation ---------------------------------------------------


class EntityPreservationScorer:
    """Reject when any named entity from the original is missing from the candidate.

    Uses exact substring matching per docs/system-design.md §Verification
    Subsystem. Substring matching accepts entity normalisation
    (``Acme Corp`` → ``Acme Corporation``) while rejecting truncation
    (``Apple Inc.`` → ``Apple``) where the full original entity no longer
    appears in the candidate.
    """

    name = "entity_preservation"

    def __init__(self, extract_entities: EntityExtractor) -> None:
        self._extract = extract_entities

    def score(self, original: str, candidate: str) -> ScoreResult:
        for entity in self._extract(original):
            if entity and entity not in candidate:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    verdict="reject",
                    rejection_reason=f"entity not preserved: {entity!r}",
                    span=entity,
                )
        return ScoreResult(name=self.name, value=1.0, verdict="accept")


def build_spacy_entity_extractor(
    model_name: str = "en_core_web_sm",
) -> EntityExtractor:  # pragma: no cover — exercised in integration; covers IO at startup
    """Build a spaCy-backed extractor.

    Lazy-imports spaCy and loads ``model_name`` once at construction time so
    the production hot path never re-loads the pipeline. Operators must have
    the model installed (``python -m spacy download en_core_web_sm``);
    missing models surface immediately with the spaCy ``OSError``.
    """
    import spacy

    nlp = spacy.load(model_name)

    def extract(text: str) -> list[str]:
        doc = nlp(text)
        return [ent.text for ent in doc.ents]

    return extract


# ---- Number preservation ---------------------------------------------------


_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<currency>[$€£¥])?"
    r"(?P<value>-?\d+(?:[.,]\d+)?)"
    r"(?:\s*(?P<unit>[A-Za-z%]+))?"
)

_MAGNITUDE_NORMAL: Final[dict[str, str]] = {
    "k": "thousand",
    "thousand": "thousand",
    "m": "million",
    "million": "million",
    "b": "billion",
    "billion": "billion",
    "t": "trillion",
    "trillion": "trillion",
    "%": "percent",
    "percent": "percent",
    "pct": "percent",
}

_RECOGNIZED_UNITS: Final[frozenset[str]] = frozenset(
    {
        "kb",
        "mb",
        "gb",
        "tb",
        "pb",
        "byte",
        "bytes",
        "ms",
        "sec",
        "secs",
        "second",
        "seconds",
        "min",
        "mins",
        "minute",
        "minutes",
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "km",
        "mi",
        "mile",
        "miles",
        "ft",
        "feet",
        "inch",
        "inches",
        "ml",
        "l",
        "liter",
        "liters",
        "gallon",
        "gallons",
        "oz",
        "ounce",
        "ounces",
        "g",
        "kg",
        "lb",
        "lbs",
        "pound",
        "pounds",
    }
)


def _normalize_unit(unit: str) -> str:
    """Return a canonical unit string. Empty string for unrecognised tokens."""
    if not unit:
        return ""
    lowered = unit.lower()
    if lowered in _MAGNITUDE_NORMAL:
        return _MAGNITUDE_NORMAL[lowered]
    if lowered in _RECOGNIZED_UNITS:
        return lowered
    return ""


def extract_number_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract ``(value, currency, unit)`` triples from ``text``.

    The unit slot is normalised: magnitude words collapse onto a canonical
    form (``94B`` and ``94 billion`` both yield ``billion``); unrecognised
    trailing words become an empty string so paraphrases that drop
    non-load-bearing words (``30 engineers`` → ``30``) still compare equal.
    """
    triples: list[tuple[str, str, str]] = []
    for match in _NUMBER_PATTERN.finditer(text):
        currency = match.group("currency") or ""
        unit = _normalize_unit(match.group("unit") or "")
        triples.append((match.group("value"), currency, unit))
    return triples


def _format_triple(triple: tuple[str, str, str]) -> str:
    value, currency, unit = triple
    pieces = []
    if currency:
        pieces.append(currency)
    pieces.append(value)
    if unit:
        pieces.append(unit)
    return " ".join(pieces).strip()


class NumberPreservationScorer:
    """Reject when any (value, currency, unit) triple from the original is missing.

    Multiset semantics ensure repeats matter: ``30 30`` paraphrased to ``30``
    rejects on the missing repeat. Decimal positions are preserved verbatim
    so ``0.012`` and ``0.12`` never compare equal.
    """

    name = "number_preservation"

    def score(self, original: str, candidate: str) -> ScoreResult:
        candidate_triples: list[tuple[str, str, str]] = list(extract_number_triples(candidate))
        for triple in extract_number_triples(original):
            try:
                candidate_triples.remove(triple)
            except ValueError:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    verdict="reject",
                    rejection_reason=f"number not preserved: {_format_triple(triple)!r}",
                    span=_format_triple(triple),
                )
        return ScoreResult(name=self.name, value=1.0, verdict="accept")


# ---- URL preservation ------------------------------------------------------


_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://\S+", re.IGNORECASE)
_URL_TRAILING_PUNCT: Final[str] = ".,;:!?)]}>"


def extract_urls(text: str) -> list[str]:
    """Extract HTTP(S) URLs from ``text``, stripping trailing sentence punctuation."""
    urls: list[str] = []
    for match in _URL_PATTERN.finditer(text):
        urls.append(_strip_trailing_punct(match.group(0)))
    return urls


def _strip_trailing_punct(url: str) -> str:
    return url.rstrip(_URL_TRAILING_PUNCT)


class UrlPreservationScorer:
    """Reject when any URL from the original is missing from the candidate."""

    name = "url_preservation"

    def score(self, original: str, candidate: str) -> ScoreResult:
        original_urls = extract_urls(original)
        for url in original_urls:
            if url not in candidate:
                return ScoreResult(
                    name=self.name,
                    value=0.0,
                    verdict="reject",
                    rejection_reason=f"url not preserved: {url!r}",
                    span=url,
                )
        return ScoreResult(name=self.name, value=1.0, verdict="accept")


__all__ = [
    "EntityExtractor",
    "EntityPreservationScorer",
    "NumberPreservationScorer",
    "UrlPreservationScorer",
    "build_spacy_entity_extractor",
    "extract_number_triples",
    "extract_urls",
]
