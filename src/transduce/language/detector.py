"""Ingress language detection (P3-LANG-01..04).

Wraps lingua-language-detector with the operator-configured language
set. The detector is cached and reused across requests; first-call
latency includes lingua's model load (cold ~10 ms, warm <2 ms per
docs/system-design.md §Language Detection).

Detection returns an ISO 639-1 code (``en``, ``de``, ``fr``, …). If
the most-confident match is below the operator-set ``min_confidence``
floor, the detector falls back to the configured default language so
the pipeline neither hard-fails nor silently mis-routes for ambiguous
short inputs. When the operator loads only one language, the detector
short-circuits to that language without invoking lingua — lingua
requires at least two languages to disambiguate, and a single-language
deployment has nothing to disambiguate.
"""

from __future__ import annotations

from collections.abc import Iterable

from lingua import IsoCode639_1, Language
from lingua import LanguageDetector as _LinguaDetector
from lingua import LanguageDetectorBuilder


class LanguageDetectionError(RuntimeError):
    """Raised when an ISO 639-1 code does not map to a lingua language."""


class LanguageNotSupportedError(RuntimeError):
    """Raised when the detected language is outside the mode's supported set (P3-LANG-03)."""

    def __init__(
        self,
        *,
        detected: str,
        supported: tuple[str, ...],
        mode_id: str,
    ) -> None:
        super().__init__(
            f"mode {mode_id!r} does not support language {detected!r} "
            f"(supports: {sorted(supported)})"
        )
        self.detected = detected
        self.supported = supported
        self.mode_id = mode_id


class LanguageDetector:
    """Operator-configured language detector backed by lingua."""

    def __init__(
        self,
        *,
        languages: Iterable[str],
        default: str,
        min_confidence: float = 0.6,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0.0, 1.0], got {min_confidence}")
        codes = tuple(dict.fromkeys(languages))
        if not codes:
            raise ValueError("LanguageDetector requires at least one language code")
        if default not in codes:
            raise ValueError(f"default {default!r} must be in the loaded languages: {list(codes)}")
        # Validate every code maps to a lingua language even when the
        # detector short-circuits, so misconfigured codes fail at startup.
        for code in codes:
            _to_lingua_language(code)
        self._codes = codes
        self._default = default
        self._min_confidence = min_confidence
        self._detector = _build_detector(codes) if len(codes) >= 2 else None

    @property
    def default(self) -> str:
        return self._default

    @property
    def loaded(self) -> tuple[str, ...]:
        return self._codes

    def detect(self, text: str) -> str:
        """Return the ISO 639-1 code for ``text``.

        Falls back to ``default`` when no candidate clears
        ``min_confidence`` or the input is empty/whitespace. With a
        single loaded language the detector returns that language for
        every non-empty input.
        """
        if not text.strip():
            return self._default
        if self._detector is None:
            return self._default
        confidences = self._detector.compute_language_confidence_values(text)
        if not confidences:
            return self._default
        top = confidences[0]
        if top.value < self._min_confidence:
            return self._default
        return _to_iso_code(top.language)


def _build_detector(codes: tuple[str, ...]) -> _LinguaDetector:
    languages = [_to_lingua_language(code) for code in codes]
    return LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()


def _to_lingua_language(iso_code: str) -> Language:
    try:
        iso_enum = getattr(IsoCode639_1, iso_code.upper())
    except AttributeError as exc:
        raise LanguageDetectionError(
            f"unknown ISO 639-1 code {iso_code!r} (lingua does not recognise it)"
        ) from exc
    if not isinstance(iso_enum, IsoCode639_1):
        raise LanguageDetectionError(
            f"unknown ISO 639-1 code {iso_code!r} (lingua does not recognise it)"
        )
    return Language.from_iso_code_639_1(iso_enum)


def _to_iso_code(language: Language) -> str:
    return language.iso_code_639_1.name.lower()


__all__ = [
    "LanguageDetectionError",
    "LanguageDetector",
    "LanguageNotSupportedError",
]
