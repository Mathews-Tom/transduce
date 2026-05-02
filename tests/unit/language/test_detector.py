"""Unit tests for the lingua-backed language detector (P3-LANG-01..04)."""

from __future__ import annotations

import pytest

from transduce.language.detector import (
    LanguageDetectionError,
    LanguageDetector,
    LanguageNotSupportedError,
)

pytestmark = pytest.mark.unit


def test_detector_single_language_returns_default_for_any_input() -> None:
    detector = LanguageDetector(languages=("en",), default="en")

    assert detector.detect("Hello world") == "en"
    assert detector.detect("") == "en"
    assert detector.detect("   ") == "en"


def test_detector_multilingual_picks_correct_language_for_clear_input() -> None:
    detector = LanguageDetector(languages=("en", "de", "fr"), default="en", min_confidence=0.5)

    assert detector.detect("This is a long enough sentence in English to disambiguate.") == "en"


def test_detector_multilingual_picks_german_for_clear_german() -> None:
    detector = LanguageDetector(languages=("en", "de", "fr"), default="en", min_confidence=0.5)

    assert (
        detector.detect("Dies ist ein deutscher Satz mit ausreichend Wörtern zur Unterscheidung.")
        == "de"
    )


def test_detector_falls_back_to_default_when_confidence_below_threshold() -> None:
    detector = LanguageDetector(languages=("en", "de"), default="en", min_confidence=0.99)

    # Very short ambiguous input cannot clear a 0.99 confidence floor.
    assert detector.detect("ok") == "en"


def test_detector_returns_default_for_empty_input() -> None:
    detector = LanguageDetector(languages=("en", "de"), default="en")

    assert detector.detect("") == "en"
    assert detector.detect("\n\t  ") == "en"


def test_detector_construction_rejects_default_outside_loaded_set() -> None:
    with pytest.raises(ValueError, match="must be in the loaded languages"):
        LanguageDetector(languages=("en", "de"), default="fr")


def test_detector_construction_rejects_unknown_iso_code() -> None:
    with pytest.raises(LanguageDetectionError, match="unknown ISO 639-1 code"):
        LanguageDetector(languages=("en", "zz"), default="en")


def test_detector_construction_rejects_invalid_min_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        LanguageDetector(languages=("en", "de"), default="en", min_confidence=1.5)


def test_detector_construction_rejects_empty_language_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LanguageDetector(languages=(), default="en")


def test_detector_loaded_property_returns_unique_codes() -> None:
    detector = LanguageDetector(languages=("en", "de", "en"), default="en")

    assert detector.loaded == ("en", "de")


def test_language_not_supported_error_carries_diagnostic_fields() -> None:
    err = LanguageNotSupportedError(detected="de", supported=("en", "fr"), mode_id="dejargon")

    assert err.detected == "de"
    assert err.supported == ("en", "fr")
    assert err.mode_id == "dejargon"
    assert "dejargon" in str(err)
    assert "de" in str(err)
