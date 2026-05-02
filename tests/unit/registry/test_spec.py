"""Unit tests for the mode specification models (P1-REG-02)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)

pytestmark = pytest.mark.unit


def _spec(**overrides: object) -> ModeSpec:
    base: dict[str, object] = {
        "id": "dejargon",
        "version": "0.1.0",
        "description": "Reduce jargon density.",
        "prompt_template": "Rewrite: {{ input }}",
        "backend_requirements": BackendRequirements(min_model_b=7.0),
        "verifier_profile": VerifierProfile(),
    }
    base.update(overrides)
    return ModeSpec(**base)  # type: ignore[arg-type]


def test_mode_spec_default_intensity_range_is_zero_to_one() -> None:
    spec = _spec()

    assert spec.intensity_range == (0.0, 1.0)


def test_mode_spec_intensity_range_inverted_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(intensity_range=(0.8, 0.2))


def test_mode_spec_intensity_range_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(intensity_range=(0.0, 1.2))


def test_mode_spec_default_supported_languages_is_english() -> None:
    spec = _spec()

    assert spec.supported_languages == ("en",)


def test_mode_spec_preserve_defaults_round_trip() -> None:
    spec = _spec(preserve_defaults=(PreserveRule.ENTITIES, PreserveRule.URLS))

    assert spec.preserve_defaults == (PreserveRule.ENTITIES, PreserveRule.URLS)


def test_mode_spec_empty_id_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(id="")


def test_mode_spec_empty_prompt_template_rejected() -> None:
    with pytest.raises(ValidationError):
        _spec(prompt_template="")


def test_mode_spec_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ModeSpec.model_validate(
            {
                "id": "dejargon",
                "version": "0.1.0",
                "description": "x",
                "prompt_template": "{{ input }}",
                "backend_requirements": {"min_model_b": 7.0},
                "verifier_profile": {},
                "unknown_field": True,
            }
        )


def test_mode_spec_is_hashable_when_frozen() -> None:
    spec = _spec()

    assert isinstance(hash(spec), int)
    assert hash(spec) == hash(_spec())


def test_verifier_profile_defaults_match_dev_plan() -> None:
    profile = VerifierProfile()

    assert profile.cosine_min == pytest.approx(0.85)
    assert profile.preserve_entities is True
    assert profile.preserve_numbers is True
    assert profile.preserve_urls is True


def test_verifier_profile_cosine_min_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        VerifierProfile(cosine_min=1.5)


def test_backend_requirements_negative_size_rejected() -> None:
    with pytest.raises(ValidationError):
        BackendRequirements(min_model_b=-1.0)
