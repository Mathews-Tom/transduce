"""Unit tests for the static mode registry and seed modes (P1-REG-01, P1-REG-03)."""

from __future__ import annotations

import pytest
from jinja2 import Environment, StrictUndefined

from transduce.registry.seed_modes import seed_modes
from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)
from transduce.registry.static import (
    ModeNotFoundError,
    StaticRegistry,
    build_default_registry,
)

pytestmark = pytest.mark.unit


def _make_spec(mode_id: str, *, prompt: str = "{{ input }}") -> ModeSpec:
    return ModeSpec(
        id=mode_id,
        version="0.1.0",
        description="x",
        prompt_template=prompt,
        backend_requirements=BackendRequirements(min_model_b=1.0),
        verifier_profile=VerifierProfile(),
    )


def test_registry_loads_three_seed_modes_by_id() -> None:
    registry = build_default_registry()

    ids = {spec.id for spec in registry.list_modes()}

    assert ids == {"dejargon", "register.casual", "length.normalize"}


def test_registry_unknown_mode_id_raises_mode_not_found() -> None:
    registry = build_default_registry()

    with pytest.raises(ModeNotFoundError):
        registry.resolve("does.not.exist")


def test_registry_resolve_returns_seed_spec() -> None:
    registry = build_default_registry()

    spec = registry.resolve("dejargon")

    assert spec.id == "dejargon"
    assert spec.backend_requirements.min_model_b == pytest.approx(14.0)


def test_registry_in_operator_reports_membership() -> None:
    registry = build_default_registry()

    assert "dejargon" in registry
    assert "missing" not in registry


def test_registry_rejects_duplicate_id_version_pair() -> None:
    with pytest.raises(ValueError, match="duplicate mode entry"):
        StaticRegistry([_make_spec("dup"), _make_spec("dup")])


def test_registry_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StaticRegistry([])


def test_registry_invalid_prompt_template_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="prompt_template failed to compile"):
        StaticRegistry([_make_spec("bad", prompt="{{ unterminated ")])


def test_seed_mode_dejargon_renders_prompt_with_input_and_intensity() -> None:
    spec = build_default_registry().resolve("dejargon")
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(  # noqa: S701
        spec.prompt_template
    )

    rendered = template.render(input="We synergize.", intensity=0.6, preserve=[])

    assert "We synergize." in rendered
    assert "0.6" in rendered


def test_seed_mode_register_casual_default_threshold_is_documented() -> None:
    spec = build_default_registry().resolve("register.casual")

    assert spec.verifier_profile.cosine_min == pytest.approx(0.82)


def test_seed_mode_length_normalize_default_threshold_is_documented() -> None:
    spec = build_default_registry().resolve("length.normalize")

    assert spec.verifier_profile.cosine_min == pytest.approx(0.78)


def test_seed_mode_dejargon_preserve_defaults_cover_entities_numbers_urls() -> None:
    spec = build_default_registry().resolve("dejargon")

    assert set(spec.preserve_defaults) == {
        PreserveRule.ENTITIES,
        PreserveRule.NUMBERS,
        PreserveRule.URLS,
    }


def test_seed_modes_all_declare_english_support() -> None:
    for spec in seed_modes():
        assert "en" in spec.supported_languages


def test_seed_mode_dejargon_min_model_floor_is_14b() -> None:
    spec = build_default_registry().resolve("dejargon")

    assert spec.backend_requirements.min_model_b == pytest.approx(14.0)
