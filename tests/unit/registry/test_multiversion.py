"""Unit tests for multi-version registry dispatch (P3-VER-01..03)."""

from __future__ import annotations

import pytest

from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)
from transduce.registry.static import (
    ModeNotFoundError,
    ModeVersionNotFoundError,
    StaticRegistry,
)

pytestmark = pytest.mark.unit


def _spec(mode_id: str, version: str) -> ModeSpec:
    return ModeSpec(
        id=mode_id,
        version=version,
        description=f"{mode_id} v{version}",
        prompt_template="rewrite {{ input }} intensity {{ intensity }}",
        preserve_defaults=(PreserveRule.ENTITIES,),
        backend_requirements=BackendRequirements(min_model_b=1.0),
        verifier_profile=VerifierProfile(),
    )


def test_registry_loads_two_versions_of_same_mode_simultaneously() -> None:
    registry = StaticRegistry(
        [
            _spec("humanize", "1.0.0"),
            _spec("humanize", "2.0.0"),
        ]
    )

    listed = {(spec.id, spec.version) for spec in registry.list_modes()}

    assert listed == {("humanize", "1.0.0"), ("humanize", "2.0.0")}


def test_registry_resolve_with_explicit_version_returns_that_revision() -> None:
    registry = StaticRegistry(
        [
            _spec("humanize", "1.0.0"),
            _spec("humanize", "2.0.0"),
        ]
    )

    spec = registry.resolve("humanize@1.0.0")

    assert spec.version == "1.0.0"


def test_registry_resolve_without_version_returns_highest_semver() -> None:
    registry = StaticRegistry(
        [
            _spec("humanize", "1.0.0"),
            _spec("humanize", "2.1.3"),
            _spec("humanize", "2.0.0"),
        ]
    )

    spec = registry.resolve("humanize")

    assert spec.version == "2.1.3"


def test_registry_resolve_unknown_id_raises_mode_not_found() -> None:
    registry = StaticRegistry([_spec("humanize", "1.0.0")])

    with pytest.raises(ModeNotFoundError, match="bogus"):
        registry.resolve("bogus")


def test_registry_resolve_known_id_unknown_version_raises_mode_version_not_found() -> None:
    registry = StaticRegistry(
        [
            _spec("humanize", "1.0.0"),
            _spec("humanize", "2.0.0"),
        ]
    )

    with pytest.raises(ModeVersionNotFoundError) as exc_info:
        registry.resolve("humanize@9.9.9")

    assert exc_info.value.mode_id == "humanize"
    assert exc_info.value.requested == "9.9.9"
    assert set(exc_info.value.available) == {"1.0.0", "2.0.0"}


def test_registry_rejects_duplicate_id_version_pair() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        StaticRegistry(
            [
                _spec("humanize", "1.0.0"),
                _spec("humanize", "1.0.0"),
            ]
        )


def test_registry_rejects_non_semver_version() -> None:
    with pytest.raises(ValueError, match="non-semver"):
        StaticRegistry([_spec("humanize", "not-a-version")])


def test_registry_resolve_empty_reference_raises_mode_not_found() -> None:
    registry = StaticRegistry([_spec("humanize", "1.0.0")])

    with pytest.raises(ModeNotFoundError, match="empty"):
        registry.resolve("")


def test_registry_resolve_invalid_separator_form_raises_mode_not_found() -> None:
    registry = StaticRegistry([_spec("humanize", "1.0.0")])

    with pytest.raises(ModeNotFoundError, match="invalid"):
        registry.resolve("@1.0.0")
    with pytest.raises(ModeNotFoundError, match="invalid"):
        registry.resolve("humanize@")


def test_registry_contains_recognises_version_and_bare_lookups() -> None:
    registry = StaticRegistry(
        [
            _spec("humanize", "1.0.0"),
            _spec("humanize", "2.0.0"),
        ]
    )

    assert "humanize" in registry
    assert "humanize@1.0.0" in registry
    assert "humanize@9.9.9" not in registry
    assert "bogus" not in registry
