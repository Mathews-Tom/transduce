"""Unit tests for the allowlist registry (P2-PLG-01..P2-PLG-03)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from transduce.registry.allowlist import (
    AllowlistRegistry,
    HashMismatchError,
    UnpinnedPackageError,
    compute_package_sha256,
    load_allowlisted_modes,
    verify_pin,
    warn_auto_discovery_enabled,
)
from transduce.registry.manifest import load_manifest

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "manifest_modes"


def test_allowlist_loads_pinned_package() -> None:
    package_root = _FIXTURE_DIR / "formal-to-warm"
    sha = compute_package_sha256(package_root)

    specs = load_allowlisted_modes(package_paths_with_pins=[(package_root, sha)])

    assert len(specs) == 1
    assert specs[0].id == "formal-to-warm"


def test_allowlist_rejects_unpinned_package() -> None:
    package_root = _FIXTURE_DIR / "formal-to-warm"

    with pytest.raises(UnpinnedPackageError, match="missing a sha256 pin"):
        load_allowlisted_modes(package_paths_with_pins=[(package_root, "")])


def test_allowlist_rejects_hash_mismatch() -> None:
    package_root = _FIXTURE_DIR / "formal-to-warm"
    bad_sha = "0" * 64

    with pytest.raises(HashMismatchError, match="sha256 mismatch"):
        load_allowlisted_modes(package_paths_with_pins=[(package_root, bad_sha)])


def test_allowlist_default_refuses_auto_discovery(caplog: pytest.LogCaptureFixture) -> None:
    """``warn_auto_discovery_enabled`` warns when the operator opts into auto."""
    caplog.set_level(logging.WARNING, logger="transduce.registry.allowlist")

    warn_auto_discovery_enabled("auto")

    assert any("auto-discovery enabled" in record.message for record in caplog.records)


def test_allowlist_default_silent_on_allowlist_source(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="transduce.registry.allowlist")

    warn_auto_discovery_enabled("allowlist")

    assert not any("auto-discovery" in record.message for record in caplog.records)


def test_compute_package_sha256_deterministic_on_directory_tree() -> None:
    package_root = _FIXTURE_DIR / "formal-to-warm"

    first = compute_package_sha256(package_root)
    second = compute_package_sha256(package_root)

    assert first == second
    assert len(first) == 64


def test_compute_package_sha256_changes_when_file_modified(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.toml").write_text("a = 1\n", encoding="utf-8")
    initial = compute_package_sha256(package)

    (package / "a.toml").write_text("a = 2\n", encoding="utf-8")
    after = compute_package_sha256(package)

    assert initial != after


def test_verify_pin_passes_on_match() -> None:
    package_root = _FIXTURE_DIR / "formal-to-warm"
    sha = compute_package_sha256(package_root)

    verify_pin(package_root, sha)


def test_allowlist_registry_resolve_round_trips() -> None:
    spec = load_manifest(_FIXTURE_DIR / "formal-to-warm")
    registry = AllowlistRegistry([spec])

    assert registry.resolve("formal-to-warm") is spec
    assert "formal-to-warm" in registry
    assert registry.list_modes() == (spec,)


def test_allowlist_registry_rejects_duplicate_ids() -> None:
    spec = load_manifest(_FIXTURE_DIR / "formal-to-warm")

    with pytest.raises(ValueError, match="duplicate mode id"):
        AllowlistRegistry([spec, spec])


def test_allowlist_registry_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one mode"):
        AllowlistRegistry([])
