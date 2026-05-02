"""Allowlist mode loader with sha256 pinning (P2-PLG-01..P2-PLG-03).

Operators declare permitted mode packages in ``transduce.yaml`` under
``modes.packages`` with name, version, sha256, and an optional
``signed_by`` identity. The loader verifies each pin before any code
runs and refuses to load packages whose computed sha256 does not match
the pin.

Auto-discovery via ``transduce.modes`` entry points is **disabled by
default**. Operators who need entry-point loading must set
``modes.source: auto`` in config; the loader emits a startup warning
when this is enabled. Manifest-only packages reference a ``manifest_dir``
inside the package contents; Python-plugin packages are out of scope
for this commit and land in the sandbox commit (P2-PLG-05).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from transduce.registry.manifest import load_manifest, load_manifests_from_directory
from transduce.registry.spec import ModeSpec

_LOG = logging.getLogger(__name__)

_HASH_BUFFER_SIZE: int = 64 * 1024


class HashMismatchError(RuntimeError):
    """Raised when a package's computed sha256 does not match the pin."""


class UnpinnedPackageError(RuntimeError):
    """Raised when a package entry is missing required pin metadata."""


def compute_package_sha256(path: Path) -> str:
    """Compute sha256 of a file or directory tree (deterministic over directory listings)."""
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_BUFFER_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"package path not found: {path}")
    for file_path in sorted(_iter_files(path)):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_BUFFER_SIZE), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not _is_ignored(path):
            yield path


def _is_ignored(path: Path) -> bool:
    parts = path.parts
    if any(part == "__pycache__" for part in parts):
        return True
    return path.suffix in {".pyc", ".pyo"}


def verify_pin(package_path: Path, expected_sha256: str) -> None:
    """Compute the sha256 of ``package_path`` and raise if it differs from ``expected_sha256``."""
    if not expected_sha256:
        raise UnpinnedPackageError(
            f"package {package_path} is missing a sha256 pin in transduce.yaml"
        )
    actual = compute_package_sha256(package_path)
    if actual != expected_sha256:
        raise HashMismatchError(
            f"sha256 mismatch for {package_path}: pin {expected_sha256} vs computed {actual}"
        )


class AllowlistRegistry:
    """Materialise modes from sha256-pinned manifest packages."""

    def __init__(self, modes: Iterable[ModeSpec]) -> None:
        modes_tuple = tuple(modes)
        if not modes_tuple:
            raise ValueError("AllowlistRegistry requires at least one mode spec")
        seen: dict[str, ModeSpec] = {}
        for spec in modes_tuple:
            if spec.id in seen:
                raise ValueError(f"duplicate mode id in allowlist: {spec.id!r}")
            seen[spec.id] = spec
        self._modes = seen

    def resolve(self, mode_id: str) -> ModeSpec:
        try:
            return self._modes[mode_id]
        except KeyError as exc:
            raise ModeNotFoundError(f"unknown mode: {mode_id!r}") from exc

    def list_modes(self) -> tuple[ModeSpec, ...]:
        return tuple(self._modes.values())

    def __contains__(self, mode_id: object) -> bool:
        return mode_id in self._modes


class ModeNotFoundError(LookupError):
    """Raised when ``resolve`` is asked for an unknown mode id."""


def load_allowlisted_modes(
    *,
    package_paths_with_pins: Iterable[tuple[Path, str]],
) -> list[ModeSpec]:
    """Verify each package's sha256 pin and load every manifest under it.

    Each tuple is ``(package_root, expected_sha256)``. The package root
    must contain a manifest tree: either a ``mode.toml`` directly (single
    mode) or one-level-deep subdirectories that each contain a
    ``mode.toml``. Hash verification runs before manifests are read so
    tampered packages never reach the loader.
    """
    specs: list[ModeSpec] = []
    for package_root, expected_sha in package_paths_with_pins:
        verify_pin(package_root, expected_sha)
        if (package_root / "mode.toml").exists():
            specs.append(load_manifest(package_root))
            continue
        specs.extend(load_manifests_from_directory(package_root))
    return specs


def warn_auto_discovery_enabled(source: str) -> None:
    """Emit a single startup warning when ``modes.source`` is not ``allowlist``."""
    if source != "allowlist":
        _LOG.warning(
            "modes.source=%s — auto-discovery enabled; production deployments "
            "should pin packages explicitly per docs/system-design.md "
            "§Mode Registry",
            source,
        )


__all__ = [
    "AllowlistRegistry",
    "HashMismatchError",
    "ModeNotFoundError",
    "UnpinnedPackageError",
    "compute_package_sha256",
    "load_allowlisted_modes",
    "verify_pin",
    "warn_auto_discovery_enabled",
]
