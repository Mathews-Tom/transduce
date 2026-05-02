"""Static mode registry with multi-version dispatch (P1-REG-01, P3-VER-01..03).

Loads the in-tree :func:`seed_modes` and exposes
``resolve``/``list_modes`` without any plugin discovery. Multi-version
dispatch lets two revisions of the same mode (``humanize@1.0.0`` and
``humanize@2.0.0``) live in the registry simultaneously; clients pin
explicitly via ``mode: "id@version"`` and bare ``mode: "id"`` lookups
resolve to the highest-SemVer revision (P3-VER-02).

The v0.5 release swapped the constructor input from a flat list to a
multi-version-friendly mapping; the call sites in the API and pipeline
depend on the same ``resolve``/``list_modes`` surface either way.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from jinja2 import Environment, StrictUndefined, TemplateError
from packaging.version import InvalidVersion, Version

from transduce.registry.seed_modes import seed_modes
from transduce.registry.spec import ModeSpec

_VERSION_SEPARATOR: Final[str] = "@"


class ModeNotFoundError(LookupError):
    """Raised when ``resolve`` is asked for a mode id the registry does not carry."""


class ModeVersionNotFoundError(LookupError):
    """Raised when the mode id exists but the requested version does not (P3-VER-03)."""

    def __init__(self, *, mode_id: str, requested: str, available: tuple[str, ...]) -> None:
        super().__init__(
            f"mode {mode_id!r} has no version {requested!r} (available: {sorted(available)})"
        )
        self.mode_id = mode_id
        self.requested = requested
        self.available = available


class StaticRegistry:
    """Hold a static set of ``ModeSpec`` instances keyed by ``(id, version)``.

    Multiple specs may share an ``id`` so long as their ``version``
    differs. ``resolve`` accepts either a bare id (returns the highest
    SemVer revision) or an ``id@version`` string (returns that exact
    revision or raises :class:`ModeVersionNotFoundError`).
    """

    def __init__(self, modes: Iterable[ModeSpec]) -> None:
        modes_tuple = tuple(modes)
        if not modes_tuple:
            raise ValueError("StaticRegistry requires at least one mode spec")
        by_pair: dict[tuple[str, str], ModeSpec] = {}
        for spec in modes_tuple:
            key = (spec.id, spec.version)
            if key in by_pair:
                raise ValueError(f"duplicate mode entry for {spec.id!r}@{spec.version!r}")
            by_pair[key] = spec
            _validate_prompt_template(spec)
        self._by_pair = by_pair
        self._latest_by_id = _index_latest(by_pair.values())

    def resolve(self, mode_ref: str) -> ModeSpec:
        mode_id, requested_version = _split_ref(mode_ref)
        if requested_version is None:
            spec = self._latest_by_id.get(mode_id)
            if spec is None:
                raise ModeNotFoundError(f"unknown mode: {mode_id!r}")
            return spec
        if mode_id not in self._latest_by_id:
            raise ModeNotFoundError(f"unknown mode: {mode_id!r}")
        spec = self._by_pair.get((mode_id, requested_version))
        if spec is None:
            available = tuple(version for (mid, version) in self._by_pair if mid == mode_id)
            raise ModeVersionNotFoundError(
                mode_id=mode_id,
                requested=requested_version,
                available=available,
            )
        return spec

    def list_modes(self) -> tuple[ModeSpec, ...]:
        return tuple(self._by_pair.values())

    def __contains__(self, mode_ref: object) -> bool:
        if not isinstance(mode_ref, str):
            return False
        try:
            self.resolve(mode_ref)
        except (ModeNotFoundError, ModeVersionNotFoundError):
            return False
        return True


def build_default_registry() -> StaticRegistry:
    """Build the registry with the v0 in-tree seed modes."""
    return StaticRegistry(seed_modes())


def _split_ref(mode_ref: str) -> tuple[str, str | None]:
    """Split a ``mode_id`` or ``mode_id@version`` reference."""
    if not mode_ref:
        raise ModeNotFoundError("empty mode reference")
    if _VERSION_SEPARATOR not in mode_ref:
        return mode_ref, None
    mode_id, _, version = mode_ref.partition(_VERSION_SEPARATOR)
    if not mode_id or not version:
        raise ModeNotFoundError(f"invalid mode reference: {mode_ref!r}")
    return mode_id, version


def _index_latest(specs: Iterable[ModeSpec]) -> dict[str, ModeSpec]:
    """Group ``specs`` by id and select the highest-SemVer per id."""
    grouped: dict[str, list[ModeSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.id, []).append(spec)
    latest: dict[str, ModeSpec] = {}
    for mode_id, candidates in grouped.items():
        latest[mode_id] = max(candidates, key=_sort_key_for)
    return latest


def _sort_key_for(spec: ModeSpec) -> Version:
    """Return a comparable :class:`Version` for ``spec``; reject malformed versions loudly."""
    try:
        return Version(spec.version)
    except InvalidVersion as exc:
        raise ValueError(
            f"mode {spec.id!r} has non-semver version {spec.version!r}; "
            "the multi-version dispatcher needs PEP 440-compatible versions"
        ) from exc


def _validate_prompt_template(spec: ModeSpec) -> None:
    """Compile the Jinja template at registry construction so syntax errors fail fast."""
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,  # noqa: S701  # nosec B701 — prompts feed an LLM, not HTML
    )
    try:
        environment.from_string(spec.prompt_template)
    except TemplateError as exc:
        raise ValueError(f"mode {spec.id!r} prompt_template failed to compile: {exc}") from exc
