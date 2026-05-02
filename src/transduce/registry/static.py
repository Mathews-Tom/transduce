"""Static mode registry (P1-REG-01).

Loads the in-tree :func:`seed_modes` and exposes ``resolve``/``list_modes``
without any plugin discovery. Phase 2 swaps this for an allowlist loader
with sha256 pinning; the call sites in the API and pipeline depend on
the same surface either way.
"""

from __future__ import annotations

from collections.abc import Iterable

from jinja2 import Environment, StrictUndefined, TemplateError

from transduce.registry.seed_modes import seed_modes
from transduce.registry.spec import ModeSpec


class ModeNotFoundError(LookupError):
    """Raised when ``resolve`` is asked for a mode id the registry does not carry."""


class StaticRegistry:
    """Hold a static set of ``ModeSpec`` instances keyed by id."""

    def __init__(self, modes: Iterable[ModeSpec]) -> None:
        modes_tuple = tuple(modes)
        if not modes_tuple:
            raise ValueError("StaticRegistry requires at least one mode spec")
        seen: dict[str, ModeSpec] = {}
        for spec in modes_tuple:
            if spec.id in seen:
                raise ValueError(f"duplicate mode id in registry: {spec.id!r}")
            seen[spec.id] = spec
            _validate_prompt_template(spec)
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


def build_default_registry() -> StaticRegistry:
    """Build the registry with the three v0 in-tree seed modes."""
    return StaticRegistry(seed_modes())


def _validate_prompt_template(spec: ModeSpec) -> None:
    """Compile the Jinja template at registry construction so syntax errors fail fast."""
    environment = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701 — prompts feed an LLM, not HTML
    try:
        environment.from_string(spec.prompt_template)
    except TemplateError as exc:
        raise ValueError(f"mode {spec.id!r} prompt_template failed to compile: {exc}") from exc
