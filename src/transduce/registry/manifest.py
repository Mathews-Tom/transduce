"""Manifest-only mode loader (P2-PLG-04).

A mode is a directory containing ``mode.toml`` and one or more Jinja
templates referenced by relative path. The manifest is parsed via
``tomllib`` (stdlib in Python 3.11+) and projected onto a ``ModeSpec``.
No Python is executed at registry-load time per ADR-0002.

Manifest schema:

    [mode]
    id = "formal-to-warm"
    version = "1.0.0"
    description = "Soften formal correspondence while preserving facts."
    prompt_template = "prompts/formal-to-warm.j2"
    intensity_range = [0.0, 1.0]
    preserve_defaults = ["entities", "numbers", "urls"]
    supported_languages = ["en"]

    [mode.backend_requirements]
    min_model_b = 7.0

    [mode.verifier_profile]
    cosine_min = 0.85
    nli_min = 0.70

The Jinja template is compiled at load time so syntax errors fail
fast, before any inference.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateError

from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)


class ManifestError(RuntimeError):
    """Raised when a manifest fails to parse, validate, or compile."""


def load_manifest(directory: Path) -> ModeSpec:
    """Load a single ``mode.toml`` from ``directory`` and return a ``ModeSpec``."""
    manifest_path = directory / "mode.toml"
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")
    with manifest_path.open("rb") as handle:
        try:
            payload = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ManifestError(f"invalid TOML in {manifest_path}: {exc}") from exc

    mode_section = payload.get("mode")
    if not isinstance(mode_section, dict):
        raise ManifestError(f"{manifest_path} missing required [mode] section")

    template_rel = mode_section.get("prompt_template")
    if not isinstance(template_rel, str) or not template_rel:
        raise ManifestError(f"{manifest_path} [mode].prompt_template must be a non-empty string")
    template_path = (directory / template_rel).resolve()
    template_root = directory.resolve()
    if template_root not in template_path.parents and template_root != template_path.parent:
        raise ManifestError(f"{manifest_path} [mode].prompt_template must reside under {directory}")
    if not template_path.exists():
        raise ManifestError(f"prompt template not found: {template_path}")
    template_source = template_path.read_text(encoding="utf-8")
    _validate_template_compiles(template_source, manifest_path=manifest_path)

    backend_requirements = _build_backend_requirements(mode_section, manifest_path)
    verifier_profile = _build_verifier_profile(mode_section, manifest_path)
    preserve_defaults = _parse_preserve_rules(
        mode_section.get("preserve_defaults", []),
        manifest_path=manifest_path,
    )

    intensity_range = mode_section.get("intensity_range", [0.0, 1.0])
    if not (
        isinstance(intensity_range, list)
        and len(intensity_range) == 2
        and all(isinstance(value, (int, float)) for value in intensity_range)
    ):
        raise ManifestError(f"{manifest_path} [mode].intensity_range must be a [low, high] pair")

    supported = mode_section.get("supported_languages", ["en"])
    if not isinstance(supported, list) or not all(isinstance(item, str) for item in supported):
        raise ManifestError(f"{manifest_path} [mode].supported_languages must be a list of strings")

    try:
        return ModeSpec(
            id=str(mode_section.get("id", "")),
            version=str(mode_section.get("version", "")),
            description=str(mode_section.get("description", "")),
            prompt_template=template_source,
            intensity_range=(float(intensity_range[0]), float(intensity_range[1])),
            preserve_defaults=preserve_defaults,
            backend_requirements=backend_requirements,
            verifier_profile=verifier_profile,
            supported_languages=tuple(supported),
        )
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{manifest_path} failed schema validation: {exc}") from exc


def _build_backend_requirements(
    mode_section: dict[str, object], manifest_path: Path
) -> BackendRequirements:
    section = mode_section.get("backend_requirements", {})
    if not isinstance(section, dict):
        raise ManifestError(f"{manifest_path} [mode.backend_requirements] must be a table")
    min_model_b = section.get("min_model_b")
    if not isinstance(min_model_b, (int, float)):
        raise ManifestError(f"{manifest_path} [mode.backend_requirements].min_model_b is required")
    return BackendRequirements(min_model_b=float(min_model_b))


def _build_verifier_profile(
    mode_section: dict[str, object], manifest_path: Path
) -> VerifierProfile:
    section = mode_section.get("verifier_profile", {})
    if not isinstance(section, dict):
        raise ManifestError(f"{manifest_path} [mode.verifier_profile] must be a table")
    try:
        return VerifierProfile.model_validate(section)
    except ValueError as exc:
        raise ManifestError(
            f"{manifest_path} [mode.verifier_profile] failed validation: {exc}"
        ) from exc


def _parse_preserve_rules(raw: object, *, manifest_path: Path) -> tuple[PreserveRule, ...]:
    if not isinstance(raw, list):
        raise ManifestError(f"{manifest_path} [mode].preserve_defaults must be a list")
    rules: list[PreserveRule] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ManifestError(f"{manifest_path} preserve_defaults entries must be strings")
        try:
            rules.append(PreserveRule(entry))
        except ValueError as exc:
            raise ManifestError(f"{manifest_path} unknown preserve rule: {entry!r}") from exc
    return tuple(rules)


def _validate_template_compiles(source: str, *, manifest_path: Path) -> None:
    environment = Environment(
        undefined=StrictUndefined,
        autoescape=False,  # noqa: S701  # nosec B701 — prompts feed an LLM, not HTML
    )
    try:
        environment.from_string(source)
    except TemplateError as exc:
        raise ManifestError(f"{manifest_path} Jinja template failed to compile: {exc}") from exc


def load_manifests_from_directory(root: Path) -> list[ModeSpec]:
    """Load every ``mode.toml`` directory under ``root`` (one level deep)."""
    if not root.exists():
        raise ManifestError(f"manifest root not found: {root}")
    specs: list[ModeSpec] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "mode.toml").exists():
            continue
        specs.append(load_manifest(entry))
    return specs


__all__ = [
    "ManifestError",
    "load_manifest",
    "load_manifests_from_directory",
]
