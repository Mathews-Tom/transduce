"""Pydantic schema for the v0.5 service configuration.

Mirrors the v0.5 subset of docs/system-design.md §Configuration. The
release adds the ``modes`` allowlist + sha256 section
(P2-PLG-01..P2-PLG-03); v1 extends ``observability`` (OTel GenAI SemConv,
P3-OBS-01..P3-OBS-05) plus ``language.detector`` (fasttext, P3-LANG-01).

Each section sets ``extra="forbid"`` so typos surface as validation
errors at startup rather than silently degrading behaviour.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServiceConfig(BaseModel):
    """Top-level service runtime parameters."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(
        default="0.0.0.0",  # noqa: S104  # nosec B104 — server bind default per docs/system-design.md §Configuration
        min_length=1,
    )
    port: int = Field(default=8080, ge=1, le=65_535)
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    max_input_chars: int = Field(default=50_000, gt=0)
    max_retries_default: int = Field(default=3, ge=0, le=5)


class BackendEntry(BaseModel):
    """One backend registry entry."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    concurrency_limit: int = Field(default=1, ge=1)


class BackendsConfig(BaseModel):
    """Backends section: a default id and a non-empty registry."""

    model_config = ConfigDict(extra="forbid")

    default: str = Field(min_length=1)
    registry: list[BackendEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_must_be_registered(self) -> BackendsConfig:
        registered = {entry.id for entry in self.registry}
        if self.default not in registered:
            raise ValueError(
                f"backends.default {self.default!r} is not present in backends.registry "
                f"(known: {sorted(registered)})"
            )
        return self


class VerificationConfig(BaseModel):
    """Verifier subsystem defaults for the v0 cosine + preservation pipeline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    default_cosine_min: float = Field(default=0.85, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0, le=5)


class LanguageConfig(BaseModel):
    """Language defaults. The v1 release introduces ``detector`` selection (P3-LANG-01)."""

    model_config = ConfigDict(extra="forbid")

    default: str = Field(default="en", min_length=1)


class ModePackageEntry(BaseModel):
    """One pinned mode package in the allowlist (P2-PLG-01)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    path: str = Field(min_length=1, description="Filesystem path to the package root.")
    signed_by: str | None = Field(default=None, min_length=1)


class ModesConfig(BaseModel):
    """Mode-loader configuration (P2-PLG-01..P2-PLG-03).

    ``source`` defaults to ``allowlist``. Setting it to ``auto`` enables
    legacy entry-point discovery and emits a startup warning per the
    dev plan; production deployments must keep it on ``allowlist``.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["allowlist", "auto"] = "allowlist"
    enforce_signing: bool = False
    packages: list[ModePackageEntry] = Field(default_factory=list)


class Config(BaseModel):
    """Full v0.5 service configuration."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    modes: ModesConfig = Field(default_factory=ModesConfig)
    backends: BackendsConfig
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
