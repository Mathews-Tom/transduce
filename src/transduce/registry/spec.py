"""Mode specification models per docs/system-design.md §Mode Registry.

The v0 subset captures the surface required by P1-REG-02 and is consumed
by the API schemas, registry loader, pipeline orchestrator, and
verification pipeline. The v0.5 release extends ``VerifierProfile`` with
NLI and HHEM thresholds (P2-VER-07); v1 widens ``supported_languages``
semantics (P3-LANG-02..P3-LANG-04).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreserveRule(StrEnum):
    """Preservation categories surfaced in mode specs and transform requests."""

    ENTITIES = "entities"
    NUMBERS = "numbers"
    URLS = "urls"


class BackendRequirements(BaseModel):
    """Minimum backend capability requirements for a mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_model_b: float = Field(
        ge=0.0,
        description="Minimum backend model size in billions of parameters.",
    )


class VerifierProfile(BaseModel):
    """Per-mode verifier thresholds (v0 subset: cosine + preservation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cosine_min: float = Field(default=0.85, ge=0.0, le=1.0)
    preserve_entities: bool = True
    preserve_numbers: bool = True
    preserve_urls: bool = True


class ModeSpec(BaseModel):
    """Declarative specification for a transformation mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    prompt_template: str = Field(
        min_length=1,
        description="Jinja2 template string rendered with input/intensity/preserve.",
    )
    intensity_range: tuple[float, float] = (0.0, 1.0)
    preserve_defaults: tuple[PreserveRule, ...] = ()
    backend_requirements: BackendRequirements
    verifier_profile: VerifierProfile
    supported_languages: tuple[str, ...] = ("en",)

    @model_validator(mode="after")
    def _validate_intensity_range(self) -> ModeSpec:
        low, high = self.intensity_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"intensity_range must be 0.0 <= low <= high <= 1.0, got ({low}, {high})"
            )
        return self
