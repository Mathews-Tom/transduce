"""Mode specification models per docs/system-design.md §Mode Registry.

The v0.5 release widens ``VerifierProfile`` with NLI and HHEM thresholds,
the negation-diff opt-out flag, and the length-band bounds (P2-VER-07).
``PreserveRule.DATES`` enables the date scorer per-request (P2-VER-04).
v1 widens ``supported_languages`` semantics (P3-LANG-02..P3-LANG-04).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PreserveRule(StrEnum):
    """Preservation categories surfaced in mode specs and transform requests."""

    ENTITIES = "entities"
    NUMBERS = "numbers"
    URLS = "urls"
    DATES = "dates"


class BackendRequirements(BaseModel):
    """Minimum backend capability requirements for a mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_model_b: float = Field(
        ge=0.0,
        description="Minimum backend model size in billions of parameters.",
    )


class VerifierProfile(BaseModel):
    """Per-mode verifier thresholds for the v0.5 ensemble (P2-VER-07).

    Defaults match docs/system-design.md §Verification Subsystem. Modes that
    need wider tolerances for length (e.g., ``length.normalize``) override
    via ``length_max_ratio``; modes that opt into date preservation set
    ``preserve_dates`` and surface the scorer in their pipeline.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cosine_min: float = Field(default=0.85, ge=0.0, le=1.0)
    nli_min: float = Field(default=0.70, ge=0.0, le=1.0)
    hhem_min: float = Field(default=0.50, ge=0.0, le=1.0)
    reject_on_negation_diff: bool = True
    preserve_entities: bool = True
    preserve_numbers: bool = True
    preserve_urls: bool = True
    preserve_dates: bool = False
    length_min_ratio: float = Field(default=0.4, ge=0.0)
    length_max_ratio: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_length_band(self) -> VerifierProfile:
        if self.length_min_ratio > self.length_max_ratio:
            raise ValueError(
                f"length_min_ratio {self.length_min_ratio} cannot exceed "
                f"length_max_ratio {self.length_max_ratio}"
            )
        return self


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
