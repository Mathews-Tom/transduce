"""Three in-tree seed modes shipped with v0 (P1-REG-03).

Each mode declares its own prompt template, intensity range, preservation
defaults, backend size floor, and verifier profile. The dev plan freezes
this set at three: ``dejargon`` (jargon reduction), ``register.casual``
(register shift), and ``length.normalize`` (length cap). The v1 release
adds five more (P3-MOD-01); ``humanize.*`` is intentionally absent per
``docs/overview.md`` Possible-Moat §3.
"""

from __future__ import annotations

from typing import Final

from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    PreserveRule,
    VerifierProfile,
)

_DEJARGON_PROMPT: Final[str] = (
    "Rewrite the following text to reduce business jargon while preserving "
    "every entity, number, URL, and concrete claim. Return only the rewritten "
    "text. Intensity: {{ intensity }} (0.0 = light edit, 1.0 = aggressive "
    "de-jargoning).{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)

_REGISTER_CASUAL_PROMPT: Final[str] = (
    "Rewrite the following text in a casual register suitable for a "
    "conversation between colleagues, while preserving every entity, "
    "number, URL, and concrete claim. Return only the rewritten text. "
    "Intensity: {{ intensity }} (0.0 = subtle warmth, 1.0 = strongly "
    "casual).{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)

_LENGTH_NORMALIZE_PROMPT: Final[str] = (
    "Rewrite the following text to fit within {{ max_chars | default(280) }} "
    "characters while preserving every entity, number, URL, and concrete "
    "claim. Return only the rewritten text. Intensity: {{ intensity }} "
    "(0.0 = preserve almost all wording, 1.0 = aggressive compression)."
    "{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)


def seed_modes() -> tuple[ModeSpec, ...]:
    """Return the v0 in-tree mode specs in registry-load order."""
    return (
        ModeSpec(
            id="dejargon",
            version="0.1.0",
            description="Reduce business jargon density without altering claims, "
            "entities, or numerical values.",
            prompt_template=_DEJARGON_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=14.0),
            verifier_profile=VerifierProfile(cosine_min=0.85),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="register.casual",
            version="0.1.0",
            description="Shift the register towards conversational while preserving "
            "named entities, numbers, and URLs.",
            prompt_template=_REGISTER_CASUAL_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=7.0),
            verifier_profile=VerifierProfile(cosine_min=0.82),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="length.normalize",
            version="0.1.0",
            description="Compress text within a target character budget while "
            "preserving facts, entities, numbers, and URLs.",
            prompt_template=_LENGTH_NORMALIZE_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=7.0),
            verifier_profile=VerifierProfile(cosine_min=0.78),
            supported_languages=("en",),
        ),
    )
