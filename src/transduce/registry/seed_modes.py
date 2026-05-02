"""Eight in-tree seed modes (P1-REG-03 + P3-MOD-01).

Each mode declares its own prompt template, intensity range, preservation
defaults, backend size floor, and verifier profile. The v0 base ships
``dejargon``, ``register.casual``, and ``length.normalize``; v1 adds
``voice-match``, ``style.match``, ``tone.us-to-uk``, ``simplify.grade-8``,
and ``formal-to-warm``. ``humanize.*`` is intentionally absent per
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

_VOICE_MATCH_PROMPT: Final[str] = (
    "Rewrite the following text to keep the author's distinctive voice — "
    "their characteristic word choices, sentence rhythm, and tone — while "
    "polishing only what would clearly fail a careful editor. Preserve "
    "every entity, number, URL, and factual claim verbatim. Return only "
    "the rewritten text. Intensity: {{ intensity }} (0.0 = preserve voice "
    "exactly, only fix typos; 1.0 = polish prose while keeping voice)."
    "{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)

_STYLE_MATCH_PROMPT: Final[str] = (
    "Rewrite the following text to preserve its stylistic features — "
    "sentence length distribution, vocabulary register, paragraph rhythm "
    "— while improving clarity. Preserve every entity, number, URL, and "
    "factual claim verbatim. Return only the rewritten text. Intensity: "
    "{{ intensity }} (0.0 = minimal changes, preserve style exactly; "
    "1.0 = stronger clarity edits within the original style envelope)."
    "{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)

_TONE_US_TO_UK_PROMPT: Final[str] = (
    "Rewrite the following text in British English: convert American "
    "spellings (color → colour, organize → organise, defense → defence), "
    "swap idioms where the British counterpart is unambiguous (sidewalk "
    "→ pavement, gas → petrol), and prefer British conventions for dates "
    "and quotation marks. Preserve every entity, number, URL, and "
    "factual claim verbatim. Return only the rewritten text. Intensity: "
    "{{ intensity }} (0.0 = spelling only; 1.0 = spelling, idioms, and "
    "phrasing).{% if preserve %} Preserve: {{ preserve | join(', ') }}."
    "{% endif %}\n\nText:\n{{ input }}"
)

_SIMPLIFY_GRADE_8_PROMPT: Final[str] = (
    "Rewrite the following text at roughly an 8th-grade U.S. reading "
    "level: shorter sentences, common vocabulary, and concrete examples "
    "where the original is abstract. Preserve every entity, number, URL, "
    "and factual claim verbatim. Do not omit information. Return only "
    "the rewritten text. Intensity: {{ intensity }} (0.0 = light "
    "vocabulary edits; 1.0 = aggressive simplification with sentence "
    "splits).{% if preserve %} Preserve: {{ preserve | join(', ') }}."
    "{% endif %}\n\nText:\n{{ input }}"
)

_FORMAL_TO_WARM_PROMPT: Final[str] = (
    "Rewrite the following formal text in a warm, conversational register "
    "suitable for customer-facing communication. Keep the meaning intact, "
    "soften corporate phrasing, and let the reader feel addressed by a "
    "person rather than a department. Preserve every entity, number, "
    "URL, and factual claim verbatim. Return only the rewritten text. "
    "Intensity: {{ intensity }} (0.0 = subtle warmth; 1.0 = strongly "
    "human and conversational)."
    "{% if preserve %} Preserve: {{ preserve | join(', ') }}.{% endif %}"
    "\n\nText:\n{{ input }}"
)


def seed_modes() -> tuple[ModeSpec, ...]:
    """Return the in-tree mode specs in registry-load order (v0 + v1 additions)."""
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
        ModeSpec(
            id="voice-match",
            version="0.1.0",
            description="Polish text while preserving the author's distinctive voice "
            "and rhythm; reserved for high-context rewriting.",
            prompt_template=_VOICE_MATCH_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=14.0),
            verifier_profile=VerifierProfile(cosine_min=0.88),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="style.match",
            version="0.1.0",
            description="Preserve the input's stylistic envelope while improving "
            "clarity at the sentence level.",
            prompt_template=_STYLE_MATCH_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=14.0),
            verifier_profile=VerifierProfile(cosine_min=0.86),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="tone.us-to-uk",
            version="0.1.0",
            description="Convert American English to British English: spelling, "
            "idioms, date conventions, and quotation marks.",
            prompt_template=_TONE_US_TO_UK_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=7.0),
            verifier_profile=VerifierProfile(cosine_min=0.88),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="simplify.grade-8",
            version="0.1.0",
            description="Rewrite text at roughly an 8th-grade U.S. reading level "
            "without dropping facts.",
            prompt_template=_SIMPLIFY_GRADE_8_PROMPT,
            intensity_range=(0.0, 1.0),
            preserve_defaults=(
                PreserveRule.ENTITIES,
                PreserveRule.NUMBERS,
                PreserveRule.URLS,
            ),
            backend_requirements=BackendRequirements(min_model_b=7.0),
            verifier_profile=VerifierProfile(cosine_min=0.80),
            supported_languages=("en",),
        ),
        ModeSpec(
            id="formal-to-warm",
            version="0.1.0",
            description="Rewrite formal corporate prose in a warm, conversational "
            "register without losing meaning.",
            prompt_template=_FORMAL_TO_WARM_PROMPT,
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
    )
