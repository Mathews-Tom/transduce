"""Tests for OTel attribute name constants (P3-OBS-01).

The attribute namespacing tests guard against accidental drift away from
the OTel GenAI SemConv ``gen_ai.*`` namespace and the documented
``transduce.*`` extension namespace. R-05 (SemConv churn) remediation
relies on these names being grep-stable.
"""

from __future__ import annotations

import pytest

from transduce.observability import attributes


@pytest.mark.unit
def test_attributes_match_gen_ai_namespacing() -> None:
    gen_ai_constants = {
        attributes.GEN_AI_SYSTEM,
        attributes.GEN_AI_REQUEST_MODEL,
        attributes.GEN_AI_USAGE_INPUT_TOKENS,
        attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
        attributes.GEN_AI_RESPONSE_FINISH_REASONS,
    }

    assert all(name.startswith("gen_ai.") for name in gen_ai_constants)


@pytest.mark.unit
def test_transduce_extensions_share_transduce_prefix() -> None:
    transduce_constants = {
        attributes.TRANSDUCE_MODE_ID,
        attributes.TRANSDUCE_MODE_VERSION,
        attributes.TRANSDUCE_LANGUAGE,
        attributes.TRANSDUCE_VERDICT,
        attributes.TRANSDUCE_RETRIES,
        attributes.TRANSDUCE_COST_USD,
        attributes.TRANSDUCE_ATTEMPT,
        attributes.TRANSDUCE_TEXT_SHA256_8,
        attributes.TRANSDUCE_TEXT_LENGTH,
        attributes.TRANSDUCE_TEXT_VALUE,
        attributes.TRANSDUCE_SCAN_MATCHED_PATTERN,
        attributes.TRANSDUCE_SCORER_COSINE,
        attributes.TRANSDUCE_SCORER_NLI_FORWARD,
        attributes.TRANSDUCE_SCORER_NLI_BACKWARD,
        attributes.TRANSDUCE_SCORER_HHEM,
        attributes.TRANSDUCE_SCORER_NEGATION_DIFF_COUNT,
        attributes.TRANSDUCE_REJECTION_REASON,
        attributes.TRANSDUCE_COMPOSE_STAGES,
        attributes.TRANSDUCE_COMPOSE_DRIFT_TOTAL,
        attributes.TRANSDUCE_DIFF_OPS_COUNT,
        attributes.TRANSDUCE_DIFF_CHARS_CHANGED,
    }

    assert all(name.startswith("transduce.") for name in transduce_constants)


@pytest.mark.unit
def test_span_names_use_documented_layout() -> None:
    assert attributes.SPAN_REQUEST == "gen_ai.client.request"
    assert attributes.SPAN_SCAN == "transduce.scan"
    assert attributes.SPAN_GENERATE == "transduce.generate"
    assert attributes.SPAN_VERIFY == "transduce.verify"
    assert attributes.SPAN_COMPOSE == "transduce.compose"
    assert attributes.SPAN_DIFF == "transduce.diff"


@pytest.mark.unit
def test_gen_ai_system_value_is_transduce() -> None:
    assert attributes.GEN_AI_SYSTEM_TRANSDUCE == "transduce"
