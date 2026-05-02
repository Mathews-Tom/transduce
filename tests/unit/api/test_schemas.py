"""Unit tests for transduce API schemas (P1-API-05, P1-API-06)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transduce.api.schemas import (
    AttemptCost,
    BackendInfo,
    BackendOverride,
    CostBreakdown,
    DiffOp,
    ErrorCode,
    ModeRef,
    StreamingMode,
    TimingBreakdown,
    TransformError,
    TransformRequest,
    TransformResponse,
    VerificationOverride,
    VerificationScores,
)
from transduce.registry.spec import PreserveRule

pytestmark = pytest.mark.unit


def _valid_response_kwargs() -> dict[str, object]:
    return {
        "request_id": "req-1",
        "mode": ModeRef(id="dejargon", version="0.1.0"),
        "language": "en",
        "original": "We synergize verticals.",
        "transformed": "We work together across teams.",
        "diff": [DiffOp(op="equal", text="We "), DiffOp(op="insert", text="work")],
        "scores": VerificationScores(
            cosine=0.91,
            preserved={"entities": True, "numbers": True, "urls": True},
            topical_similarity=0.91,
        ),
        "backend_used": BackendInfo(provider="ollama", model="qwen2.5:14b"),
        "timing": TimingBreakdown(resolve_ms=2, generate_ms=180, verify_ms=12, diff_ms=1),
        "retries": 0,
        "cost": CostBreakdown(
            tokens_in_total=42,
            tokens_out_total=18,
            usd_total=0.0,
            by_attempt=[AttemptCost(attempt=1, tokens_in=42, tokens_out=18, usd=0.0)],
        ),
    }


def test_transform_request_min_length_validates_one_char_passes() -> None:
    request = TransformRequest(text="x", mode="dejargon")

    assert request.text == "x"
    assert request.mode == "dejargon"


def test_transform_request_max_length_50001_chars_rejected() -> None:
    oversized = "x" * 50_001

    with pytest.raises(ValidationError) as exc:
        TransformRequest(text=oversized, mode="dejargon")

    assert "at most 50000" in str(exc.value)


def test_transform_request_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        TransformRequest(text="", mode="dejargon")


def test_transform_request_intensity_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        TransformRequest(text="hi", mode="dejargon", intensity=1.5)


def test_transform_request_extra_field_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        TransformRequest.model_validate({"text": "hi", "mode": "dejargon", "extra": True})

    assert "Extra inputs" in str(exc.value)


def test_transform_request_compose_chain_accepted_for_pipeline_rejection() -> None:
    request = TransformRequest(text="hi", mode=["dejargon", "register.casual"])

    assert request.mode == ["dejargon", "register.casual"]


def test_transform_request_default_streaming_is_off() -> None:
    request = TransformRequest(text="hi", mode="dejargon")

    assert request.streaming == StreamingMode.OFF


def test_transform_request_preserve_rule_string_coerced_to_enum() -> None:
    request = TransformRequest.model_validate(
        {"text": "hi", "mode": "dejargon", "preserve": ["entities"]}
    )

    assert request.preserve == [PreserveRule.ENTITIES]


def test_verification_override_max_retries_ceiling_enforced() -> None:
    with pytest.raises(ValidationError):
        VerificationOverride(max_retries=6)


def test_backend_override_unknown_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        BackendOverride.model_validate({"provider": "anthropic", "model": "claude-haiku-4-5"})


def test_diff_op_invalid_op_rejected() -> None:
    with pytest.raises(ValidationError):
        DiffOp.model_validate({"op": "rewrite", "text": "x"})


def test_verification_scores_no_verdict_field_after_v0_migration() -> None:
    """v0.5 (P2-MIG-02) removes the legacy ``verdict`` field; HTTP status carries it."""
    with pytest.raises(ValidationError):
        VerificationScores.model_validate(
            {
                "cosine": 0.9,
                "preserved": {"entities": True},
                "topical_similarity": 0.9,
                "verdict": "accept",
            }
        )


def test_verification_scores_negation_diff_defaults_empty() -> None:
    scores = VerificationScores(
        cosine=0.9,
        preserved={"entities": True},
        topical_similarity=0.9,
    )

    assert scores.negation_diff.added == ()
    assert scores.negation_diff.removed == ()


def test_verification_scores_optional_nli_defaults_to_none() -> None:
    scores = VerificationScores(
        cosine=0.9,
        preserved={"entities": True},
        topical_similarity=0.9,
    )

    assert scores.nli_forward is None
    assert scores.nli_backward is None
    assert scores.hhem is None
    assert scores.mode_specific == {}


def test_transform_response_round_trips_via_model_dump() -> None:
    response = TransformResponse(**_valid_response_kwargs())  # type: ignore[arg-type]

    payload = response.model_dump()
    rebuilt = TransformResponse.model_validate(payload)

    assert rebuilt == response


def test_transform_error_with_validation_error_code_matches_enum() -> None:
    error = TransformError(
        request_id="req-9",
        error=ErrorCode.VALIDATION_ERROR,
        message="text must not be empty",
    )

    assert error.error == ErrorCode.VALIDATION_ERROR
    assert error.error.value == "validation_error"


def test_error_code_enum_matches_documented_codes() -> None:
    documented = {
        "mode_not_found",
        "mode_hash_mismatch",
        "backend_unavailable",
        "verification_failed",
        "input_too_long",
        "input_injection_detected",
        "generation_failed",
        "not_implemented",
        "timeout",
        "validation_error",
    }

    assert {code.value for code in ErrorCode} == documented


def test_cost_breakdown_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        CostBreakdown(
            tokens_in_total=-1,
            tokens_out_total=0,
            usd_total=0.0,
            by_attempt=[AttemptCost(attempt=1, tokens_in=0, tokens_out=0, usd=0.0)],
        )
