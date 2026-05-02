"""Unit tests for the min_model_b precondition (P3-BACK-09)."""

from __future__ import annotations

import pytest

from transduce.backends.preconditions import (
    BackendMinModelNotMetError,
    enforce_min_model_b,
)

pytestmark = pytest.mark.unit


def test_enforce_min_model_b_passes_when_floor_is_zero() -> None:
    enforce_min_model_b(
        mode_id="dejargon",
        required_b=0.0,
        backend_id="ollama_qwen",
        backend_model="qwen2.5:1.5b",
        declared_b=None,
    )


def test_enforce_min_model_b_passes_when_declared_meets_floor() -> None:
    enforce_min_model_b(
        mode_id="voice-match",
        required_b=14.0,
        backend_id="vllm_qwen",
        backend_model="Qwen/Qwen2.5-14B-Instruct",
        declared_b=14.0,
    )


def test_enforce_min_model_b_passes_when_declared_exceeds_floor() -> None:
    enforce_min_model_b(
        mode_id="voice-match",
        required_b=14.0,
        backend_id="vllm_qwen",
        backend_model="Qwen/Qwen2.5-32B-Instruct",
        declared_b=32.0,
    )


def test_enforce_min_model_b_raises_when_declared_below_floor() -> None:
    with pytest.raises(BackendMinModelNotMetError) as exc_info:
        enforce_min_model_b(
            mode_id="voice-match",
            required_b=14.0,
            backend_id="ollama_qwen",
            backend_model="qwen2.5:1.5b",
            declared_b=1.5,
        )

    assert exc_info.value.mode_id == "voice-match"
    assert exc_info.value.backend_id == "ollama_qwen"
    assert exc_info.value.required_b == pytest.approx(14.0)
    assert exc_info.value.actual_b == pytest.approx(1.5)
    assert "1.5B" in str(exc_info.value)


def test_enforce_min_model_b_raises_when_declared_is_none_and_floor_positive() -> None:
    with pytest.raises(BackendMinModelNotMetError) as exc_info:
        enforce_min_model_b(
            mode_id="dejargon",
            required_b=14.0,
            backend_id="ollama_qwen",
            backend_model="qwen2.5:1.5b",
            declared_b=None,
        )

    assert exc_info.value.actual_b is None
    assert "leaves model_size_b unset" in str(exc_info.value)
