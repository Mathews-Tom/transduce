"""HTTP API contracts for transduce.

Mirrors docs/system-design.md §Data Models. The v0.5 release widens
``VerificationScores`` with NLI/HHEM/negation-diff fields (P2-VER-09);
v1 adds multi-mode dispatch (P3-COMP-01) and streaming-advisory
(P3-STR-01).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from transduce.registry.spec import PreserveRule
from transduce.verification.negation import NegationDiffResult


class ErrorCode(StrEnum):
    """Stable error codes returned in the ``TransformError`` envelope.

    v0.5 additions: ``input_injection_detected`` (P2-INJ-03) and
    ``mode_hash_mismatch`` (P2-PLG-02). v1 adds
    ``language_not_supported`` (P3-LANG-03), ``budget_exceeded``
    (P3-BUDG-04), ``concurrency_limit_exceeded`` (P3-BACK-06),
    ``backend_min_model_not_met`` (P3-BACK-09),
    ``mode_version_not_found`` (P3-VER-03), and
    ``composite_verification_failed`` (P3-COMP-06).
    """

    MODE_NOT_FOUND = "mode_not_found"
    MODE_VERSION_NOT_FOUND = "mode_version_not_found"
    MODE_HASH_MISMATCH = "mode_hash_mismatch"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_MIN_MODEL_NOT_MET = "backend_min_model_not_met"
    VERIFICATION_FAILED = "verification_failed"
    COMPOSITE_VERIFICATION_FAILED = "composite_verification_failed"
    INPUT_TOO_LONG = "input_too_long"
    INPUT_INJECTION_DETECTED = "input_injection_detected"
    LANGUAGE_NOT_SUPPORTED = "language_not_supported"
    BUDGET_EXCEEDED = "budget_exceeded"
    CONCURRENCY_LIMIT_EXCEEDED = "concurrency_limit_exceeded"
    GENERATION_FAILED = "generation_failed"
    NOT_IMPLEMENTED = "not_implemented"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


class StreamingMode(StrEnum):
    """Streaming options. ``advisory`` lands with v1 (P3-STR-01); ``strict`` aliases ``off``."""

    OFF = "off"


class ModeRef(BaseModel):
    """Resolved mode identity returned in responses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class BackendOverride(BaseModel):
    """Per-request backend override."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"]
    model: str = Field(min_length=1)


class VerificationOverride(BaseModel):
    """Per-request verifier overrides for the v0 cosine + preservation pipeline."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cosine_min: float | None = Field(default=None, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0, le=5)
    advisory: bool = False


class TransformRequest(BaseModel):
    """Inbound transformation request."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=50_000)
    mode: str | list[str] = Field(
        description="Mode id; list form is reserved for compose chains and is rejected in v0.",
    )
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    preserve: list[PreserveRule] = Field(default_factory=list)
    backend: BackendOverride | None = None
    verification: VerificationOverride | None = None
    streaming: StreamingMode = StreamingMode.OFF
    request_id: str | None = None


class DiffOp(BaseModel):
    """Single operation in a word-level diff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op: Literal["equal", "insert", "delete"]
    text: str


class VerificationScores(BaseModel):
    """Per-scorer outcomes attached to a transformation response.

    The v0.5 ensemble reports cosine, bidirectional NLI (forward and
    backward directions), HHEM factuality, the negation-diff structure,
    preservation outcomes, and any mode-specific scorer outputs.
    ``topical_similarity`` is the client-facing aggregate per
    docs/system-design.md §Verification Subsystem.

    Scorers that did not run because the ensemble short-circuited earlier
    have their numeric fields set to ``None`` (e.g., a request that fails
    cosine never runs NLI; ``nli_forward`` and ``nli_backward`` are
    ``None`` in that response). The ``verdict`` literal field present in
    v0 is removed in v0.5; the HTTP status (200 vs 422) carries the
    accept/reject signal at the response level (P2-VER-09, P2-MIG-02).
    """

    model_config = ConfigDict(extra="forbid")

    cosine: float = Field(ge=0.0, le=1.0)
    nli_forward: float | None = Field(default=None, ge=0.0, le=1.0)
    nli_backward: float | None = Field(default=None, ge=0.0, le=1.0)
    hhem: float | None = Field(default=None, ge=0.0, le=1.0)
    negation_diff: NegationDiffResult = Field(default_factory=NegationDiffResult)
    preserved: dict[str, bool]
    mode_specific: dict[str, float] = Field(default_factory=dict)
    topical_similarity: float = Field(ge=0.0, le=1.0)
    rejection_reason: str | None = None


class BackendInfo(BaseModel):
    """Backend identity surfaced on responses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class TimingBreakdown(BaseModel):
    """Per-stage timing in milliseconds for the v0 5-stage pipeline."""

    model_config = ConfigDict(extra="forbid")

    resolve_ms: int = Field(ge=0)
    generate_ms: int = Field(ge=0)
    verify_ms: int = Field(ge=0)
    diff_ms: int = Field(ge=0)


class AttemptCost(BaseModel):
    """Cost of one generation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt: int = Field(ge=1)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    usd: float = Field(ge=0.0)


class CostBreakdown(BaseModel):
    """Total cost of a request across attempts."""

    model_config = ConfigDict(extra="forbid")

    tokens_in_total: int = Field(ge=0)
    tokens_out_total: int = Field(ge=0)
    usd_total: float = Field(ge=0.0)
    by_attempt: list[AttemptCost]


class TransformResponse(BaseModel):
    """Successful transformation response per docs/system-design.md §Data Models.

    ``mode`` is a single :class:`ModeRef` for single-mode requests and a
    list of :class:`ModeRef` for compose chains (P3-COMP-01).
    ``composite_score`` is populated only when the composite verifier
    ran across a chain (P3-COMP-02).
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    mode: ModeRef | list[ModeRef]
    language: str = Field(min_length=1)
    original: str
    transformed: str
    diff: list[DiffOp]
    scores: VerificationScores
    backend_used: BackendInfo
    timing: TimingBreakdown
    retries: int = Field(ge=0)
    cost: CostBreakdown
    composite_score: float | None = Field(default=None, ge=0.0, le=1.0)


class TransformError(BaseModel):
    """Error envelope returned for non-2xx responses (P1-API-06)."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    error: ErrorCode
    message: str = Field(min_length=1)
    details: dict[str, Any] | None = None
    last_candidate: str | None = None
    scores: VerificationScores | None = None
