"""HTTP API contracts for transduce v0.

Mirrors docs/system-design.md §Data Models. The v0 subset omits compose
chains, streaming-strict mode, language detection, cost-budget overrides,
and ensemble-verifier scores; the v0.5 release widens ``VerificationScores``
with NLI/HHEM/negation outputs (P2-VER-01..P2-VER-09) and v1 adds multi-mode
dispatch (P3-COMP-01).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from transduce.registry.spec import PreserveRule


class ErrorCode(StrEnum):
    """Stable error codes returned in the ``TransformError`` envelope.

    The v0 subset; later releases add ``input_injection_detected``
    (P2-INJ-03), ``language_not_supported`` (P3-LANG-03),
    ``budget_exceeded`` (P3-BUDG-04), and others.
    """

    MODE_NOT_FOUND = "mode_not_found"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    VERIFICATION_FAILED = "verification_failed"
    INPUT_TOO_LONG = "input_too_long"
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

    The v0 subset reports cosine and preservation outcomes. ``topical_similarity``
    is the client-facing aggregate per docs/system-design.md §Verification Subsystem.
    The v0.5 release widens this with bidirectional NLI, HHEM, and negation-diff
    fields (P2-VER-01..P2-VER-09).
    """

    model_config = ConfigDict(extra="forbid")

    cosine: float = Field(ge=0.0, le=1.0)
    preserved: dict[str, bool]
    topical_similarity: float = Field(ge=0.0, le=1.0)
    verdict: Literal["accept", "reject"]
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
    """Successful transformation response per docs/system-design.md §Data Models."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    mode: ModeRef
    language: str = Field(min_length=1)
    original: str
    transformed: str
    diff: list[DiffOp]
    scores: VerificationScores
    backend_used: BackendInfo
    timing: TimingBreakdown
    retries: int = Field(ge=0)
    cost: CostBreakdown


class TransformError(BaseModel):
    """Error envelope returned for non-2xx responses (P1-API-06)."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    error: ErrorCode
    message: str = Field(min_length=1)
    details: dict[str, Any] | None = None
    last_candidate: str | None = None
    scores: VerificationScores | None = None
