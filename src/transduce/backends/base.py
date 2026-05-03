"""Backend adapter protocol per docs/system-design.md §Backend Adapter Layer.

Every backend honours the same ``Backend`` Protocol — ``generate``,
``health``, and ``cost_estimate`` — so the pipeline orchestrator picks
implementations interchangeably. The Phase-1 Ollama adapter, the Phase-3
cloud adapters (Anthropic, OpenAI-compat, LiteLLM), and the
self-hosted OpenAI-compat adapters (vLLM, llama.cpp) all plug into the
same surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class BackendCapabilities(BaseModel):
    """Static feature flags advertised by a backend implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    streaming: bool = False
    json_mode: bool = False
    attention_output: bool = False


class GenerationResult(BaseModel):
    """Outcome of a single ``generate`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)


class StreamTextDelta(BaseModel):
    """One text-delta event yielded by a streaming generation (P3-STR-01).

    Streaming backends yield zero-or-more ``StreamTextDelta`` events
    with monotonically-extending text segments, then a single
    :class:`StreamFinal` carrying the prompt and completion token totals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    text: str = Field(min_length=1)


class StreamFinal(BaseModel):
    """Terminal event for a streaming generation (P3-STR-01).

    Closes a stream with the final token totals so the budgeter and the
    OTel ``transduce.generate`` span can record the same usage as the
    non-streaming path. Backends that cannot extract the token totals
    from their stream (some OpenAI-compatible servers omit ``usage``
    under streaming) return zeros — the budgeter records ``0.0`` for
    the attempt while still bounding ``max_retries``, mirroring the
    local-backend cost-estimate convention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["final"] = "final"
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)


StreamChunk = StreamTextDelta | StreamFinal
"""Discriminated union of events a backend yields during streaming."""


class BackendHealth(BaseModel):
    """Liveness/readiness probe outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    healthy: bool
    detail: str | None = None


class TokenPricing(BaseModel):
    """USD-per-million-tokens pricing for a backend (P3-BACK-07).

    Cloud backends compose ``TokenPricing`` from their published rates
    (or operator overrides via ``BackendEntry.cost_in_per_million_usd``)
    and call :meth:`estimate` on every generate to record per-attempt
    cost into the budgeter. Local backends (Ollama, vLLM, llama.cpp) do
    not pay per-token costs and return ``None`` from ``cost_estimate``;
    the budget guard treats ``None`` as ``0.0`` so local retries stay
    unbounded by money but still bounded by ``max_retries``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    in_per_million_usd: float = Field(ge=0.0)
    out_per_million_usd: float = Field(ge=0.0)

    def estimate(self, *, tokens_in: int, tokens_out: int) -> float:
        """Return projected USD cost for a (``tokens_in``, ``tokens_out``) call."""
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError("token counts must be non-negative")
        return (
            tokens_in * self.in_per_million_usd / 1_000_000.0
            + tokens_out * self.out_per_million_usd / 1_000_000.0
        )


class BackendError(RuntimeError):
    """Base class for backend-side failures the pipeline can map to error codes."""


class BackendUnavailableError(BackendError):
    """Raised when the backend cannot be reached (connection refused, DNS fail)."""


class GenerationTimeoutError(BackendError):
    """Raised when generation exceeds the configured timeout."""


class GenerationFailedError(BackendError):
    """Raised on non-success HTTP status or malformed response payload."""


class Backend(Protocol):
    """The minimum contract every transduce backend implementation honours."""

    name: str
    model: str
    capabilities: BackendCapabilities

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Generate a completion for ``prompt`` and return its text + token counts."""
        ...  # pragma: no cover — Protocol method

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a generation for ``prompt`` as text deltas + a final event (P3-STR-01).

        Concrete implementations are ``async def`` generators: they
        yield :class:`StreamTextDelta` instances for each provider
        chunk and exactly one :class:`StreamFinal` after the stream
        closes. Backends that do not advertise ``capabilities.streaming``
        may raise :class:`NotImplementedError` from this method;
        ``post_transform_stream`` validates the capability before
        invoking it so honest 400 responses are emitted at ingress.
        """
        ...  # pragma: no cover — Protocol method

    async def health(self) -> BackendHealth:
        """Probe the backend service and return a structured health verdict."""
        ...  # pragma: no cover — Protocol method

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        """Return projected USD cost for a generate call, or ``None`` for unpriced backends.

        Local backends (Ollama, vLLM, llama.cpp) return ``None``; the
        budgeter records ``0.0`` for the attempt but still enforces the
        max-retry ceiling. Cloud backends return a positive float
        derived from their :class:`TokenPricing`.
        """
        ...  # pragma: no cover — Protocol method
