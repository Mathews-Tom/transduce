"""Backend adapter protocol per docs/system-design.md §Backend Adapter Layer.

Every backend honours the same ``Backend`` Protocol — ``generate``,
``health``, and ``cost_estimate`` — so the pipeline orchestrator picks
implementations interchangeably. The Phase-1 Ollama adapter, the Phase-3
cloud adapters (Anthropic, OpenAI-compat, LiteLLM), and the
self-hosted OpenAI-compat adapters (vLLM, llama.cpp) all plug into the
same surface.
"""

from __future__ import annotations

from typing import Protocol

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
