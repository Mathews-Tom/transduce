"""Backend adapter protocol per docs/system-design.md §Backend Adapter Layer.

The v0 surface is the ``Backend`` Protocol with ``generate`` and ``health``
methods. The v1 expansion (Anthropic, vLLM, llama.cpp, OpenAI-compat,
LiteLLM router — P3-BACK-01..P3-BACK-05) shares the same surface, so
wiring v0 against the protocol now means later releases plug in without
refactoring the pipeline orchestrator.
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
