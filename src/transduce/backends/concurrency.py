"""Per-backend concurrency semaphore (P3-BACK-06).

Wraps a :class:`Backend` with an :class:`asyncio.Semaphore`. Generate
calls beyond the configured limit raise
:class:`ConcurrencyLimitExceededError` immediately rather than blocking
on Litestar's request timeout — the API layer maps the exception onto
``429 Concurrency Limit Exceeded`` with a ``Retry-After`` header per
docs/system-design.md §Backend Adapter Layer.

``health`` and ``cost_estimate`` bypass the semaphore. Health probes
are operator-side and must succeed even under load; cost estimation is
pure arithmetic with no resource cost.
"""

from __future__ import annotations

import asyncio

from transduce.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    BackendHealth,
    GenerationResult,
)


class ConcurrencyLimitExceededError(BackendError):
    """Raised when a backend's concurrency semaphore has no permits left."""

    def __init__(
        self,
        *,
        backend_id: str,
        limit: int,
        retry_after_s: float,
    ) -> None:
        super().__init__(
            f"backend {backend_id!r} concurrency limit {limit} exhausted; "
            f"retry after {retry_after_s:g}s"
        )
        self.backend_id = backend_id
        self.limit = limit
        self.retry_after_s = retry_after_s


class SemaphoreBackend:
    """Decorate a backend with a concurrency limit and retry-after metadata.

    The wrapper forwards :class:`Backend` protocol attributes (``name``,
    ``model``, ``capabilities``) so callers cannot tell the wrapper from
    the inner backend by inspection. The semaphore guards ``generate``
    only; ``health`` and ``cost_estimate`` are unaffected.
    """

    def __init__(
        self,
        *,
        inner: Backend,
        backend_id: str,
        limit: int,
        retry_after_s: float = 1.0,
    ) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        if retry_after_s < 0.0:
            raise ValueError(f"retry_after_s must be non-negative, got {retry_after_s}")
        self._inner = inner
        self._backend_id = backend_id
        self._limit = limit
        self._retry_after_s = retry_after_s
        self._semaphore = asyncio.Semaphore(limit)
        self.name = inner.name
        self.model = inner.model
        self.capabilities: BackendCapabilities = inner.capabilities

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def limit(self) -> int:
        return self._limit

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        if self._semaphore.locked():
            raise ConcurrencyLimitExceededError(
                backend_id=self._backend_id,
                limit=self._limit,
                retry_after_s=self._retry_after_s,
            )
        async with self._semaphore:
            return await self._inner.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def health(self) -> BackendHealth:
        return await self._inner.health()

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        return self._inner.cost_estimate(tokens_in=tokens_in, tokens_out=tokens_out)


__all__ = ["ConcurrencyLimitExceededError", "SemaphoreBackend"]
