"""Unit tests for the concurrency-semaphore backend wrapper (P3-BACK-06)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    GenerationResult,
    StreamChunk,
    StreamFinal,
    StreamTextDelta,
)
from transduce.backends.concurrency import (
    ConcurrencyLimitExceededError,
    SemaphoreBackend,
)

pytestmark = pytest.mark.unit


class _FakeBackend:
    """Minimal backend stand-in honouring the protocol surface."""

    name: str = "fake"
    model: str = "fake-model"
    capabilities = BackendCapabilities()

    def __init__(self, *, cost: float | None = None) -> None:
        self._cost = cost
        self.health_calls = 0
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        self.generate_calls += 1
        await asyncio.sleep(0)
        return GenerationResult(text=f"echo:{prompt}", tokens_in=1, tokens_out=1)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        del max_tokens, temperature
        self.stream_calls += 1
        yield StreamTextDelta(text=f"echo:{prompt}")
        yield StreamFinal(tokens_in=1, tokens_out=1)

    async def health(self) -> BackendHealth:
        self.health_calls += 1
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        return self._cost


async def test_semaphore_acquires_within_limit_and_returns_result() -> None:
    inner = _FakeBackend()
    wrapped = SemaphoreBackend(inner=inner, backend_id="ollama_qwen", limit=2)

    result = await wrapped.generate("hi", max_tokens=4, temperature=0.0)

    assert result.text == "echo:hi"
    assert inner.generate_calls == 1


async def test_semaphore_exhausted_raises_concurrency_limit_exceeded() -> None:
    release = asyncio.Event()
    inner = _FakeBackend()

    async def slow_generate(
        self: _FakeBackend,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        self.generate_calls += 1
        await release.wait()
        return GenerationResult(text=prompt, tokens_in=1, tokens_out=1)

    inner.generate = slow_generate.__get__(inner, _FakeBackend)  # type: ignore[method-assign]

    wrapped = SemaphoreBackend(inner=inner, backend_id="ollama_qwen", limit=1)
    holder = asyncio.create_task(wrapped.generate("first", max_tokens=4, temperature=0.0))
    await asyncio.sleep(0)  # let holder enter the semaphore

    with pytest.raises(ConcurrencyLimitExceededError) as exc_info:
        await wrapped.generate("second", max_tokens=4, temperature=0.0)

    assert exc_info.value.backend_id == "ollama_qwen"
    assert exc_info.value.limit == 1
    assert exc_info.value.retry_after_s == pytest.approx(1.0)

    release.set()
    await holder


async def test_semaphore_releases_permit_after_inner_exception() -> None:
    failures = {"count": 0}

    async def failing_generate(
        self: _FakeBackend,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        failures["count"] += 1
        raise RuntimeError("boom")

    inner = _FakeBackend()
    inner.generate = failing_generate.__get__(inner, _FakeBackend)  # type: ignore[method-assign]
    wrapped = SemaphoreBackend(inner=inner, backend_id="b", limit=1)

    with pytest.raises(RuntimeError, match="boom"):
        await wrapped.generate("a", max_tokens=4, temperature=0.0)

    # Permit returned: the next call succeeds rather than raising
    # ConcurrencyLimitExceededError.
    with pytest.raises(RuntimeError, match="boom"):
        await wrapped.generate("b", max_tokens=4, temperature=0.0)

    assert failures["count"] == 2


async def test_semaphore_health_bypasses_limit() -> None:
    inner = _FakeBackend()
    wrapped = SemaphoreBackend(inner=inner, backend_id="b", limit=1)

    # Hold the semaphore via a never-resolving generate.
    release = asyncio.Event()

    async def long_generate(
        self: _FakeBackend,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        await release.wait()
        return GenerationResult(text=prompt, tokens_in=1, tokens_out=1)

    inner.generate = long_generate.__get__(inner, _FakeBackend)  # type: ignore[method-assign]
    holder = asyncio.create_task(wrapped.generate("x", max_tokens=4, temperature=0.0))
    await asyncio.sleep(0)

    health = await wrapped.health()

    assert health.healthy is True
    assert inner.health_calls == 1

    release.set()
    await holder


def test_semaphore_cost_estimate_forwards_to_inner() -> None:
    inner = _FakeBackend(cost=0.0125)
    wrapped = SemaphoreBackend(inner=inner, backend_id="b", limit=1)

    estimate = wrapped.cost_estimate(tokens_in=100, tokens_out=200)

    assert estimate == pytest.approx(0.0125)


def test_semaphore_invalid_limit_raises_value_error() -> None:
    inner = _FakeBackend()

    with pytest.raises(ValueError, match=">= 1"):
        SemaphoreBackend(inner=inner, backend_id="b", limit=0)


def test_semaphore_negative_retry_after_raises_value_error() -> None:
    inner = _FakeBackend()

    with pytest.raises(ValueError, match="non-negative"):
        SemaphoreBackend(inner=inner, backend_id="b", limit=1, retry_after_s=-0.1)


def test_semaphore_forwards_identity_attrs() -> None:
    inner = _FakeBackend()
    wrapped = SemaphoreBackend(inner=inner, backend_id="b", limit=2)

    assert wrapped.name == "fake"
    assert wrapped.model == "fake-model"
    assert wrapped.backend_id == "b"
    assert wrapped.limit == 2
