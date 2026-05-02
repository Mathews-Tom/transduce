"""Unit tests for the LiteLLM router meta-backend (P3-BACK-05)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from litellm import exceptions as litellm_exceptions

from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
    TokenPricing,
)
from transduce.backends.litellm_router import LiteLLMRouterBackend

pytestmark = pytest.mark.unit


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _StubResponse:
    choices: list[_StubChoice]
    usage: _StubUsage | None = None


def _async_raise(exc: BaseException) -> Any:
    async def _coro(**_: Any) -> Any:
        raise exc

    return _coro


async def test_litellm_generate_returns_text_and_tokens() -> None:
    captured: dict[str, Any] = {}

    async def completion(**kwargs: Any) -> _StubResponse:
        captured.update(kwargs)
        return _StubResponse(
            choices=[_StubChoice(_StubMessage(content="rewritten"))],
            usage=_StubUsage(prompt_tokens=8, completion_tokens=2),
        )

    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        completion=completion,
    )

    result = await backend.generate("input", max_tokens=32, temperature=0.0)

    assert result.text == "rewritten"
    assert result.tokens_in == 8
    assert result.tokens_out == 2
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["api_key"] == "sk-test"
    assert captured["messages"] == [{"role": "user", "content": "input"}]
    assert captured["max_tokens"] == 32
    assert captured["timeout"] == pytest.approx(60.0)


async def test_litellm_generate_timeout_raises_generation_timeout() -> None:
    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        completion=_async_raise(
            litellm_exceptions.Timeout("boom", "anthropic", "claude-haiku-4-5")
        ),
    )

    with pytest.raises(GenerationTimeoutError, match="timed out"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_litellm_generate_connection_error_raises_backend_unavailable() -> None:
    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        completion=_async_raise(
            litellm_exceptions.APIConnectionError(
                message="refused", llm_provider="anthropic", model="claude-haiku-4-5"
            )
        ),
    )

    with pytest.raises(BackendUnavailableError, match="unreachable"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_litellm_generate_api_error_raises_generation_failed() -> None:
    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        completion=_async_raise(
            litellm_exceptions.APIError(
                status_code=502,
                message="upstream error",
                llm_provider="anthropic",
                model="claude-haiku-4-5",
            )
        ),
    )

    with pytest.raises(GenerationFailedError, match="API error"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_litellm_generate_missing_content_raises_generation_failed() -> None:
    async def completion(**_: Any) -> _StubResponse:
        return _StubResponse(choices=[])

    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        completion=completion,
    )

    with pytest.raises(GenerationFailedError, match="choices"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_litellm_health_returns_true_for_known_alias() -> None:
    backend = LiteLLMRouterBackend(model="claude-haiku-4-5", api_key="sk-test")

    health = await backend.health()

    assert health.healthy is True
    assert "anthropic" in (health.detail or "")


async def test_litellm_health_returns_false_for_unknown_alias() -> None:
    backend = LiteLLMRouterBackend(model="bogus-model-xyz-12345", api_key="sk-test")

    health = await backend.health()

    assert health.healthy is False


def test_litellm_cost_estimate_with_pricing_returns_dollar_amount() -> None:
    pricing = TokenPricing(in_per_million_usd=2.00, out_per_million_usd=8.00)
    backend = LiteLLMRouterBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        pricing=pricing,
    )

    estimate = backend.cost_estimate(tokens_in=500_000, tokens_out=125_000)

    assert estimate == pytest.approx(500_000 * 2.00 / 1_000_000 + 125_000 * 8.00 / 1_000_000)


def test_litellm_cost_estimate_without_pricing_returns_none() -> None:
    backend = LiteLLMRouterBackend(model="claude-haiku-4-5", api_key="sk-test")

    assert backend.cost_estimate(tokens_in=1, tokens_out=1) is None


def test_litellm_construction_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        LiteLLMRouterBackend(model="", api_key="x")


def test_litellm_construction_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        LiteLLMRouterBackend(model="m", api_key="")
