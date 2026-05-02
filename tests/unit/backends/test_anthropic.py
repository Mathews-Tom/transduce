"""Unit tests for the Anthropic Messages backend (P3-BACK-01).

The tests inject a fake :class:`AsyncAnthropic` substitute through the
constructor's ``client`` parameter. The real SDK is exercised only by
the (deferred) integration suite gated by an Anthropic test API key.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIStatusError as AnthropicStatusError
from anthropic import APITimeoutError as AnthropicTimeoutError

from transduce.backends.anthropic import AnthropicBackend
from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
    TokenPricing,
)

pytestmark = pytest.mark.unit


@dataclass
class _StubTextBlock:
    text: str
    type: str = "text"


@dataclass
class _StubUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _StubMessage:
    content: list[_StubTextBlock]
    usage: _StubUsage | None = None


@dataclass
class _StubMessages:
    create: Callable[..., Awaitable[Any]]
    count_tokens: Callable[..., Awaitable[Any]] = field(
        default=lambda **_: _async_return(object())
    )


@dataclass
class _StubClient:
    messages: _StubMessages


def _async_return(value: Any) -> Awaitable[Any]:
    async def _coro() -> Any:
        return value

    return _coro()


def _make_client(
    *,
    create: Callable[..., Awaitable[Any]] | None = None,
    count_tokens: Callable[..., Awaitable[Any]] | None = None,
) -> _StubClient:
    create = create or (lambda **_: _async_return(_StubMessage(content=[])))
    count_tokens = count_tokens or (lambda **_: _async_return(object()))
    return _StubClient(messages=_StubMessages(create=create, count_tokens=count_tokens))


def _build_status_error(status_code: int, message: str) -> AnthropicStatusError:
    """Build a real ``APIStatusError`` instance without performing a network call."""

    class _StubResponse:
        def __init__(self) -> None:
            self.status_code = status_code
            self.headers: dict[str, str] = {}

    error = AnthropicStatusError.__new__(AnthropicStatusError)
    error.status_code = status_code
    error.message = message
    error.response = _StubResponse()  # type: ignore[assignment]
    error.body = None
    error.request_id = None
    Exception.__init__(error, message)
    return error


async def test_anthropic_generate_returns_concatenated_text_and_tokens() -> None:
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> _StubMessage:
        captured.update(kwargs)
        return _StubMessage(
            content=[_StubTextBlock(text="hello "), _StubTextBlock(text="world")],
            usage=_StubUsage(input_tokens=12, output_tokens=2),
        )

    client = _make_client(create=create)
    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=client,  # type: ignore[arg-type]
    )

    result = await backend.generate("rewrite this", max_tokens=64, temperature=0.0)

    assert result.text == "hello world"
    assert result.tokens_in == 12
    assert result.tokens_out == 2
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["max_tokens"] == 64
    assert captured["messages"] == [{"role": "user", "content": "rewrite this"}]
    assert "system" not in captured


async def test_anthropic_generate_includes_system_prompt_when_provided() -> None:
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> _StubMessage:
        captured.update(kwargs)
        return _StubMessage(
            content=[_StubTextBlock(text="ok")],
            usage=_StubUsage(input_tokens=1, output_tokens=1),
        )

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(create=create),  # type: ignore[arg-type]
        system_prompt="You are a transformation engine.",
    )

    await backend.generate("x", max_tokens=4, temperature=0.0)

    assert captured["system"] == "You are a transformation engine."


async def test_anthropic_generate_timeout_maps_to_generation_timeout_error() -> None:
    async def create(**_: Any) -> _StubMessage:
        raise AnthropicTimeoutError(request=None)  # type: ignore[arg-type]

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(create=create),  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationTimeoutError, match="timed out"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_anthropic_generate_connection_error_maps_to_backend_unavailable() -> None:
    async def create(**_: Any) -> _StubMessage:
        raise AnthropicConnectionError(request=None)  # type: ignore[arg-type]

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(create=create),  # type: ignore[arg-type]
    )

    with pytest.raises(BackendUnavailableError, match="unreachable"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_anthropic_generate_status_error_maps_to_generation_failed() -> None:
    async def create(**_: Any) -> _StubMessage:
        raise _build_status_error(429, "rate limited")

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(create=create),  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationFailedError, match="status 429"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_anthropic_generate_response_without_text_blocks_raises() -> None:
    async def create(**_: Any) -> _StubMessage:
        return _StubMessage(content=[])

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(create=create),  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationFailedError, match="no text blocks"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_anthropic_health_returns_true_when_count_tokens_succeeds() -> None:
    async def count_tokens(**_: Any) -> object:
        return object()

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(count_tokens=count_tokens),  # type: ignore[arg-type]
    )

    health = await backend.health()

    assert health.healthy is True


async def test_anthropic_health_returns_false_on_status_error() -> None:
    async def count_tokens(**_: Any) -> object:
        raise _build_status_error(503, "unavailable")

    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(count_tokens=count_tokens),  # type: ignore[arg-type]
    )

    health = await backend.health()

    assert health.healthy is False
    assert "503" in (health.detail or "")


def test_anthropic_cost_estimate_with_pricing_returns_dollar_amount() -> None:
    pricing = TokenPricing(in_per_million_usd=1.00, out_per_million_usd=5.00)
    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        pricing=pricing,
        client=_make_client(),  # type: ignore[arg-type]
    )

    estimate = backend.cost_estimate(tokens_in=10_000, tokens_out=2_000)

    assert estimate == pytest.approx(10_000 * 1.00 / 1_000_000 + 2_000 * 5.00 / 1_000_000)


def test_anthropic_cost_estimate_without_pricing_returns_none() -> None:
    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="sk-test",
        client=_make_client(),  # type: ignore[arg-type]
    )

    estimate = backend.cost_estimate(tokens_in=10_000, tokens_out=2_000)

    assert estimate is None


def test_anthropic_construction_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        AnthropicBackend(model="", api_key="sk-test", client=_make_client())  # type: ignore[arg-type]


def test_anthropic_construction_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AnthropicBackend(
            model="claude-haiku-4-5",
            api_key="",
            client=_make_client(),  # type: ignore[arg-type]
        )


def test_anthropic_construction_rejects_zero_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        AnthropicBackend(
            model="claude-haiku-4-5",
            api_key="sk-test",
            timeout_s=0.0,
            client=_make_client(),  # type: ignore[arg-type]
        )
