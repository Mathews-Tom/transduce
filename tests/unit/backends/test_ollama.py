"""Unit tests for the Ollama backend adapter (P1-BACK-01..03)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
)
from transduce.backends.ollama import OllamaBackend

pytestmark = pytest.mark.unit


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.local:11434",
    )


def _ollama_ok(payload: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    return handler


async def test_ollama_payload_construction_includes_model_and_prompt() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"response": "ok", "prompt_eval_count": 4, "eval_count": 2},
        )

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    result = await backend.generate("hello", max_tokens=64, temperature=0.2)

    assert result.text == "ok"
    assert result.tokens_in == 4
    assert result.tokens_out == 2
    assert "qwen2.5:14b" in captured["body"]
    assert "hello" in captured["body"]


async def test_ollama_unreachable_raises_backend_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    with pytest.raises(BackendUnavailableError, match="ollama unreachable"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_timeout_raises_generation_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("read timeout")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        timeout_s=5.0,
        client=_client(handler),
    )

    with pytest.raises(GenerationTimeoutError, match="timed out after 5"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_non_200_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model loading")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="status 500"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_non_json_response_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="non-JSON"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_response_missing_text_raises_generation_failed() -> None:
    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(_ollama_ok({"prompt_eval_count": 1, "eval_count": 1})),
    )

    with pytest.raises(GenerationFailedError, match="missing required string field"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_token_count_negative_raises_generation_failed() -> None:
    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(_ollama_ok({"response": "ok", "prompt_eval_count": -1})),
    )

    with pytest.raises(GenerationFailedError, match="non-integer"):
        await backend.generate("x", max_tokens=8, temperature=0.0)


async def test_ollama_token_count_missing_defaults_to_zero() -> None:
    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(_ollama_ok({"response": "ok"})),
    )

    result = await backend.generate("x", max_tokens=8, temperature=0.0)

    assert result.tokens_in == 0
    assert result.tokens_out == 0


async def test_ollama_max_tokens_must_be_positive() -> None:
    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(_ollama_ok({"response": "ok"})),
    )

    with pytest.raises(ValueError, match="max_tokens"):
        await backend.generate("x", max_tokens=0, temperature=0.0)


async def test_ollama_health_returns_true_when_tags_responds_ok() -> None:
    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(_ollama_ok({"response": "ok"})),
    )

    health = await backend.health()

    assert health.healthy is True


async def test_ollama_health_returns_false_on_connection_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    health = await backend.health()

    assert health.healthy is False
    assert "connection refused" in (health.detail or "")


async def test_ollama_health_returns_false_on_non_200() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_client(handler),
    )

    health = await backend.health()

    assert health.healthy is False
    assert "status 503" in (health.detail or "")


async def test_ollama_async_context_manager_closes_client() -> None:
    handler_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        handler_calls.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    async with OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
    ) as backend:
        backend._client = _client(handler)
        backend._owns_client = True
        await backend.health()

    assert handler_calls == ["/api/tags"]


def test_ollama_construction_rejects_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        OllamaBackend(endpoint="", model="qwen2.5:14b")


def test_ollama_construction_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OllamaBackend(endpoint="http://ollama.local", model="")


def test_ollama_cost_estimate_returns_none_for_local_backend() -> None:
    backend = OllamaBackend(endpoint="http://ollama.local", model="qwen2.5:14b")

    estimate = backend.cost_estimate(tokens_in=1000, tokens_out=500)

    assert estimate is None


def test_ollama_cost_estimate_negative_tokens_raises_value_error() -> None:
    backend = OllamaBackend(endpoint="http://ollama.local", model="qwen2.5:14b")

    with pytest.raises(ValueError, match="non-negative"):
        backend.cost_estimate(tokens_in=-1, tokens_out=10)
