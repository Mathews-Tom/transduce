"""Unit tests for the OpenAI-compat backend (P3-BACK-02..04)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
    TokenPricing,
)
from transduce.backends.openai_compat import OpenAICompatBackend

pytestmark = pytest.mark.unit


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.local:8000/v1",
    )


def _ok_response(
    content: str, *, prompt_tokens: int = 5, completion_tokens: int = 3
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-x",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def test_openai_compat_generate_returns_text_and_tokens() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url.path
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_ok_response("rewritten output"))

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
        client=_client(handler),
    )

    result = await backend.generate("input text", max_tokens=64, temperature=0.0)

    assert result.text == "rewritten output"
    assert result.tokens_in == 5
    assert result.tokens_out == 3
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"] == [{"role": "user", "content": "input text"}]
    assert "authorization" not in {k.lower() for k in captured["headers"]}


async def test_openai_compat_generate_includes_bearer_when_api_key_set() -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(200, json=_ok_response("ok"))

    backend = OpenAICompatBackend(
        name="openai_compat",
        endpoint="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4.5",
        api_key="or-test-key",
        client=_client(handler),
    )

    await backend.generate("x", max_tokens=4, temperature=0.0)

    assert captured_headers["authorization"] == "Bearer or-test-key"


async def test_openai_compat_generate_timeout_raises_generation_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(GenerationTimeoutError, match="timed out"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_generate_connect_error_raises_backend_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = OpenAICompatBackend(
        name="llama_cpp",
        endpoint="http://llama.local:8080/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(BackendUnavailableError, match="unreachable"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_generate_non_200_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="status 500"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_generate_non_json_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="non-JSON"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_generate_missing_choices_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="choices"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_generate_missing_content_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]})

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="content"):
        await backend.generate("x", max_tokens=4, temperature=0.0)


async def test_openai_compat_health_returns_true_on_models_endpoint_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    health = await backend.health()

    assert health.healthy is True


async def test_openai_compat_health_returns_false_on_connect_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
        client=_client(handler),
    )

    health = await backend.health()

    assert health.healthy is False
    assert "nope" in (health.detail or "")


def test_openai_compat_cost_estimate_with_pricing_returns_dollar_amount() -> None:
    pricing = TokenPricing(in_per_million_usd=0.50, out_per_million_usd=1.50)
    backend = OpenAICompatBackend(
        name="openai_compat",
        endpoint="https://openrouter.ai/api/v1",
        model="m",
        api_key="x",
        pricing=pricing,
    )

    estimate = backend.cost_estimate(tokens_in=1_000_000, tokens_out=500_000)

    assert estimate == pytest.approx(0.50 + 0.75)


def test_openai_compat_cost_estimate_without_pricing_returns_none() -> None:
    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="m",
    )

    assert backend.cost_estimate(tokens_in=1, tokens_out=1) is None


def test_openai_compat_construction_rejects_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        OpenAICompatBackend(name="vllm", endpoint="", model="m")


def test_openai_compat_construction_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        OpenAICompatBackend(name="", endpoint="http://x", model="m")


def test_openai_compat_construction_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAICompatBackend(name="vllm", endpoint="http://x", model="")
