"""Unit tests for backend streaming adapters (P3-STR-01).

Each adapter projects its native streaming protocol (Ollama NDJSON,
Anthropic SDK events, OpenAI-compat SSE, LiteLLM async iterator) onto
the unified :class:`~transduce.backends.base.StreamChunk` union. The
tests below cover the happy path (text deltas + final usage) and the
transport-failure mappings for each adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicConnectionError
from litellm import exceptions as litellm_exceptions

from transduce.backends.anthropic import AnthropicBackend
from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
    StreamFinal,
    StreamTextDelta,
)
from transduce.backends.litellm_router import LiteLLMRouterBackend
from transduce.backends.ollama import OllamaBackend
from transduce.backends.openai_compat import OpenAICompatBackend

pytestmark = pytest.mark.unit


def _ndjson(chunks: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(chunk) for chunk in chunks) + "\n").encode("utf-8")


def _sse(events: list[dict[str, Any]], *, terminate: bool = True) -> bytes:
    parts = [f"data: {json.dumps(event)}\n\n" for event in events]
    if terminate:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Ollama streaming
# ---------------------------------------------------------------------------


def _ollama_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama.local:11434",
    )


async def test_ollama_stream_emits_text_deltas_then_final_usage() -> None:
    body = _ndjson(
        [
            {"response": "Hello", "done": False},
            {"response": " ", "done": False},
            {"response": "world", "done": False},
            {
                "response": "",
                "done": True,
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_ollama_client(handler),
    )

    chunks = [chunk async for chunk in backend.stream("hi", max_tokens=8, temperature=0.0)]

    text_deltas = [chunk for chunk in chunks if isinstance(chunk, StreamTextDelta)]
    finals = [chunk for chunk in chunks if isinstance(chunk, StreamFinal)]
    assert [delta.text for delta in text_deltas] == ["Hello", " ", "world"]
    assert len(finals) == 1
    assert finals[0].tokens_in == 12
    assert finals[0].tokens_out == 5


async def test_ollama_stream_non_200_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"model loading")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_ollama_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="status 500"):
        async for _ in backend.stream("hi", max_tokens=8, temperature=0.0):
            pass


async def test_ollama_stream_invalid_json_line_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json\n")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_ollama_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="non-JSON line"):
        async for _ in backend.stream("hi", max_tokens=8, temperature=0.0):
            pass


async def test_ollama_stream_connect_error_raises_backend_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        client=_ollama_client(handler),
    )

    with pytest.raises(BackendUnavailableError, match="ollama unreachable"):
        async for _ in backend.stream("hi", max_tokens=8, temperature=0.0):
            pass


async def test_ollama_stream_timeout_raises_generation_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("read timeout")

    backend = OllamaBackend(
        endpoint="http://ollama.local:11434",
        model="qwen2.5:14b",
        timeout_s=4.0,
        client=_ollama_client(handler),
    )

    with pytest.raises(GenerationTimeoutError, match="streaming timed out"):
        async for _ in backend.stream("hi", max_tokens=8, temperature=0.0):
            pass


async def test_ollama_stream_max_tokens_must_be_positive() -> None:
    backend = OllamaBackend(endpoint="http://ollama.local", model="qwen2.5:14b")

    with pytest.raises(ValueError, match="max_tokens"):
        async for _ in backend.stream("hi", max_tokens=0, temperature=0.0):
            pass


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------


@dataclass
class _FakeAnthropicTextStream:
    chunks: list[str]

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        for chunk in self.chunks:
            yield chunk


class _FakeAnthropicStream:
    def __init__(self, chunks: list[str], *, usage_in: int, usage_out: int) -> None:
        self.text_stream = _FakeAnthropicTextStream(chunks)
        self._usage_in = usage_in
        self._usage_out = usage_out

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        del exc_type, exc, tb

    async def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=self._usage_in, output_tokens=self._usage_out)
        )


class _FakeAnthropicMessages:
    def __init__(self, fake_stream: _FakeAnthropicStream) -> None:
        self._stream = fake_stream
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.calls.append(kwargs)
        return self._stream


class _FakeAnthropicClient:
    def __init__(self, fake_stream: _FakeAnthropicStream) -> None:
        self.messages = _FakeAnthropicMessages(fake_stream)

    async def close(self) -> None:
        return None


async def test_anthropic_stream_maps_text_delta_events_and_final_usage() -> None:
    fake_stream = _FakeAnthropicStream(["Hello", " ", "world"], usage_in=10, usage_out=4)
    client = _FakeAnthropicClient(fake_stream)
    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="test",
        client=client,  # type: ignore[arg-type]
    )

    chunks = [chunk async for chunk in backend.stream("hi", max_tokens=64, temperature=0.0)]

    assert [c.text for c in chunks if isinstance(c, StreamTextDelta)] == ["Hello", " ", "world"]
    finals = [c for c in chunks if isinstance(c, StreamFinal)]
    assert len(finals) == 1
    assert finals[0].tokens_in == 10
    assert finals[0].tokens_out == 4
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


class _RaisingAnthropicMessages:
    def stream(self, **kwargs: Any) -> Any:
        del kwargs
        raise AnthropicConnectionError(request=httpx.Request("POST", "http://x"))


class _RaisingAnthropicClient:
    def __init__(self) -> None:
        self.messages = _RaisingAnthropicMessages()

    async def close(self) -> None:
        return None


async def test_anthropic_stream_connect_error_raises_backend_unavailable() -> None:
    backend = AnthropicBackend(
        model="claude-haiku-4-5",
        api_key="test",
        client=_RaisingAnthropicClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(BackendUnavailableError, match="anthropic API unreachable"):
        async for _ in backend.stream("hi", max_tokens=64, temperature=0.0):
            pass


# ---------------------------------------------------------------------------
# OpenAI-compat streaming
# ---------------------------------------------------------------------------


def _openai_compat_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm.local:8000/v1",
    )


async def test_openai_compat_stream_parses_sse_chunks_and_usage() -> None:
    body = _sse(
        [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, content=body)

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
        client=_openai_compat_client(handler),
    )

    chunks = [chunk async for chunk in backend.stream("hi", max_tokens=64, temperature=0.0)]

    text_deltas = [c for c in chunks if isinstance(c, StreamTextDelta)]
    assert [d.text for d in text_deltas] == ["Hello", " world"]
    finals = [c for c in chunks if isinstance(c, StreamFinal)]
    assert len(finals) == 1
    assert finals[0].tokens_in == 8
    assert finals[0].tokens_out == 3


async def test_openai_compat_stream_omitted_usage_returns_zero_token_counts() -> None:
    body = _sse([{"choices": [{"delta": {"content": "Hi"}}]}])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OpenAICompatBackend(
        name="openrouter",
        endpoint="http://openrouter.test/v1",
        model="openrouter/claude-haiku",
        client=_openai_compat_client(handler),
    )

    chunks = [chunk async for chunk in backend.stream("hi", max_tokens=64, temperature=0.0)]

    finals = [c for c in chunks if isinstance(c, StreamFinal)]
    assert finals[0].tokens_in == 0
    assert finals[0].tokens_out == 0


async def test_openai_compat_stream_non_200_raises_generation_failed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"model not ready")

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="qwen",
        client=_openai_compat_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="status 503"):
        async for _ in backend.stream("hi", max_tokens=64, temperature=0.0):
            pass


async def test_openai_compat_stream_invalid_payload_raises_generation_failed() -> None:
    body = b"data: {not-json}\n\n"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OpenAICompatBackend(
        name="vllm",
        endpoint="http://vllm.local:8000/v1",
        model="qwen",
        client=_openai_compat_client(handler),
    )

    with pytest.raises(GenerationFailedError, match="non-JSON payload"):
        async for _ in backend.stream("hi", max_tokens=64, temperature=0.0):
            pass


# ---------------------------------------------------------------------------
# LiteLLM router streaming
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(
        self,
        *,
        content: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        delta = SimpleNamespace(content=content)
        choice = SimpleNamespace(delta=delta)
        self.choices = [choice]
        self.usage: SimpleNamespace | None
        if prompt_tokens is not None or completion_tokens is not None:
            self.usage = SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        else:
            self.usage = None


class _FakeAsyncIter:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[_FakeChunk]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[_FakeChunk]:
        for chunk in self._chunks:
            yield chunk


async def test_litellm_router_stream_passes_through_provider_chunks() -> None:
    chunks_in = [
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" world"),
        _FakeChunk(prompt_tokens=8, completion_tokens=3),
    ]

    async def fake_completion(**kwargs: Any) -> _FakeAsyncIter:
        assert kwargs["stream"] is True
        return _FakeAsyncIter(chunks_in)

    backend = LiteLLMRouterBackend(
        model="claude-haiku",
        api_key="test",
        completion=fake_completion,
    )

    chunks = [chunk async for chunk in backend.stream("hi", max_tokens=64, temperature=0.0)]

    text_deltas = [c for c in chunks if isinstance(c, StreamTextDelta)]
    assert [d.text for d in text_deltas] == ["Hello", " world"]
    finals = [c for c in chunks if isinstance(c, StreamFinal)]
    assert finals[0].tokens_in == 8
    assert finals[0].tokens_out == 3


async def test_litellm_router_stream_timeout_raises_generation_timeout() -> None:
    async def fake_completion(**kwargs: Any) -> Any:
        del kwargs
        raise litellm_exceptions.Timeout(
            message="timed out",
            model="claude-haiku",
            llm_provider="anthropic",
        )

    backend = LiteLLMRouterBackend(
        model="claude-haiku",
        api_key="test",
        completion=fake_completion,
    )

    with pytest.raises(GenerationTimeoutError, match="timed out"):
        async for _ in backend.stream("hi", max_tokens=64, temperature=0.0):
            pass


async def test_litellm_router_stream_connection_error_raises_backend_unavailable() -> None:
    async def fake_completion(**kwargs: Any) -> Any:
        del kwargs
        raise litellm_exceptions.APIConnectionError(
            message="conn refused",
            model="claude-haiku",
            llm_provider="anthropic",
        )

    backend = LiteLLMRouterBackend(
        model="claude-haiku",
        api_key="test",
        completion=fake_completion,
    )

    with pytest.raises(BackendUnavailableError, match="upstream unreachable"):
        async for _ in backend.stream("hi", max_tokens=64, temperature=0.0):
            pass
