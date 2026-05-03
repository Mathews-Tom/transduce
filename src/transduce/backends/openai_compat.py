"""OpenAI Chat Completions backend adapter (P3-BACK-02..04, P3-STR-01).

A single httpx-based implementation serves the three providers that
speak the OpenAI-compatible wire protocol on a configurable endpoint:
``vllm``, ``llama_cpp``, and the catch-all ``openai_compat`` (used
for OpenRouter, Together, Groq, fireworks.ai, etc.). The provider
identity is carried on ``name`` so observability spans and
``BackendInfo`` responses surface the operator's chosen provider
rather than a generic shared name.

Bearer authorization is optional. Local vLLM and llama-cpp typically
run unauthenticated; cloud OpenAI-compat endpoints require an API
key. The adapter sends the ``Authorization: Bearer <api_key>`` header
only when ``api_key`` is non-empty.

Pricing is operator-supplied via :class:`TokenPricing`; local
deployments leave it ``None`` so ``cost_estimate`` returns ``None``
and the budgeter records 0.0 per attempt while still bounding retries.

The streaming variant parses the standard OpenAI Server-Sent-Events
format (``data: {...}\\n\\n`` lines terminated by ``data: [DONE]``)
and forwards delta content as :class:`StreamTextDelta`. Servers that
emit ``stream_options: {include_usage: true}`` (vLLM, OpenRouter,
recent OpenAI) populate the final usage; older servers leave token
counts at zero, which the budgeter treats as the local-backend
``None`` price equivalent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Final

import httpx

from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    BackendUnavailableError,
    GenerationFailedError,
    GenerationResult,
    GenerationTimeoutError,
    StreamChunk,
    StreamFinal,
    StreamTextDelta,
    TokenPricing,
)

_CHAT_PATH: Final[str] = "/chat/completions"
_MODELS_PATH: Final[str] = "/models"

OpenAICompatProvider = str
"""Provider identifier baked onto :attr:`OpenAICompatBackend.name`."""


class OpenAICompatBackend:
    """OpenAI Chat Completions adapter for vLLM, llama.cpp, and cloud compat APIs."""

    capabilities = BackendCapabilities(streaming=True, json_mode=False, attention_output=False)

    def __init__(
        self,
        *,
        name: OpenAICompatProvider,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        pricing: TokenPricing | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not name:
            raise ValueError("OpenAICompatBackend requires a non-empty name")
        if not endpoint:
            raise ValueError("OpenAICompatBackend requires a non-empty endpoint")
        if not model:
            raise ValueError("OpenAICompatBackend requires a non-empty model name")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.name = name
        self.model = model
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._pricing = pricing
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OpenAICompatBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _resolve_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._endpoint, timeout=self._timeout_s)
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }
        client = self._resolve_client()
        try:
            response = await client.post(_CHAT_PATH, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(
                f"{self.name} generation timed out after {self._timeout_s}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(
                f"{self.name} unreachable at {self._endpoint}: {exc}"
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise GenerationFailedError(
                f"{self.name} returned status {response.status_code}: {response.text}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise GenerationFailedError(f"{self.name} returned non-JSON response: {exc}") from exc
        return _to_generation_result(body, provider=self.name)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a Chat Completions response as SSE-decoded chunks (P3-STR-01).

        Sends ``stream_options.include_usage: true`` so servers that
        support it populate the final ``usage`` block; the adapter
        falls back to zero token counts when the server omits usage
        rather than fabricating numbers.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
        }
        client = self._resolve_client()
        tokens_in = 0
        tokens_out = 0
        try:
            async with client.stream(
                "POST", _CHAT_PATH, json=payload, headers=self._headers()
            ) as response:
                if response.status_code != httpx.codes.OK:
                    body_text = (await response.aread()).decode("utf-8", errors="replace")
                    raise GenerationFailedError(
                        f"{self.name} returned status {response.status_code}: {body_text}"
                    )
                async for line in response.aiter_lines():
                    payload_str = _sse_payload(line)
                    if payload_str is None:
                        continue
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except ValueError as exc:
                        raise GenerationFailedError(
                            f"{self.name} streaming non-JSON payload: {exc}"
                        ) from exc
                    delta_text = _extract_delta_content(chunk)
                    if delta_text:
                        yield StreamTextDelta(text=delta_text)
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        tokens_in = _coerce_token_count(
                            usage.get("prompt_tokens"), provider=self.name
                        )
                        tokens_out = _coerce_token_count(
                            usage.get("completion_tokens"), provider=self.name
                        )
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(
                f"{self.name} streaming timed out after {self._timeout_s}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(
                f"{self.name} unreachable at {self._endpoint}: {exc}"
            ) from exc
        yield StreamFinal(tokens_in=tokens_in, tokens_out=tokens_out)

    async def health(self) -> BackendHealth:
        client = self._resolve_client()
        try:
            response = await client.get(_MODELS_PATH, headers=self._headers())
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            return BackendHealth(healthy=False, detail=str(exc))
        if response.status_code != httpx.codes.OK:
            return BackendHealth(
                healthy=False,
                detail=f"{self.name} returned status {response.status_code}",
            )
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        if self._pricing is None:
            return None
        return self._pricing.estimate(tokens_in=tokens_in, tokens_out=tokens_out)


def _to_generation_result(body: dict[str, Any], *, provider: str) -> GenerationResult:
    """Project an OpenAI Chat Completions response into our typed result."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GenerationFailedError(f"{provider} response missing 'choices' list")
    first = choices[0]
    if not isinstance(first, dict):
        raise GenerationFailedError(f"{provider} choices[0] is not a mapping")
    message = first.get("message")
    if not isinstance(message, dict):
        raise GenerationFailedError(f"{provider} choices[0].message is not a mapping")
    text = message.get("content")
    if not isinstance(text, str):
        raise GenerationFailedError(
            f"{provider} choices[0].message.content missing or not a string"
        )
    usage = body.get("usage") or {}
    if not isinstance(usage, dict):
        raise GenerationFailedError(f"{provider} usage field is not a mapping")
    tokens_in = _coerce_token_count(usage.get("prompt_tokens"), provider=provider)
    tokens_out = _coerce_token_count(usage.get("completion_tokens"), provider=provider)
    return GenerationResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out)


def _coerce_token_count(value: Any, *, provider: str) -> int:
    if value is None:
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    raise GenerationFailedError(f"{provider} returned non-integer token count: {value!r}")


def _sse_payload(line: str) -> str | None:
    """Return the payload portion of an SSE ``data: ...`` line, or ``None`` for skips.

    OpenAI-compatible streams emit blank separator lines and
    occasional ``event: ...`` lines; only ``data: ...`` lines carry
    JSON payloads. Returns ``None`` for non-data lines so the caller
    skips them without parsing.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _extract_delta_content(chunk: dict[str, Any]) -> str | None:
    """Pull ``choices[0].delta.content`` out of an OpenAI streaming chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    if isinstance(content, str) and content:
        return content
    return None


__all__ = ["OpenAICompatBackend", "OpenAICompatProvider"]
