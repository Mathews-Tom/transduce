"""OpenAI Chat Completions backend adapter (P3-BACK-02..04).

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
"""

from __future__ import annotations

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
    TokenPricing,
)

_CHAT_PATH: Final[str] = "/chat/completions"
_MODELS_PATH: Final[str] = "/models"

OpenAICompatProvider = str
"""Provider identifier baked onto :attr:`OpenAICompatBackend.name`."""


class OpenAICompatBackend:
    """OpenAI Chat Completions adapter for vLLM, llama.cpp, and cloud compat APIs."""

    capabilities = BackendCapabilities(streaming=False, json_mode=False, attention_output=False)

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


__all__ = ["OpenAICompatBackend", "OpenAICompatProvider"]
