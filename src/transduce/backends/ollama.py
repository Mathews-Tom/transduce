"""Ollama backend adapter (P1-BACK-02).

Wraps the Ollama HTTP API at ``/api/generate`` and ``/api/tags`` per
docs/system-design.md §Backend Adapter Layer. Maps transport failures
onto the backend exception hierarchy so the pipeline orchestrator can
emit stable error codes.
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
)

_DEFAULT_GENERATE_PATH: Final[str] = "/api/generate"
_DEFAULT_HEALTH_PATH: Final[str] = "/api/tags"


class OllamaBackend:
    """Backend adapter targeting an Ollama server."""

    name = "ollama"
    capabilities = BackendCapabilities(streaming=False, json_mode=False, attention_output=False)

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("OllamaBackend requires a non-empty endpoint")
        if not model:
            raise ValueError("OllamaBackend requires a non-empty model name")
        self._endpoint = endpoint.rstrip("/")
        self.model = model
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OllamaBackend:
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
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        client = self._resolve_client()
        try:
            response = await client.post(_DEFAULT_GENERATE_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(
                f"ollama generation timed out after {self._timeout_s}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(f"ollama unreachable at {self._endpoint}: {exc}") from exc

        if response.status_code != httpx.codes.OK:
            raise GenerationFailedError(
                f"ollama returned status {response.status_code}: {response.text}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise GenerationFailedError(f"ollama returned non-JSON response: {exc}") from exc

        return _to_generation_result(body)

    async def health(self) -> BackendHealth:
        client = self._resolve_client()
        try:
            response = await client.get(_DEFAULT_HEALTH_PATH)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            return BackendHealth(healthy=False, detail=str(exc))
        if response.status_code != httpx.codes.OK:
            return BackendHealth(
                healthy=False,
                detail=f"ollama returned status {response.status_code}",
            )
        return BackendHealth(healthy=True)


def _to_generation_result(payload: dict[str, Any]) -> GenerationResult:
    """Convert an Ollama response body into a typed ``GenerationResult``."""
    text = payload.get("response")
    if not isinstance(text, str):
        raise GenerationFailedError("ollama response missing required string field 'response'")
    tokens_in = _coerce_token_count(payload.get("prompt_eval_count"))
    tokens_out = _coerce_token_count(payload.get("eval_count"))
    return GenerationResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out)


def _coerce_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    raise GenerationFailedError(f"ollama returned non-integer token count: {value!r}")
