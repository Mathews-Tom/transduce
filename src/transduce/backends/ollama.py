"""Ollama backend adapter (P1-BACK-02, P3-STR-01).

Wraps the Ollama HTTP API at ``/api/generate`` and ``/api/tags`` per
docs/system-design.md §Backend Adapter Layer. Maps transport failures
onto the backend exception hierarchy so the pipeline orchestrator can
emit stable error codes. The streaming variant returns NDJSON-encoded
text deltas with a final usage line; the adapter projects them onto
the unified :class:`~transduce.backends.base.StreamChunk` union.
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
)

_DEFAULT_GENERATE_PATH: Final[str] = "/api/generate"
_DEFAULT_HEALTH_PATH: Final[str] = "/api/tags"


class OllamaBackend:
    """Backend adapter targeting an Ollama server."""

    name = "ollama"
    capabilities = BackendCapabilities(streaming=True, json_mode=False, attention_output=False)

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

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream an Ollama generation as NDJSON-decoded chunks (P3-STR-01).

        The Ollama ``/api/generate`` endpoint with ``stream: true``
        emits one JSON object per line: each non-final line carries a
        ``response`` token, the final line carries ``done: true`` plus
        ``prompt_eval_count`` / ``eval_count`` for the usage totals.
        The adapter forwards every non-empty token as a
        :class:`StreamTextDelta` and closes the stream with a
        :class:`StreamFinal` whose token counts come from the final
        line.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        client = self._resolve_client()
        tokens_in = 0
        tokens_out = 0
        try:
            async with client.stream("POST", _DEFAULT_GENERATE_PATH, json=payload) as response:
                if response.status_code != httpx.codes.OK:
                    body_text = (await response.aread()).decode("utf-8", errors="replace")
                    raise GenerationFailedError(
                        f"ollama returned status {response.status_code}: {body_text}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError as exc:
                        raise GenerationFailedError(
                            f"ollama returned non-JSON line: {exc}"
                        ) from exc
                    response_chunk = data.get("response")
                    if isinstance(response_chunk, str) and response_chunk:
                        yield StreamTextDelta(text=response_chunk)
                    if data.get("done"):
                        tokens_in = _coerce_token_count(data.get("prompt_eval_count"))
                        tokens_out = _coerce_token_count(data.get("eval_count"))
        except httpx.TimeoutException as exc:
            raise GenerationTimeoutError(
                f"ollama streaming timed out after {self._timeout_s}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise BackendUnavailableError(f"ollama unreachable at {self._endpoint}: {exc}") from exc
        yield StreamFinal(tokens_in=tokens_in, tokens_out=tokens_out)

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

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        """Local Ollama runs on operator hardware; per-token USD cost is undefined."""
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError("token counts must be non-negative")
        return None


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
