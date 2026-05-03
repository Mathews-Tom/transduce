"""Anthropic Messages API backend adapter (P3-BACK-01, P3-STR-01).

Wraps :class:`anthropic.AsyncAnthropic` and maps SDK exceptions onto
the transduce backend hierarchy so the pipeline orchestrator emits
stable error codes regardless of provider. The streaming variant
projects the SDK's ``messages.stream`` async context onto the unified
:class:`~transduce.backends.base.StreamChunk` union.

The adapter sends the rendered prompt as a single ``user`` message and
relies on the per-request spotlight fence inside the prompt template to
isolate untrusted user text. A subsequent commit can split the
instruction-hierarchy more aggressively (Wallace 2024) by routing mode
instructions into a system prompt and the fenced input into the user
message; the surface here keeps the change to the AnthropicBackend
constructor when it lands.

Pricing is operator-supplied via :class:`TokenPricing` (sourced from
``BackendEntry.cost_in_per_million_usd`` / ``cost_out_per_million_usd``)
rather than baked into a static table — model names live in operator
configuration per the project policy on hardcoded model references.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIError as AnthropicAPIError
from anthropic import APIStatusError as AnthropicStatusError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AsyncAnthropic

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


class AnthropicBackend:
    """Backend adapter targeting the Anthropic Messages API."""

    name = "anthropic"
    capabilities = BackendCapabilities(streaming=True, json_mode=False, attention_output=False)

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_s: float = 60.0,
        pricing: TokenPricing | None = None,
        client: AsyncAnthropic | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("AnthropicBackend requires a non-empty model name")
        if not api_key:
            raise ValueError("AnthropicBackend requires a non-empty api_key")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.model = model
        self._timeout_s = timeout_s
        self._pricing = pricing
        self._owns_client = client is None
        self._client = client or AsyncAnthropic(api_key=api_key, timeout=timeout_s)
        self._system_prompt = system_prompt

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt
        try:
            response = await self._client.messages.create(**kwargs)
        except AnthropicTimeoutError as exc:
            raise GenerationTimeoutError(
                f"anthropic generation timed out after {self._timeout_s}s"
            ) from exc
        except AnthropicConnectionError as exc:
            raise BackendUnavailableError(f"anthropic API unreachable: {exc}") from exc
        except AnthropicStatusError as exc:
            raise GenerationFailedError(
                f"anthropic returned status {exc.status_code}: {exc.message}"
            ) from exc
        except AnthropicAPIError as exc:
            raise GenerationFailedError(f"anthropic API error: {exc}") from exc

        return _to_generation_result(response)

    async def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream an Anthropic generation as text deltas plus a final usage event (P3-STR-01).

        Uses the SDK's ``messages.stream`` async context manager, which
        exposes a ``text_stream`` async iterator over text-delta events
        and a ``get_final_message`` coroutine that returns the
        accumulated message with usage totals once the stream closes.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for chunk in stream.text_stream:
                    if chunk:
                        yield StreamTextDelta(text=chunk)
                final = await stream.get_final_message()
        except AnthropicTimeoutError as exc:
            raise GenerationTimeoutError(
                f"anthropic streaming timed out after {self._timeout_s}s"
            ) from exc
        except AnthropicConnectionError as exc:
            raise BackendUnavailableError(f"anthropic API unreachable: {exc}") from exc
        except AnthropicStatusError as exc:
            raise GenerationFailedError(
                f"anthropic returned status {exc.status_code}: {exc.message}"
            ) from exc
        except AnthropicAPIError as exc:
            raise GenerationFailedError(f"anthropic API error: {exc}") from exc
        usage = getattr(final, "usage", None)
        tokens_in = _coerce_token_count(getattr(usage, "input_tokens", None))
        tokens_out = _coerce_token_count(getattr(usage, "output_tokens", None))
        yield StreamFinal(tokens_in=tokens_in, tokens_out=tokens_out)

    async def health(self) -> BackendHealth:
        """Health check: lightweight ``messages.count_tokens`` call against the model.

        The Messages API has no dedicated probe endpoint; counting tokens
        for a tiny payload is the lowest-cost call that exercises auth,
        network reachability, and model availability simultaneously.
        """
        try:
            await self._client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
            )
        except (AnthropicConnectionError, AnthropicTimeoutError) as exc:
            return BackendHealth(healthy=False, detail=str(exc))
        except AnthropicStatusError as exc:
            return BackendHealth(
                healthy=False, detail=f"anthropic returned status {exc.status_code}"
            )
        except AnthropicAPIError as exc:  # pragma: no cover — defensive catch-all
            return BackendHealth(healthy=False, detail=str(exc))
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        if self._pricing is None:
            return None
        return self._pricing.estimate(tokens_in=tokens_in, tokens_out=tokens_out)


def _to_generation_result(response: Any) -> GenerationResult:
    """Project an :class:`anthropic.types.Message` into our typed result.

    The SDK returns ``content`` as a list of typed blocks; we concatenate
    every ``text`` block in order. ``usage`` carries token counts.
    """
    content_blocks = getattr(response, "content", None)
    if not isinstance(content_blocks, list):
        raise GenerationFailedError("anthropic response missing 'content' list")
    text_parts: list[str] = []
    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                raise GenerationFailedError("anthropic text block missing 'text' field")
            text_parts.append(text)
    if not text_parts:
        raise GenerationFailedError("anthropic response contained no text blocks")
    usage = getattr(response, "usage", None)
    tokens_in = _coerce_token_count(getattr(usage, "input_tokens", None))
    tokens_out = _coerce_token_count(getattr(usage, "output_tokens", None))
    return GenerationResult(text="".join(text_parts), tokens_in=tokens_in, tokens_out=tokens_out)


def _coerce_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    raise GenerationFailedError(f"anthropic returned non-integer token count: {value!r}")


__all__ = ["AnthropicBackend"]
