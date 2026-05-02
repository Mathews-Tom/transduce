"""LiteLLM router meta-backend (P3-BACK-05).

LiteLLM resolves a model alias to the upstream provider and dispatches
the call. The router itself does not own a connection — its health
posture is "the alias is recognised and the upstream credentials are
configured", not "the upstream is reachable". The dev plan documents
LiteLLM as a routing meta-backend, not a probe target; operators who
want true upstream health probes configure the upstream-specific
backend (anthropic, openai_compat) alongside the router.

Pricing follows the operator-supplied :class:`TokenPricing` pattern.
LiteLLM ships per-model price tables, but those drift over time and we
already accept the price-override fields on ``BackendEntry``; treating
operator config as the source of truth keeps cost accounting honest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import litellm
from litellm import exceptions as litellm_exceptions

from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    BackendUnavailableError,
    GenerationFailedError,
    GenerationResult,
    GenerationTimeoutError,
    TokenPricing,
)

CompletionFn = Callable[..., Any]
"""Callable that mirrors :func:`litellm.acompletion`; injectable for tests."""


class LiteLLMRouterBackend:
    """Backend adapter that delegates dispatch to the LiteLLM router."""

    name = "litellm"
    capabilities = BackendCapabilities(streaming=False, json_mode=False, attention_output=False)

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_s: float = 60.0,
        pricing: TokenPricing | None = None,
        completion: CompletionFn | None = None,
    ) -> None:
        if not model:
            raise ValueError("LiteLLMRouterBackend requires a non-empty model alias")
        if not api_key:
            raise ValueError("LiteLLMRouterBackend requires a non-empty api_key")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._pricing = pricing
        self._completion = completion or litellm.acompletion

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        try:
            response = await self._completion(
                model=self.model,
                api_key=self._api_key,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=self._timeout_s,
            )
        except litellm_exceptions.Timeout as exc:
            raise GenerationTimeoutError(
                f"litellm router timed out after {self._timeout_s}s"
            ) from exc
        except litellm_exceptions.APIConnectionError as exc:
            raise BackendUnavailableError(f"litellm upstream unreachable: {exc}") from exc
        except litellm_exceptions.APIError as exc:
            raise GenerationFailedError(f"litellm router API error: {exc}") from exc

        return _to_generation_result(response)

    async def health(self) -> BackendHealth:
        """Validate the alias resolves to a known provider; the router has no socket."""
        try:
            provider_info = litellm.get_llm_provider(model=self.model)
        except litellm_exceptions.BadRequestError as exc:
            return BackendHealth(healthy=False, detail=f"unknown model alias: {exc}")
        except (litellm_exceptions.NotFoundError, ValueError) as exc:
            return BackendHealth(healthy=False, detail=f"alias resolution failed: {exc}")
        provider_name = _extract_provider_name(provider_info)
        return BackendHealth(healthy=True, detail=f"alias routes to {provider_name}")

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        if self._pricing is None:
            return None
        return self._pricing.estimate(tokens_in=tokens_in, tokens_out=tokens_out)


def _extract_provider_name(provider_info: Any) -> str:
    """``litellm.get_llm_provider`` returns ``(model, provider, dynamic_api_key, api_base)``."""
    if isinstance(provider_info, tuple) and len(provider_info) >= 2:
        candidate = provider_info[1]
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def _to_generation_result(response: Any) -> GenerationResult:
    """Project a LiteLLM ``ModelResponse`` into our typed result.

    LiteLLM's response object follows the OpenAI Chat Completions shape:
    ``response.choices[0].message.content`` carries the text and
    ``response.usage`` carries token counts.
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise GenerationFailedError("litellm response missing 'choices' list")
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        raise GenerationFailedError("litellm choices[0].message is missing")
    text = getattr(message, "content", None)
    if not isinstance(text, str):
        raise GenerationFailedError("litellm choices[0].message.content missing or not a string")
    usage = getattr(response, "usage", None)
    tokens_in = _coerce_token_count(getattr(usage, "prompt_tokens", None))
    tokens_out = _coerce_token_count(getattr(usage, "completion_tokens", None))
    return GenerationResult(text=text, tokens_in=tokens_in, tokens_out=tokens_out)


def _coerce_token_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    raise GenerationFailedError(f"litellm returned non-integer token count: {value!r}")


__all__ = ["CompletionFn", "LiteLLMRouterBackend"]
