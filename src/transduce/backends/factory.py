"""Backend dispatch table from ``BackendEntry`` to a wired :class:`Backend`.

The factory translates a single ``BackendEntry`` into the right
provider-specific adapter, threads operator-supplied pricing through
:class:`TokenPricing`, and wraps the result in
:class:`SemaphoreBackend` so the per-backend concurrency limit is
enforced uniformly. The CLI ``serve`` command calls
:func:`build_backend` for the operator-selected default backend; the
factory raises with a descriptive message when secrets or endpoints
are missing so misconfigurations surface at startup rather than mid-
request.
"""

from __future__ import annotations

import os

from transduce.backends.anthropic import AnthropicBackend
from transduce.backends.base import Backend, TokenPricing
from transduce.backends.concurrency import SemaphoreBackend
from transduce.backends.litellm_router import LiteLLMRouterBackend
from transduce.backends.ollama import OllamaBackend
from transduce.backends.openai_compat import OpenAICompatBackend
from transduce.config.schema import BackendEntry


class BackendFactoryError(RuntimeError):
    """Raised when the factory cannot build a backend from the entry."""


def build_backend(
    entry: BackendEntry,
    *,
    env: dict[str, str] | None = None,
) -> Backend:
    """Build a wired :class:`Backend` from a configuration ``entry``.

    Args:
        entry: The validated :class:`BackendEntry` selected by the
            caller (typically ``config.backends.default``).
        env: Mapping consulted for ``api_key_env`` lookups. Defaults to
            ``os.environ``. Tests inject a fixture mapping to avoid
            depending on real shell state.

    Returns:
        :class:`SemaphoreBackend` wrapping the provider-specific
        adapter sized at ``entry.concurrency_limit``.

    Raises:
        BackendFactoryError: configured ``api_key_env`` variable is
            unset, or the provider name is not yet wired.
    """
    environ = env if env is not None else dict(os.environ)
    pricing = _build_pricing(entry)
    inner = _build_inner(entry, environ=environ, pricing=pricing)
    return SemaphoreBackend(
        inner=inner,
        backend_id=entry.id,
        limit=entry.concurrency_limit,
    )


def _build_inner(
    entry: BackendEntry,
    *,
    environ: dict[str, str],
    pricing: TokenPricing | None,
) -> Backend:
    if entry.provider == "ollama":
        if entry.endpoint is None:  # pragma: no cover — validator enforces this
            raise BackendFactoryError(f"backend {entry.id!r}: ollama provider requires an endpoint")
        return OllamaBackend(
            endpoint=entry.endpoint,
            model=entry.model,
            timeout_s=entry.timeout_s,
        )
    if entry.provider == "anthropic":
        api_key = _require_env(entry, environ)
        return AnthropicBackend(
            model=entry.model,
            api_key=api_key,
            timeout_s=entry.timeout_s,
            pricing=pricing,
        )
    if entry.provider in ("openai_compat", "vllm", "llama_cpp"):
        if entry.endpoint is None:  # pragma: no cover — validator enforces this
            raise BackendFactoryError(
                f"backend {entry.id!r}: {entry.provider!r} requires an endpoint"
            )
        optional_api_key: str | None = environ.get(entry.api_key_env) if entry.api_key_env else None
        return OpenAICompatBackend(
            name=entry.provider,
            endpoint=entry.endpoint,
            model=entry.model,
            api_key=optional_api_key,
            timeout_s=entry.timeout_s,
            pricing=pricing,
        )
    if entry.provider == "litellm":
        api_key = _require_env(entry, environ)
        return LiteLLMRouterBackend(
            model=entry.model,
            api_key=api_key,
            timeout_s=entry.timeout_s,
            pricing=pricing,
        )
    raise BackendFactoryError(  # pragma: no cover — Literal forbids other values
        f"backend {entry.id!r}: provider {entry.provider!r} not handled"
    )


def _build_pricing(entry: BackendEntry) -> TokenPricing | None:
    in_rate = entry.cost_in_per_million_usd
    out_rate = entry.cost_out_per_million_usd
    if in_rate is None or out_rate is None:
        return None
    return TokenPricing(in_per_million_usd=in_rate, out_per_million_usd=out_rate)


def _require_env(entry: BackendEntry, environ: dict[str, str]) -> str:
    if entry.api_key_env is None:  # pragma: no cover — validator enforces this
        raise BackendFactoryError(
            f"backend {entry.id!r}: provider {entry.provider!r} requires api_key_env"
        )
    api_key = environ.get(entry.api_key_env)
    if not api_key:
        raise BackendFactoryError(
            f"backend {entry.id!r}: env var {entry.api_key_env!r} is unset; "
            "set it in the deployment environment before starting the service"
        )
    return api_key


__all__ = ["BackendFactoryError", "build_backend"]
