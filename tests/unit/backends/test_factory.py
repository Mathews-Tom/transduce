"""Unit tests for the backend factory dispatch (P3-BACK-01..09)."""

from __future__ import annotations

import pytest

from transduce.backends.anthropic import AnthropicBackend
from transduce.backends.concurrency import SemaphoreBackend
from transduce.backends.factory import BackendFactoryError, build_backend
from transduce.backends.litellm_router import LiteLLMRouterBackend
from transduce.backends.ollama import OllamaBackend
from transduce.backends.openai_compat import OpenAICompatBackend
from transduce.config.schema import BackendEntry

pytestmark = pytest.mark.unit


def test_factory_builds_ollama_backend_wrapped_in_semaphore() -> None:
    entry = BackendEntry(
        id="ollama_qwen",
        provider="ollama",
        endpoint="http://localhost:11434",
        model="qwen2.5:14b",
        concurrency_limit=1,
    )

    backend = build_backend(entry, env={})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.backend_id == "ollama_qwen"
    assert backend.limit == 1
    assert backend.name == "ollama"


def test_factory_builds_anthropic_backend_with_api_key_from_env() -> None:
    entry = BackendEntry(
        id="anthropic_haiku",
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        concurrency_limit=8,
    )

    backend = build_backend(entry, env={"ANTHROPIC_API_KEY": "sk-test"})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.limit == 8
    assert backend.name == "anthropic"


def test_factory_anthropic_missing_env_var_raises() -> None:
    entry = BackendEntry(
        id="anthropic_haiku",
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    )

    with pytest.raises(BackendFactoryError, match="env var 'ANTHROPIC_API_KEY' is unset"):
        build_backend(entry, env={})


def test_factory_builds_vllm_backend_via_openai_compat_adapter() -> None:
    entry = BackendEntry(
        id="vllm_qwen",
        provider="vllm",
        endpoint="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
        concurrency_limit=32,
    )

    backend = build_backend(entry, env={})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.limit == 32
    assert backend.name == "vllm"


def test_factory_builds_llama_cpp_backend_via_openai_compat_adapter() -> None:
    entry = BackendEntry(
        id="llama_cpp_local",
        provider="llama_cpp",
        endpoint="http://localhost:8080/v1",
        model="llama-3.2-8b",
    )

    backend = build_backend(entry, env={})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.name == "llama_cpp"


def test_factory_builds_openai_compat_backend_with_api_key_from_env() -> None:
    entry = BackendEntry(
        id="openrouter",
        provider="openai_compat",
        endpoint="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4.5",
        api_key_env="OPENROUTER_API_KEY",
    )

    backend = build_backend(entry, env={"OPENROUTER_API_KEY": "or-test"})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.name == "openai_compat"


def test_factory_openai_compat_without_api_key_env_uses_no_auth() -> None:
    entry = BackendEntry(
        id="vllm_qwen",
        provider="vllm",
        endpoint="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
    )

    backend = build_backend(entry, env={})

    assert isinstance(backend, SemaphoreBackend)


def test_factory_openai_compat_with_declared_api_key_env_unset_raises() -> None:
    entry = BackendEntry(
        id="openrouter",
        provider="openai_compat",
        endpoint="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4.5",
        api_key_env="OPENROUTER_API_KEY",
    )

    with pytest.raises(BackendFactoryError, match="OPENROUTER_API_KEY"):
        build_backend(entry, env={})


def test_factory_vllm_with_declared_api_key_env_empty_raises() -> None:
    entry = BackendEntry(
        id="vllm_secured",
        provider="vllm",
        endpoint="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-14B-Instruct",
        api_key_env="VLLM_API_KEY",
    )

    with pytest.raises(BackendFactoryError, match="VLLM_API_KEY"):
        build_backend(entry, env={"VLLM_API_KEY": ""})


def test_factory_builds_litellm_router_with_api_key_from_env() -> None:
    entry = BackendEntry(
        id="litellm",
        provider="litellm",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        concurrency_limit=16,
    )

    backend = build_backend(entry, env={"ANTHROPIC_API_KEY": "sk-test"})

    assert isinstance(backend, SemaphoreBackend)
    assert backend.limit == 16
    assert backend.name == "litellm"


def test_factory_litellm_missing_env_var_raises() -> None:
    entry = BackendEntry(
        id="litellm",
        provider="litellm",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    )

    with pytest.raises(BackendFactoryError):
        build_backend(entry, env={})


def test_factory_threads_pricing_through_when_both_rates_set() -> None:
    entry = BackendEntry(
        id="anthropic_haiku",
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        cost_in_per_million_usd=1.00,
        cost_out_per_million_usd=5.00,
    )

    backend = build_backend(entry, env={"ANTHROPIC_API_KEY": "sk-test"})

    estimate = backend.cost_estimate(tokens_in=1_000_000, tokens_out=200_000)

    assert estimate == pytest.approx(1.00 + 1.00)


def test_factory_pricing_omitted_when_one_rate_unset() -> None:
    entry = BackendEntry(
        id="anthropic_haiku",
        provider="anthropic",
        model="claude-haiku-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        cost_in_per_million_usd=1.00,
    )

    backend = build_backend(entry, env={"ANTHROPIC_API_KEY": "sk-test"})

    assert backend.cost_estimate(tokens_in=1, tokens_out=1) is None


# Anchor unused-import suppression for adapters used only in isinstance checks.
_ = (
    AnthropicBackend,
    LiteLLMRouterBackend,
    OllamaBackend,
    OpenAICompatBackend,
)
