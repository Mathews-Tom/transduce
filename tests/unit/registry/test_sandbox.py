"""Unit tests for the subprocess sandbox (P2-PLG-05)."""

from __future__ import annotations

import os

import pytest

from transduce.registry.sandbox import (
    SandboxError,
    filter_environment,
    run_in_sandbox,
)

pytestmark = pytest.mark.unit


def test_sandbox_strips_anthropic_api_key() -> None:
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "PATH": "/usr/bin",
    }

    filtered = filter_environment(env, ["ANTHROPIC_API_KEY"])

    assert "ANTHROPIC_API_KEY" not in filtered
    assert filtered["PATH"] == "/usr/bin"


def test_sandbox_strips_glob_pattern_secrets() -> None:
    env = {
        "GITHUB_TOKEN": "gho-test",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "OPENAI_API_KEY": "sk-test",
        "PATH": "/usr/bin",
    }

    filtered = filter_environment(env, ["*_TOKEN", "*_SECRET", "*_KEY"])

    assert "GITHUB_TOKEN" not in filtered
    assert "AWS_SECRET_ACCESS_KEY" not in filtered
    assert "OPENAI_API_KEY" not in filtered
    assert filtered["PATH"] == "/usr/bin"


def test_sandbox_filter_with_empty_pattern_list_returns_copy() -> None:
    env = {"FOO": "bar"}

    filtered = filter_environment(env, [])

    assert filtered == env
    assert filtered is not env


def _read_env_var(name: str) -> str | None:
    return os.environ.get(name)


def _add(a: int, b: int) -> int:
    return a + b


def _boom() -> None:
    raise RuntimeError("scorer crashed")


@pytest.mark.skipif(os.name == "nt", reason="multiprocessing spawn semantics differ on Windows CI")
def test_sandbox_subprocess_isolation() -> None:
    os.environ["TRANSDUCE_TEST_LEAK"] = "leaked-value"
    try:
        result = run_in_sandbox(
            _read_env_var,
            ["TRANSDUCE_TEST_LEAK"],
            strip_env_vars=["TRANSDUCE_TEST_LEAK"],
        )
    finally:
        del os.environ["TRANSDUCE_TEST_LEAK"]

    assert result is None


@pytest.mark.skipif(os.name == "nt", reason="multiprocessing spawn semantics differ on Windows CI")
def test_sandbox_returns_target_value() -> None:
    assert run_in_sandbox(_add, [2, 3], strip_env_vars=[]) == 5


@pytest.mark.skipif(os.name == "nt", reason="multiprocessing spawn semantics differ on Windows CI")
def test_sandbox_propagates_worker_error() -> None:
    with pytest.raises(SandboxError, match="scorer crashed"):
        run_in_sandbox(_boom, [], strip_env_vars=[])


def test_sandbox_invalid_budget_rejected() -> None:
    with pytest.raises(ValueError, match="budget_seconds"):
        run_in_sandbox(_read_env_var, ["X"], strip_env_vars=[], budget_seconds=0.0)
