"""Subprocess sandbox for Python-plugin scorers (P2-PLG-05).

Python-plugin scorers run in an isolated ``multiprocessing.Process``
with the parent's environment filtered through ``strip_env_vars``. The
filter accepts both exact variable names and glob patterns (``*_TOKEN``,
``*_SECRET``). The default deny list mirrors the reference allowlist in
``transduce.example.yaml`` and the worked example in
``docs/system-design.md`` §Mode Registry.

The sandbox is opt-in: modes that ship a TOML manifest with no Python
scorer never go through the sandbox. Modes that declare a Python scorer
must register it via ``transduce.scorers`` (kept disabled-by-default in
v0.5; the enforcement gate lives in the allowlist loader).
"""

from __future__ import annotations

import fnmatch
import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import Any, Final

_DEFAULT_BUDGET_SECONDS: Final[float] = 5.0
"""POSIX uses ``fork`` so worker processes inherit the parent's
``sys.path`` and import state. Windows has no ``fork`` and falls
back to ``spawn``."""


def filter_environment(env: dict[str, str], strip_patterns: Sequence[str]) -> dict[str, str]:
    """Return a copy of ``env`` with any matching variable removed.

    Patterns may be exact names (``ANTHROPIC_API_KEY``) or fnmatch globs
    (``*_TOKEN``, ``*_SECRET``). Matching is case-sensitive against the
    variable names.
    """
    if not strip_patterns:
        return dict(env)
    filtered: dict[str, str] = {}
    for key, value in env.items():
        if any(fnmatch.fnmatchcase(key, pattern) for pattern in strip_patterns):
            continue
        filtered[key] = value
    return filtered


class SandboxError(RuntimeError):
    """Raised when the sandboxed worker fails or exceeds its budget."""


def run_in_sandbox(
    target: Callable[..., Any],
    args: Sequence[Any],
    *,
    strip_env_vars: Sequence[str],
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> Any:
    """Run ``target(*args)`` in a forked subprocess with a filtered environment.

    The parent uses a ``multiprocessing.Pipe`` to receive either a pickled
    return value or a pickled exception from the worker. The worker's
    ``os.environ`` is replaced with the filtered copy at startup so any
    code reading secrets via ``os.environ.get(...)`` sees absence rather
    than the parent's value.

    Raises:
        SandboxError: the worker exited non-zero, exceeded the budget,
            or returned an exception.
    """
    if budget_seconds <= 0:
        raise ValueError(f"budget_seconds must be positive, got {budget_seconds}")
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    filtered_env = filter_environment(dict(os.environ), strip_env_vars)
    context = (
        multiprocessing.get_context("spawn")
        if os.name == "nt"
        else multiprocessing.get_context("fork")
    )
    process = context.Process(
        target=_sandbox_entrypoint,
        args=(child_conn, target, tuple(args), filtered_env),
        daemon=True,
    )
    process.start()
    child_conn.close()
    try:
        if not parent_conn.poll(budget_seconds):
            process.terminate()
            process.join(timeout=1.0)
            raise SandboxError(f"sandbox worker exceeded {budget_seconds}s budget")
        payload = parent_conn.recv()
    finally:
        parent_conn.close()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    kind, body = payload
    if kind == "ok":
        return body
    if kind == "error":
        raise SandboxError(f"sandbox worker raised: {body}")
    raise SandboxError(f"sandbox worker returned unknown payload kind: {kind!r}")


def _sandbox_entrypoint(  # pragma: no cover - exercised by integration of run_in_sandbox
    pipe: Any,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    env: dict[str, str],
) -> None:
    os.environ.clear()
    os.environ.update(env)
    try:
        result = target(*args)
        pipe.send(("ok", result))
    except BaseException as exc:
        pipe.send(("error", repr(exc)))
    finally:
        pipe.close()


__all__ = [
    "SandboxError",
    "filter_environment",
    "run_in_sandbox",
]
