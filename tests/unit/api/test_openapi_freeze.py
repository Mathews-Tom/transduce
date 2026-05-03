"""Regression test for the frozen v1 OpenAPI schema.

The frozen schema lives at ``docs/api/openapi-v1.json``. Any breaking
change to a route, request schema, or response schema fails this test;
the gate forces a deliberate freeze update via
``uv run python tools/freeze_openapi.py`` rather than silent drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_PATH = REPO_ROOT / "docs" / "api" / "openapi-v1.json"
_FREEZE_SCRIPT = REPO_ROOT / "tools" / "freeze_openapi.py"


def _load_build_schema() -> Any:
    """Import ``tools/freeze_openapi.py`` by file path.

    The ``tools/`` directory ships scripts, not a package, so it is
    not on ``sys.path``. Importing by spec keeps the test gate close
    to the freeze script without forcing a packaging restructure.
    """
    spec = importlib.util.spec_from_file_location("transduce_freeze_openapi", _FREEZE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {_FREEZE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_schema


def test_openapi_v1_frozen_matches_app_routes() -> None:
    build_schema = _load_build_schema()
    actual = build_schema()
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    assert actual == frozen, (
        "OpenAPI schema diverged from the frozen v1 contract. "
        "Run `uv run python tools/freeze_openapi.py` and review the diff "
        "before committing — the freeze update is the deliberate breaking "
        "change signal."
    )


def test_openapi_v1_frozen_includes_streaming_endpoint() -> None:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    assert "/v1/transform/stream" in frozen["paths"], (
        "frozen OpenAPI must include the advisory streaming endpoint"
    )


def test_openapi_v1_frozen_includes_core_paths() -> None:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    expected = {
        "/v1/transform",
        "/v1/transform/stream",
        "/v1/modes",
        "/v1/backends",
        "/v1/scorers",
        "/healthz",
        "/readyz",
        "/metrics",
    }
    actual = set(frozen["paths"].keys())

    missing = expected - actual
    assert not missing, f"frozen schema missing core paths: {sorted(missing)}"
