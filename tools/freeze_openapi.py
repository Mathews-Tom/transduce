"""Freeze the v1 OpenAPI schema for the transduce HTTP API.

Run with ``uv run python tools/freeze_openapi.py``. Generates
``docs/api/openapi-v1.json`` from the Litestar app — using a stub
backend and stub scorer so the freeze works on any developer machine
without an Ollama server, fastembed weights, or spaCy model.

The frozen schema is the v1 contract. Any breaking change to a route,
request schema, or response schema fails the
``test_openapi_v1_frozen_matches_app_routes`` regression gate; the gate
forces a deliberate bump (or freeze update) rather than silent drift.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from transduce.api.app import create_app
from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    GenerationResult,
    StreamChunk,
    StreamFinal,
)
from transduce.config.schema import (
    BackendEntry,
    BackendsConfig,
    BudgetConfig,
    Config,
    LanguageConfig,
    ServiceConfig,
    VerificationConfig,
)
from transduce.verification.base import Scorer, ScoreResult

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "openapi-v1.json"


class _StubBackend:
    name = "ollama"
    model = "qwen2.5:1.5b"
    capabilities = BackendCapabilities(streaming=True)

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> GenerationResult:
        del prompt, max_tokens, temperature
        return GenerationResult(text="ok", tokens_in=2, tokens_out=2)

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamChunk]:
        del prompt, max_tokens, temperature
        yield StreamFinal(tokens_in=2, tokens_out=2)

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        del tokens_in, tokens_out
        return None


class _StubScorer:
    name = "cosine_similarity"

    def score(self, original: str, candidate: str) -> ScoreResult:
        del original, candidate
        return ScoreResult(name=self.name, value=1.0, verdict="accept", details={})


def build_schema() -> dict[str, Any]:
    config = Config(
        service=ServiceConfig(),
        backends=BackendsConfig(
            default="ollama_qwen",
            registry=[
                BackendEntry(
                    id="ollama_qwen",
                    provider="ollama",
                    endpoint="http://ollama.local:11434",
                    model="qwen2.5:1.5b",
                    model_size_b=14.0,
                )
            ],
        ),
        verification=VerificationConfig(),
        budget=BudgetConfig(),
        language=LanguageConfig(),
    )
    scorers: list[Scorer] = [_StubScorer()]
    app = create_app(config, backend=_StubBackend(), scorers=scorers)
    if app.openapi_schema is None:
        raise RuntimeError("openapi_schema is unset on the Litestar app")
    schema = app.openapi_schema.to_schema()
    if not isinstance(schema, dict):
        raise RuntimeError("openapi_schema.to_schema() did not return a dict")
    return schema


def main() -> None:
    schema = build_schema()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote OpenAPI schema to {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
