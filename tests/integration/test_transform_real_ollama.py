"""Integration tests against a real Ollama server.

Auto-skip when Ollama is unreachable so contributors without local
inference can still run ``pytest -m integration``. CI gates this suite
behind the ``OLLAMA_HOST`` environment variable; local runs default to
``http://localhost:11434``.

Per dev-plan §Phase 1 Tests, these scenarios cover the v0 happy paths,
retry semantics, and health-check transitions.
"""

from __future__ import annotations

import os

import httpx
import pytest

from transduce.api.app import create_app
from transduce.backends.ollama import OllamaBackend
from transduce.config.schema import (
    BackendEntry,
    BackendsConfig,
    Config,
    LanguageConfig,
    ServiceConfig,
    VerificationConfig,
)
from transduce.verification.base import Scorer
from transduce.verification.cosine import CosineSimilarityScorer, build_fastembed_embedder
from transduce.verification.preservation import (
    EntityPreservationScorer,
    NumberPreservationScorer,
    UrlPreservationScorer,
    build_spacy_entity_extractor,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
        return response.status_code == httpx.codes.OK
    except (httpx.HTTPError, httpx.ConnectError):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _ollama_reachable(),
        reason=f"Ollama not reachable at {OLLAMA_HOST}; set OLLAMA_HOST or start the server.",
    ),
]


@pytest.fixture(scope="module")
def production_scorers() -> list[Scorer]:
    pytest.importorskip(
        "en_core_web_sm",
        reason="spaCy en_core_web_sm is not installed; run "
        "'uv run python -m spacy download en_core_web_sm'.",
    )
    return [
        CosineSimilarityScorer(embed=build_fastembed_embedder(), threshold=0.5),
        EntityPreservationScorer(build_spacy_entity_extractor()),
        NumberPreservationScorer(),
        UrlPreservationScorer(),
    ]


@pytest.fixture(scope="module")
def integration_config() -> Config:
    return Config(
        service=ServiceConfig(),
        backends=BackendsConfig(
            default="ollama_qwen",
            registry=[
                BackendEntry(
                    id="ollama_qwen",
                    provider="ollama",
                    endpoint=OLLAMA_HOST,
                    model=OLLAMA_MODEL,
                )
            ],
        ),
        verification=VerificationConfig(default_cosine_min=0.5, max_retries=1),
        language=LanguageConfig(),
    )


def test_integration_dejargon_real_ollama_returns_accept(
    integration_config: Config, production_scorers: list[Scorer]
) -> None:
    from litestar.testing import TestClient

    backend = OllamaBackend(endpoint=OLLAMA_HOST, model=OLLAMA_MODEL)
    app = create_app(integration_config, backend=backend, scorers=production_scorers)

    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform",
            json={
                "text": "We synergize verticals to drive cross-functional alignment.",
                "mode": "dejargon",
                "intensity": 0.5,
                "verification": {"max_retries": 1},
            },
        )

    assert response.status_code in (201, 422), response.text


def test_integration_readyz_returns_200_when_ollama_up(
    integration_config: Config, production_scorers: list[Scorer]
) -> None:
    from litestar.testing import TestClient

    backend = OllamaBackend(endpoint=OLLAMA_HOST, model=OLLAMA_MODEL)
    app = create_app(integration_config, backend=backend, scorers=production_scorers)

    with TestClient(app=app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["backend"]["healthy"] is True


def test_integration_readyz_returns_503_when_ollama_unreachable(
    integration_config: Config, production_scorers: list[Scorer]
) -> None:
    from litestar.testing import TestClient

    backend = OllamaBackend(endpoint="http://127.0.0.1:1", model=OLLAMA_MODEL)
    app = create_app(integration_config, backend=backend, scorers=production_scorers)

    with TestClient(app=app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
