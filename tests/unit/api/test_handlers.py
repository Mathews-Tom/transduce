"""Unit tests for the v0 API handlers (P1-API-01..06)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from transduce.api.app import create_app
from transduce.api.schemas import StreamingMode
from transduce.backends.base import BackendCapabilities, BackendHealth, GenerationResult
from transduce.config.schema import (
    BackendEntry,
    BackendsConfig,
    Config,
    LanguageConfig,
    ServiceConfig,
    VerificationConfig,
)
from transduce.verification.base import ScoreResult

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class StubBackend:
    """Backend test double surfacing scripted generations."""

    queue: list[GenerationResult] = field(default_factory=list)
    name: str = "ollama"
    model: str = "qwen2.5:1.5b"
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    healthy: bool = True

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> GenerationResult:
        del prompt, max_tokens, temperature
        if not self.queue:
            return GenerationResult(text="ok", tokens_in=2, tokens_out=2)
        return self.queue.pop(0)

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=self.healthy, detail=None if self.healthy else "down")


@dataclass
class StubScorer:
    name: str
    queue: list[str] = field(default_factory=list)

    def score(self, original: str, candidate: str) -> ScoreResult:
        del original, candidate
        verdict = self.queue.pop(0) if self.queue else "accept"
        return ScoreResult(
            name=self.name,
            value=1.0 if verdict == "accept" else 0.2,
            verdict=verdict,  # type: ignore[arg-type]
            rejection_reason=None if verdict == "accept" else f"{self.name} rejected",
        )


def _config(default_cosine_min: float = 0.85) -> Config:
    return Config(
        service=ServiceConfig(),
        backends=BackendsConfig(
            default="ollama_qwen",
            registry=[
                BackendEntry(
                    id="ollama_qwen",
                    provider="ollama",
                    endpoint="http://ollama.local:11434",
                    model="qwen2.5:1.5b",
                )
            ],
        ),
        verification=VerificationConfig(default_cosine_min=default_cosine_min),
        language=LanguageConfig(),
    )


def _app(
    *,
    backend: StubBackend | None = None,
    scorer_queues: Sequence[Sequence[str]] | None = None,
) -> Litestar:
    from transduce.verification.base import Scorer

    backend = backend or StubBackend(queue=[GenerationResult(text="ok", tokens_in=2, tokens_out=2)])
    queues = scorer_queues or [["accept"]]
    scorers: list[Scorer] = [StubScorer(name="cosine_similarity", queue=list(queues[0]))]
    return create_app(_config(), backend=backend, scorers=scorers)


def _client(app: Litestar) -> TestClient[Litestar]:
    return TestClient(app=app)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_post_transform_invalid_json_returns_400() -> None:
    with _client(_app()) as client:
        response = client.post(
            "/v1/transform", content=b"{not json", headers={"content-type": "application/json"}
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["request_id"]


def test_post_transform_missing_mode_returns_400() -> None:
    with _client(_app()) as client:
        response = client.post("/v1/transform", json={"text": "hi"})

    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"


def test_post_transform_oversize_input_returns_400_input_too_long() -> None:
    payload = {"text": "x" * 50_001, "mode": "dejargon"}
    with _client(_app()) as client:
        response = client.post("/v1/transform", json=payload)

    assert response.status_code == 400
    assert response.json()["error"] == "input_too_long"


def test_post_transform_unknown_mode_returns_404_mode_not_found() -> None:
    with _client(_app()) as client:
        response = client.post("/v1/transform", json={"text": "hi", "mode": "missing"})

    assert response.status_code == 404
    assert response.json()["error"] == "mode_not_found"


def test_post_transform_compose_chain_returns_400_not_implemented() -> None:
    with _client(_app()) as client:
        response = client.post(
            "/v1/transform",
            json={"text": "hi", "mode": ["dejargon", "register.casual"]},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "not_implemented"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_post_transform_returns_200_with_diff_and_scores() -> None:
    backend = StubBackend(queue=[GenerationResult(text="hello earth", tokens_in=5, tokens_out=4)])
    app = _app(backend=backend)

    with _client(app) as client:
        response = client.post(
            "/v1/transform",
            json={
                "text": "hello world",
                "mode": "dejargon",
                "intensity": 0.6,
            },
        )

    assert response.status_code == 201
    body = response.json()
    assert body["transformed"] == "hello earth"
    assert body["scores"]["rejection_reason"] is None
    assert "verdict" not in body["scores"]
    assert body["mode"] == {"id": "dejargon", "version": "0.1.0"}
    assert body["language"] == "en"
    assert body["backend_used"] == {"provider": "ollama", "model": "qwen2.5:1.5b"}
    assert any(op["op"] == "delete" for op in body["diff"])


def test_post_transform_streaming_off_default_serialised_as_off() -> None:
    with _client(_app()) as client:
        response = client.post("/v1/transform", json={"text": "hi", "mode": "dejargon"})

    assert response.status_code == 201
    # The schema default is OFF; round-trip should work for explicit OFF too.
    response_explicit = client.post(
        "/v1/transform",
        json={"text": "hi", "mode": "dejargon", "streaming": StreamingMode.OFF.value},
    )
    assert response_explicit.status_code == 201


# ---------------------------------------------------------------------------
# Catalog routes
# ---------------------------------------------------------------------------


def test_get_modes_returns_three_seed_entries() -> None:
    with _client(_app()) as client:
        response = client.get("/v1/modes")

    assert response.status_code == 200
    body = response.json()
    ids = {entry["id"] for entry in body["modes"]}
    assert ids == {"dejargon", "register.casual", "length.normalize"}


def test_get_mode_unknown_id_returns_404() -> None:
    with _client(_app()) as client:
        response = client.get("/v1/modes/missing")

    assert response.status_code == 404
    assert response.json()["error"] == "mode_not_found"


def test_get_mode_returns_full_spec() -> None:
    with _client(_app()) as client:
        response = client.get("/v1/modes/dejargon")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "dejargon"
    assert body["backend_requirements"]["min_model_b"] == 14.0


def test_get_backends_returns_configured_set() -> None:
    with _client(_app()) as client:
        response = client.get("/v1/backends")

    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "ollama_qwen"
    assert body["backends"][0]["provider"] == "ollama"


def test_get_scorers_returns_registered_set() -> None:
    with _client(_app()) as client:
        response = client.get("/v1/scorers")

    assert response.status_code == 200
    assert response.json() == {"scorers": ["cosine_similarity"]}


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------


def test_healthz_returns_200() -> None:
    with _client(_app()) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_200_when_all_green() -> None:
    with _client(_app(backend=StubBackend(healthy=True))) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["backend"]["healthy"] is True
    assert "dejargon" in body["modes"]


def test_readyz_returns_503_when_backend_unreachable() -> None:
    with _client(_app(backend=StubBackend(healthy=False))) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["backend"]["healthy"] is False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_exposes_transduce_requests_total_after_request() -> None:
    backend = StubBackend(queue=[GenerationResult(text="ok", tokens_in=1, tokens_out=1)])
    app = _app(backend=backend)

    with _client(app) as client:
        client.post("/v1/transform", json={"text": "hi", "mode": "dejargon"})
        response = client.get("/metrics")

    assert response.status_code == 200
    text = response.text
    assert "transduce_requests_total" in text
    assert 'mode="dejargon"' in text


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


def test_error_envelope_contains_request_id() -> None:
    with _client(_app()) as client:
        response = client.post("/v1/transform", json={"text": "", "mode": "dejargon"})

    body = response.json()
    assert body["request_id"]
    assert body["error"] == "validation_error"


def test_error_envelope_carries_inbound_request_id_header() -> None:
    with _client(_app()) as client:
        response = client.post(
            "/v1/transform",
            json={"text": "", "mode": "dejargon"},
            headers={"x-request-id": "client-supplied-id"},
        )

    assert response.json()["request_id"] == "client-supplied-id"


def test_post_transform_verification_failure_returns_422() -> None:
    backend = StubBackend(
        queue=[GenerationResult(text=str(i), tokens_in=1, tokens_out=1) for i in range(5)]
    )
    app = _app(backend=backend, scorer_queues=[["reject", "reject", "reject", "reject"]])

    with _client(app) as client:
        response = client.post(
            "/v1/transform",
            json={
                "text": "hi",
                "mode": "dejargon",
                "verification": {"max_retries": 3},
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "verification_failed"
    assert body["scores"]["rejection_reason"] == "cosine_similarity"
    assert "verdict" not in body["scores"]
    assert body["last_candidate"] == "3"


def test_create_app_without_scorers_raises() -> None:
    with pytest.raises(ValueError, match="scorers"):
        create_app(_config(), backend=StubBackend())


def test_post_transform_request_id_generated_when_absent() -> None:
    with _client(_app()) as client:
        response = client.post("/v1/transform", json={"text": "hi", "mode": "dejargon"})

    body = response.json()
    assert body["request_id"]
    assert len(body["request_id"]) >= 16
