"""Unit tests for the advisory SSE transform endpoint (P3-STR-01..02)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest
from litestar import Litestar
from litestar.testing import TestClient

from transduce.api.app import create_app
from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    BackendUnavailableError,
    GenerationResult,
    GenerationTimeoutError,
    StreamChunk,
    StreamFinal,
    StreamTextDelta,
)
from transduce.budget.budgeter import BudgetExceededError, BudgetState
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

pytestmark = pytest.mark.unit


@dataclass
class _StreamingStubBackend:
    """Stub that emits a fixed sequence of streaming chunks."""

    name: str = "ollama"
    model: str = "qwen2.5:1.5b"
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(streaming=True)
    )
    chunks: list[str] = field(default_factory=lambda: ["Hello", " ", "world"])
    tokens_in: int = 8
    tokens_out: int = 3

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> GenerationResult:
        del prompt, max_tokens, temperature
        return GenerationResult(
            text="".join(self.chunks),
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
        )

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamChunk]:
        del prompt, max_tokens, temperature
        for chunk in self.chunks:
            yield StreamTextDelta(text=chunk)
        yield StreamFinal(tokens_in=self.tokens_in, tokens_out=self.tokens_out)

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        del tokens_in, tokens_out
        return None


@dataclass
class _NonStreamingStubBackend(_StreamingStubBackend):
    capabilities: BackendCapabilities = field(
        default_factory=lambda: BackendCapabilities(streaming=False)
    )


@dataclass
class _ScriptedScorer:
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


def _config() -> Config:
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
                    model_size_b=14.0,
                )
            ],
        ),
        verification=VerificationConfig(default_cosine_min=0.85),
        budget=BudgetConfig(),
        language=LanguageConfig(),
    )


def _app(
    *,
    backend: _StreamingStubBackend | None = None,
    scorer_queues: Sequence[Sequence[str]] | None = None,
) -> Litestar:
    backend = backend or _StreamingStubBackend()
    queues = scorer_queues or [["accept"]]
    scorers: list[Scorer] = [
        _ScriptedScorer(name="cosine_similarity", queue=list(queues[0])),
    ]
    return create_app(_config(), backend=backend, scorers=scorers)


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    current_event: str | None = None
    current_data: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            if current_event is not None and current_data:
                payload = "\n".join(current_data)
                events.append((current_event, json.loads(payload)))
            current_event = None
            current_data = []
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:") :].strip())
    return events


def test_stream_strict_returns_400_not_implemented() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi", "mode": "dejargon", "streaming": "strict"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "not_implemented"
    assert "strict verification" in response.json()["message"]


def test_stream_off_on_streaming_endpoint_returns_400_validation_error() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi", "mode": "dejargon", "streaming": "off"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"
    assert "advisory" in response.json()["message"]


def test_stream_compose_chain_returns_400_not_implemented() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={
                "text": "hi",
                "mode": ["dejargon", "register.casual"],
                "streaming": "advisory",
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == "not_implemented"
    assert "compose chains" in response.json()["message"]


def test_stream_non_streaming_backend_returns_400_not_implemented() -> None:
    app = _app(backend=_NonStreamingStubBackend())
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi", "mode": "dejargon", "streaming": "advisory"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == "not_implemented"
    assert "does not support streaming" in response.json()["message"]


def test_stream_advisory_returns_chunk_then_verdict_event_on_accept() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    chunk_events = [data for name, data in events if name == "chunk"]
    verdict_events = [data for name, data in events if name == "verdict"]
    assert [c["text"] for c in chunk_events] == ["Hello", " ", "world"]
    assert len(verdict_events) == 1
    verdict = verdict_events[0]
    assert verdict["verdict"] == "accept"
    assert verdict["transformed"] == "Hello world"
    assert verdict["tokens_in"] == 8
    assert verdict["tokens_out"] == 3


def test_stream_advisory_emits_reject_verdict_when_verifier_rejects() -> None:
    app = _app(scorer_queues=[["reject"]])
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    verdict_events = [data for name, data in events if name == "verdict"]
    assert len(verdict_events) == 1
    verdict = verdict_events[0]
    assert verdict["verdict"] == "reject"
    assert verdict["rejection_reason"] is not None
    assert verdict["transformed"] == "Hello world"


def test_stream_advisory_unknown_mode_returns_404_mode_not_found() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi", "mode": "missing.mode", "streaming": "advisory"},
        )

    assert response.status_code == 404
    assert response.json()["error"] == "mode_not_found"


@dataclass
class _RaisingStreamingBackend(_StreamingStubBackend):
    """Streaming stub that raises a configured exception mid-stream."""

    raise_after: int = 1
    exc_to_raise: Exception | None = None

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamChunk]:
        del prompt, max_tokens, temperature
        for emitted, chunk in enumerate(self.chunks):
            if emitted >= self.raise_after and self.exc_to_raise is not None:
                raise self.exc_to_raise
            yield StreamTextDelta(text=chunk)
        if self.exc_to_raise is not None:
            raise self.exc_to_raise
        yield StreamFinal(tokens_in=self.tokens_in, tokens_out=self.tokens_out)


def test_stream_advisory_emits_error_event_with_backend_unavailable_code() -> None:
    backend = _RaisingStreamingBackend(
        chunks=["partial"],
        raise_after=1,
        exc_to_raise=BackendUnavailableError("ollama unreachable at http://x"),
    )
    app = _app(backend=backend)
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    error_events = [data for name, data in events if name == "error"]
    assert len(error_events) == 1
    assert error_events[0]["error"] == "backend_unavailable"


def test_stream_advisory_emits_error_event_with_timeout_code() -> None:
    backend = _RaisingStreamingBackend(
        chunks=["partial"],
        raise_after=1,
        exc_to_raise=GenerationTimeoutError("ollama streaming timed out after 60s"),
    )
    app = _app(backend=backend)
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    events = _parse_sse(response.text)
    error_events = [data for name, data in events if name == "error"]
    assert error_events[0]["error"] == "timeout"


def test_stream_advisory_emits_error_event_with_budget_exceeded_code() -> None:
    state = BudgetState(total_cost_usd=0.10, attempts=2, scores=(0.5, 0.4))
    backend = _RaisingStreamingBackend(
        chunks=["partial"],
        raise_after=1,
        exc_to_raise=BudgetExceededError(
            reason="budget_exceeded",
            state=state,
            limit=0.05,
        ),
    )
    app = _app(backend=backend)
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    events = _parse_sse(response.text)
    error_events = [data for name, data in events if name == "error"]
    assert error_events[0]["error"] == "budget_exceeded"


def test_stream_advisory_includes_diff_in_verdict_event() -> None:
    app = _app()
    with TestClient(app=app) as client:
        response = client.post(
            "/v1/transform/stream",
            json={"text": "hi there", "mode": "dejargon", "streaming": "advisory"},
        )

    events = _parse_sse(response.text)
    verdict = next(data for name, data in events if name == "verdict")
    assert isinstance(verdict["diff"], list)
    assert len(verdict["diff"]) >= 1
    for op in verdict["diff"]:
        assert isinstance(op, dict)
        assert op["op"] in {"equal", "insert", "delete"}
