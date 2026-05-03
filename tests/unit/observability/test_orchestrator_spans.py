"""Integration-shape unit tests for orchestrator span emission (P3-OBS-01).

These tests run the orchestrator end-to-end against an in-memory tracer
provider and assert that the documented spans (`transduce.generate`,
`transduce.verify`, `transduce.diff`, `transduce.compose`) appear with
the expected attribute namespacing. They use the same stub backend +
stub scorer pattern as the existing orchestrator unit tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transduce.backends.base import (
    BackendCapabilities,
    BackendHealth,
    GenerationResult,
    StreamChunk,
    StreamFinal,
    StreamTextDelta,
)
from transduce.config.schema import BudgetConfig
from transduce.observability import SpanEmitter
from transduce.observability.attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    SPAN_COMPOSE,
    SPAN_DIFF,
    SPAN_GENERATE,
    SPAN_VERIFY,
    TRANSDUCE_ATTEMPT,
    TRANSDUCE_COMPOSE_STAGES,
    TRANSDUCE_DIFF_OPS_COUNT,
    TRANSDUCE_MODE_ID,
    TRANSDUCE_VERDICT,
)
from transduce.pipeline.orchestrator import Orchestrator
from transduce.registry.spec import (
    BackendRequirements,
    ModeSpec,
    VerifierProfile,
)
from transduce.registry.static import StaticRegistry
from transduce.verification.base import Scorer, ScoreResult
from transduce.verification.composite import CompositeVerifier
from transduce.verification.pipeline import VerifierPipeline


class _StubBackend:
    name = "stub"
    model = "stub-model"
    capabilities = BackendCapabilities(streaming=False, json_mode=False, attention_output=False)

    def __init__(self, response: str = "stub-output") -> None:
        self._response = response

    async def generate(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> GenerationResult:
        del prompt, max_tokens, temperature
        return GenerationResult(text=self._response, tokens_in=12, tokens_out=8)

    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True)

    def cost_estimate(self, *, tokens_in: int, tokens_out: int) -> float | None:
        del tokens_in, tokens_out
        return None

    async def stream(
        self, prompt: str, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamChunk]:
        del prompt, max_tokens, temperature
        yield StreamTextDelta(text=self._response)
        yield StreamFinal(tokens_in=12, tokens_out=8)


class _AcceptScorer(Scorer):
    name = "cosine_similarity"

    def score(self, original: str, candidate: str) -> ScoreResult:
        del original, candidate
        return ScoreResult(name=self.name, value=0.95, verdict="accept", details={})


def _build_registry() -> StaticRegistry:
    spec = ModeSpec(
        id="stub.mode",
        version="1.0.0",
        description="stub mode",
        prompt_template="rewrite: {{ input }}",
        intensity_range=(0.0, 1.0),
        preserve_defaults=(),
        verifier_profile=VerifierProfile(),
        backend_requirements=BackendRequirements(min_model_b=0.0),
        supported_languages=("en",),
    )
    return StaticRegistry((spec,))


def _make_emitter() -> tuple[SpanEmitter, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "transduce-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("transduce.orchestrator-test")
    emitter = SpanEmitter(tracer=tracer, redact_text_in_spans=True, debug_include_text=False)
    return emitter, exporter


def _build_orchestrator(emitter: SpanEmitter) -> Orchestrator:
    scorer = _AcceptScorer()
    pipeline = VerifierPipeline([scorer])
    composite = CompositeVerifier(scorers=[scorer], threshold=0.5)
    return Orchestrator(
        registry=_build_registry(),
        backend=_StubBackend(),
        verifier=pipeline,
        budget_config=BudgetConfig(),
        composite_verifier=composite,
        default_max_retries=1,
        span_emitter=emitter,
    )


def _spans_by_name(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


@pytest.mark.unit
async def test_single_mode_transform_emits_generate_verify_and_diff_spans() -> None:
    emitter, exporter = _make_emitter()
    orchestrator = _build_orchestrator(emitter)

    await orchestrator.transform(
        text="The launch succeeded.",
        mode="stub.mode",
        intensity=0.5,
        preserve=(),
        request_id="req-1",
    )

    generate_spans = _spans_by_name(exporter, SPAN_GENERATE)
    verify_spans = _spans_by_name(exporter, SPAN_VERIFY)
    diff_spans = _spans_by_name(exporter, SPAN_DIFF)
    assert len(generate_spans) == 1
    assert len(verify_spans) == 1
    assert len(diff_spans) == 1


@pytest.mark.unit
async def test_generate_span_carries_gen_ai_namespaced_attributes() -> None:
    emitter, exporter = _make_emitter()
    orchestrator = _build_orchestrator(emitter)

    await orchestrator.transform(
        text="The launch succeeded.",
        mode="stub.mode",
        intensity=0.5,
        preserve=(),
        request_id="req-2",
    )

    span = _spans_by_name(exporter, SPAN_GENERATE)[0]
    attrs = span.attributes or {}
    assert attrs.get(GEN_AI_SYSTEM) == "transduce"
    assert attrs.get(GEN_AI_REQUEST_MODEL) == "stub-model"
    assert attrs.get(GEN_AI_USAGE_INPUT_TOKENS) == 12
    assert attrs.get(GEN_AI_USAGE_OUTPUT_TOKENS) == 8
    assert attrs.get(TRANSDUCE_MODE_ID) == "stub.mode"
    assert attrs.get(TRANSDUCE_ATTEMPT) == 1


@pytest.mark.unit
async def test_verify_span_records_verdict_and_per_scorer_attributes() -> None:
    emitter, exporter = _make_emitter()
    orchestrator = _build_orchestrator(emitter)

    await orchestrator.transform(
        text="The launch succeeded.",
        mode="stub.mode",
        intensity=0.5,
        preserve=(),
        request_id="req-3",
    )

    span = _spans_by_name(exporter, SPAN_VERIFY)[0]
    attrs = span.attributes or {}
    assert attrs.get(TRANSDUCE_VERDICT) == "accept"
    assert attrs.get("transduce.scorer.cosine") == pytest.approx(0.95)


@pytest.mark.unit
async def test_diff_span_records_ops_count_for_word_level_diff() -> None:
    emitter, exporter = _make_emitter()
    orchestrator = _build_orchestrator(emitter)

    await orchestrator.transform(
        text="The launch succeeded.",
        mode="stub.mode",
        intensity=0.5,
        preserve=(),
        request_id="req-4",
    )

    span = _spans_by_name(exporter, SPAN_DIFF)[0]
    attrs = span.attributes or {}
    ops_count = attrs.get(TRANSDUCE_DIFF_OPS_COUNT)
    assert isinstance(ops_count, int)
    assert ops_count >= 1


@pytest.mark.unit
async def test_compose_chain_emits_compose_span_with_stage_count() -> None:
    emitter, exporter = _make_emitter()
    spec = ModeSpec(
        id="other.mode",
        version="1.0.0",
        description="other",
        prompt_template="rewrite: {{ input }}",
        intensity_range=(0.0, 1.0),
        preserve_defaults=(),
        verifier_profile=VerifierProfile(),
        backend_requirements=BackendRequirements(min_model_b=0.0),
        supported_languages=("en",),
    )
    base = _build_registry()
    registry = StaticRegistry((*base.list_modes(), spec))
    scorer = _AcceptScorer()
    orchestrator = Orchestrator(
        registry=registry,
        backend=_StubBackend(),
        verifier=VerifierPipeline([scorer]),
        budget_config=BudgetConfig(),
        composite_verifier=CompositeVerifier(scorers=[scorer], threshold=0.5),
        default_max_retries=1,
        span_emitter=emitter,
    )

    await orchestrator.transform(
        text="The launch succeeded.",
        mode=["stub.mode", "other.mode"],
        intensity=0.5,
        preserve=(),
        request_id="req-5",
    )

    compose_spans = _spans_by_name(exporter, SPAN_COMPOSE)
    assert len(compose_spans) == 1
    attrs = compose_spans[0].attributes or {}
    assert attrs.get(TRANSDUCE_COMPOSE_STAGES) == 2


@pytest.mark.unit
async def test_disabled_emitter_does_not_break_transform_path() -> None:
    emitter = SpanEmitter.disabled()
    orchestrator = _build_orchestrator(emitter)

    result = await orchestrator.transform(
        text="The launch succeeded.",
        mode="stub.mode",
        intensity=0.5,
        preserve=(),
        request_id="req-6",
    )

    # Disabled emitter routes through the OTel global no-op tracer; the
    # contract is that transform still succeeds end-to-end without any
    # span side-effects. A successful return without raising is the
    # only assertable signal at this layer.
    assert result.transformed == "stub-output"
