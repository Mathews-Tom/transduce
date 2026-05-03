"""Tests for SpanEmitter and the orchestrator span wiring (P3-OBS-01..04)."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from transduce.config.schema import ObservabilityConfig
from transduce.observability import SpanEmitter, build_tracer_provider
from transduce.observability.attributes import (
    SPAN_GENERATE,
    SPAN_VERIFY,
    TRANSDUCE_TEXT_LENGTH,
    TRANSDUCE_TEXT_SHA256_8,
    TRANSDUCE_TEXT_VALUE,
)


def _make_in_memory_emitter(
    *,
    redact_text_in_spans: bool = True,
    debug_include_text: bool = False,
) -> tuple[SpanEmitter, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "transduce-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("transduce.test")
    emitter = SpanEmitter(
        tracer=tracer,
        redact_text_in_spans=redact_text_in_spans,
        debug_include_text=debug_include_text,
    )
    return emitter, exporter


def _finished(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return list(exporter.get_finished_spans())


@pytest.mark.unit
def test_span_context_manager_emits_span_with_initial_attributes() -> None:
    emitter, exporter = _make_in_memory_emitter()

    with emitter.span(SPAN_GENERATE, {"transduce.attempt": 1}):
        pass

    spans = _finished(exporter)
    assert len(spans) == 1
    assert spans[0].name == SPAN_GENERATE
    attrs = spans[0].attributes or {}
    assert attrs.get("transduce.attempt") == 1


@pytest.mark.unit
def test_span_emitter_set_attribute_after_open() -> None:
    emitter, exporter = _make_in_memory_emitter()

    with emitter.span(SPAN_VERIFY) as span:
        span.set_attribute("transduce.verdict", "accept")
        span.set_attribute("transduce.scorer.cosine", 0.91)

    spans = _finished(exporter)
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get("transduce.verdict") == "accept"
    assert attrs.get("transduce.scorer.cosine") == pytest.approx(0.91)


@pytest.mark.unit
def test_redaction_replaces_raw_text_with_sha256_8_by_default() -> None:
    emitter, _ = _make_in_memory_emitter()

    attrs = emitter.text_attributes("Acme reported $4.2M in Q3 revenue.")

    assert TRANSDUCE_TEXT_SHA256_8 in attrs
    assert TRANSDUCE_TEXT_LENGTH in attrs
    assert TRANSDUCE_TEXT_VALUE not in attrs
    digest = attrs[TRANSDUCE_TEXT_SHA256_8]
    assert isinstance(digest, str)
    assert len(digest) == 8
    assert attrs[TRANSDUCE_TEXT_LENGTH] == len("Acme reported $4.2M in Q3 revenue.")


@pytest.mark.unit
def test_debug_include_text_requires_explicit_opt_in() -> None:
    emitter, _ = _make_in_memory_emitter(redact_text_in_spans=False, debug_include_text=True)

    text = "Q3 adoption hit 12% of active users."
    attrs = emitter.text_attributes(text)

    assert attrs[TRANSDUCE_TEXT_VALUE] == text
    assert TRANSDUCE_TEXT_SHA256_8 in attrs


@pytest.mark.unit
def test_debug_include_text_with_redaction_on_keeps_text_redacted() -> None:
    """The config validator forbids ``debug_include_text=true`` together with
    ``redact_text_in_spans=true``. The runtime treats that combination as
    redact-only so a misconfigured deployment cannot leak text.
    """
    emitter, _ = _make_in_memory_emitter(redact_text_in_spans=True, debug_include_text=True)

    attrs = emitter.text_attributes("Acme reported $4.2M.")

    assert TRANSDUCE_TEXT_VALUE not in attrs
    assert TRANSDUCE_TEXT_SHA256_8 in attrs


@pytest.mark.unit
def test_text_attributes_with_custom_prefix_uses_prefix_for_all_keys() -> None:
    emitter, _ = _make_in_memory_emitter()

    attrs = emitter.text_attributes("hello", key_prefix="transduce.text.input")

    assert "transduce.text.input.sha256_8" in attrs
    assert "transduce.text.input.length" in attrs
    assert "transduce.text.input.value" not in attrs


@pytest.mark.unit
def test_disabled_emitter_uses_no_op_tracer_with_redaction_enforced() -> None:
    emitter = SpanEmitter.disabled()

    assert emitter.redact_text_in_spans is True
    assert emitter.debug_include_text is False

    attrs = emitter.text_attributes("secret payload")

    assert TRANSDUCE_TEXT_VALUE not in attrs


@pytest.mark.unit
def test_build_tracer_provider_returns_none_when_observability_disabled() -> None:
    config = ObservabilityConfig(enabled=False)

    provider = build_tracer_provider(config)

    assert provider is None


@pytest.mark.unit
def test_build_tracer_provider_returns_provider_when_enabled() -> None:
    config = ObservabilityConfig(enabled=True)

    provider = build_tracer_provider(config)

    assert provider is not None
    assert isinstance(provider, TracerProvider)


@pytest.mark.unit
def test_emitter_from_config_respects_redaction_flags() -> None:
    config = ObservabilityConfig(
        enabled=False,
        redact_text_in_spans=False,
        debug_include_text=True,
    )

    emitter = SpanEmitter.from_config(config)

    assert emitter.redact_text_in_spans is False
    assert emitter.debug_include_text is True
