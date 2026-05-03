"""Span emitter binding tracer + redaction policy (P3-OBS-01..04).

The :class:`SpanEmitter` is the only abstraction the orchestrator and
the HTTP handlers use to emit OTel spans. It hides three concerns:

1. **Test seam** — tests construct an emitter from an
   :class:`~opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter`
   tracer to assert attributes; production wires the OTLP exporter via
   :func:`build_tracer_provider`.
2. **Redaction policy** — :meth:`text_attributes` returns ``sha256_8 +
   length`` by default and only includes the raw text when
   ``debug_include_text`` is true (gated by the config validator so the
   pairing cannot be set unsafely).
3. **No-op default** — :meth:`disabled` returns an emitter backed by the
   OTel global no-op tracer so call sites stay unconditional and pay
   near-zero cost when ``observability.enabled=false``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from transduce.config.schema import ObservabilityConfig
from transduce.observability.attributes import (
    TRANSDUCE_TEXT_LENGTH,
    TRANSDUCE_TEXT_SHA256_8,
    TRANSDUCE_TEXT_VALUE,
)
from transduce.observability.redaction import sha256_8

_TRACER_NAME: Final[str] = "transduce"

AttrValue = str | int | float | bool
AttrMap = Mapping[str, AttrValue]


def build_tracer_provider(
    config: ObservabilityConfig,
    *,
    service_name: str = "transduce",
) -> TracerProvider | None:
    """Build a configured :class:`TracerProvider`, or ``None`` when disabled.

    Returns ``None`` when ``config.enabled`` is false; callers MUST treat
    that signal as "do not install a global provider" so the OTel global
    state stays at its no-op default. When ``otel_endpoint`` is set the
    provider ships spans to that collector via the OTLP HTTP exporter
    (P3-OBS-01); when unset the provider is built without an exporter
    so spans are still produced but never leave the process — useful
    for in-process introspection and integration tests.
    """
    if not config.enabled:
        return None
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if config.otel_endpoint is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otel_endpoint))
        )
    return provider


@dataclass(frozen=True)
class SpanEmitter:
    """Bound tracer + redaction policy used by orchestrator and handlers."""

    tracer: trace.Tracer
    redact_text_in_spans: bool
    debug_include_text: bool

    @classmethod
    def from_config(
        cls,
        config: ObservabilityConfig,
        *,
        tracer: trace.Tracer | None = None,
    ) -> SpanEmitter:
        """Build an emitter from ``config``, defaulting to the global tracer.

        Tests pass an explicit ``tracer`` from a private TracerProvider;
        production wiring (CLI ``serve``) installs the provider globally
        and lets this fall through to ``trace.get_tracer``.
        """
        return cls(
            tracer=tracer or trace.get_tracer(_TRACER_NAME),
            redact_text_in_spans=config.redact_text_in_spans,
            debug_include_text=config.debug_include_text,
        )

    @classmethod
    def disabled(cls) -> SpanEmitter:
        """Return an emitter backed by the OTel no-op tracer.

        ``redact_text_in_spans=True`` is preserved so the redaction
        policy is correct even when no-op spans are produced. The
        no-op tracer is what ``trace.get_tracer`` returns before any
        provider is installed; subsequent ``with emitter.span(...)``
        blocks are zero-overhead.
        """
        return cls(
            tracer=trace.get_tracer(_TRACER_NAME),
            redact_text_in_spans=True,
            debug_include_text=False,
        )

    @contextmanager
    def span(self, name: str, attributes: AttrMap | None = None) -> Iterator[trace.Span]:
        """Start a span with ``name`` and the optional initial ``attributes``."""
        with self.tracer.start_as_current_span(name, attributes=dict(attributes or {})) as span:
            yield span

    def text_attributes(
        self, text: str, *, key_prefix: str = "transduce.text"
    ) -> dict[str, AttrValue]:
        """Return redaction-aware attributes for ``text``.

        Always emits ``<prefix>.sha256_8`` and ``<prefix>.length``; emits
        ``<prefix>.value`` only when ``debug_include_text`` is true and
        ``redact_text_in_spans`` is false. The pairing is enforced by
        :meth:`ObservabilityConfig._debug_text_requires_redaction_off`,
        so the runtime never has to defend against the ambiguous case
        of "include text but also redact." When ``key_prefix`` matches
        the canonical ``transduce.text`` namespace the constants in
        :mod:`transduce.observability.attributes` line up; ``key_prefix``
        is exposed so per-stage spans can scope ``transduce.text.input``
        vs ``transduce.text.output`` without rebuilding the dict.
        """
        digest = sha256_8(text)
        length = len(text)
        if key_prefix == "transduce.text":
            attrs: dict[str, AttrValue] = {
                TRANSDUCE_TEXT_SHA256_8: digest,
                TRANSDUCE_TEXT_LENGTH: length,
            }
            if self._allow_raw_text():
                attrs[TRANSDUCE_TEXT_VALUE] = text
            return attrs
        attrs = {
            f"{key_prefix}.sha256_8": digest,
            f"{key_prefix}.length": length,
        }
        if self._allow_raw_text():
            attrs[f"{key_prefix}.value"] = text
        return attrs

    def _allow_raw_text(self) -> bool:
        return self.debug_include_text and not self.redact_text_in_spans


__all__ = [
    "AttrMap",
    "AttrValue",
    "SpanEmitter",
    "build_tracer_provider",
]
