"""OTel GenAI SemConv emission for transduce (P3-OBS-01..04).

Public entry points:

- :class:`SpanEmitter` — the only abstraction orchestrator and handlers
  use to emit spans; pre-bound with the redaction policy from
  :class:`~transduce.config.schema.ObservabilityConfig`.
- :func:`build_tracer_provider` — construct an OTLP-exporting
  TracerProvider from config; returns ``None`` when observability is
  disabled so the global no-op tracer stays in place.
- :func:`sha256_8` — direct redaction helper for non-span code paths.

Attribute and span-name constants live in
:mod:`transduce.observability.attributes`.
"""

from transduce.observability.redaction import sha256_8
from transduce.observability.spans import (
    AttrMap,
    AttrValue,
    SpanEmitter,
    build_tracer_provider,
)

__all__ = [
    "AttrMap",
    "AttrValue",
    "SpanEmitter",
    "build_tracer_provider",
    "sha256_8",
]
