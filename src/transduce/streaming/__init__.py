"""Embedded-library client for the advisory SSE transform endpoint (P3-STR-03).

The Python client is the in-process equivalent of the TypeScript
reference SDK shipped from the ``armory`` repo. It consumes the
``/v1/transform/stream`` event-stream, parses ``chunk`` and ``verdict``
events into typed values, and surfaces a ``rollback`` signal on
``verdict: reject`` so callers can discard the streamed text.

The public surface is intentionally small: one ``stream_transform``
async iterator coroutine and the typed event classes. Tests and
embedded callers that already manage their own ``httpx.AsyncClient``
can also use :func:`parse_sse_events` directly against an arbitrary
``AsyncIterator[str]``.
"""

from transduce.streaming.client import (
    ChunkEvent,
    ClientEvent,
    ErrorEvent,
    StreamingClientError,
    VerdictEvent,
    parse_sse_events,
    stream_transform,
)

__all__ = [
    "ChunkEvent",
    "ClientEvent",
    "ErrorEvent",
    "StreamingClientError",
    "VerdictEvent",
    "parse_sse_events",
    "stream_transform",
]
