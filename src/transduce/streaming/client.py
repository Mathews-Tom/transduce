"""Streaming client helper with rollback semantics (P3-STR-03)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx

ClientVerdict = Literal["accept", "reject"]


class StreamingClientError(RuntimeError):
    """Raised when the server returns a non-200 response to the SSE request.

    Server-side errors that arrive *inside* the event stream (after a
    200 has already been emitted) surface as :class:`ErrorEvent` and do
    NOT raise — the iterator yields the error event so the caller can
    log it and roll back any partially-rendered text.
    """

    def __init__(self, *, status_code: int, body: str) -> None:
        super().__init__(f"streaming endpoint returned {status_code}: {body}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ChunkEvent:
    """One text-delta event the server forwarded from the backend."""

    text: str


@dataclass(frozen=True)
class VerdictEvent:
    """Terminal event carrying the verifier verdict and full result payload."""

    verdict: ClientVerdict
    transformed: str
    rejection_reason: str | None
    diff: list[dict[str, Any]]
    scores: dict[str, Any]
    raw: dict[str, Any]

    @property
    def rollback(self) -> bool:
        """``True`` iff the verifier rejected — caller discards streamed text."""
        return self.verdict == "reject"


@dataclass(frozen=True)
class ErrorEvent:
    """Error event the server emitted mid-stream (concurrency / backend failure)."""

    error: str
    message: str
    raw: dict[str, Any]


ClientEvent = ChunkEvent | VerdictEvent | ErrorEvent


async def parse_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[ClientEvent]:
    """Parse a stream of SSE-formatted lines into typed events.

    Tolerates the standard SSE quirks: ``\\r`` line endings on the
    wire, blank-line separators between events, and tail events that
    arrive without a trailing blank line. Unknown event names are
    silently dropped — the contract is that ``chunk``, ``verdict``,
    and ``error`` are the only events the server emits.
    """
    current_event: str | None = None
    current_data: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            event = _flush(current_event, current_data)
            if event is not None:
                yield event
            current_event = None
            current_data = []
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data.append(line[len("data:") :].strip())
    event = _flush(current_event, current_data)
    if event is not None:
        yield event


async def stream_transform(
    *,
    client: httpx.AsyncClient,
    url: str,
    text: str,
    mode: str,
    intensity: float = 0.5,
    preserve: list[str] | None = None,
    extra_payload: Mapping[str, Any] | None = None,
) -> AsyncIterator[ClientEvent]:
    """POST to the advisory SSE endpoint and yield typed client events.

    Caller owns ``client`` (so the same connection pool can serve the
    streaming endpoint and the non-streaming endpoint side by side).
    A non-200 response raises :class:`StreamingClientError` before any
    event is yielded. Mid-stream server errors arrive as
    :class:`ErrorEvent`; the iterator never raises after the first
    chunk has been emitted.

    Rollback contract: a :class:`VerdictEvent` with
    ``verdict == "reject"`` (equivalently ``rollback is True``) is the
    signal that the caller must discard the accumulated text from
    every preceding :class:`ChunkEvent`.
    """
    payload: dict[str, Any] = {
        "text": text,
        "mode": mode,
        "intensity": intensity,
        "streaming": "advisory",
    }
    if preserve is not None:
        payload["preserve"] = list(preserve)
    if extra_payload is not None:
        payload.update(dict(extra_payload))

    async with client.stream("POST", url, json=payload) as response:
        if response.status_code != httpx.codes.OK:
            body = (await response.aread()).decode("utf-8", errors="replace")
            raise StreamingClientError(status_code=response.status_code, body=body)
        async for event in parse_sse_events(response.aiter_lines()):
            yield event


def _flush(event_name: str | None, data_lines: list[str]) -> ClientEvent | None:
    if event_name is None or not data_lines:
        return None
    payload = json.loads("\n".join(data_lines))
    if not isinstance(payload, dict):
        return None
    if event_name == "chunk":
        text = payload.get("text")
        if not isinstance(text, str):
            return None
        return ChunkEvent(text=text)
    if event_name == "verdict":
        verdict = payload.get("verdict")
        if verdict not in ("accept", "reject"):
            return None
        diff = payload.get("diff")
        scores = payload.get("scores")
        return VerdictEvent(
            verdict=verdict,
            transformed=str(payload.get("transformed", "")),
            rejection_reason=_optional_str(payload.get("rejection_reason")),
            diff=list(diff) if isinstance(diff, list) else [],
            scores=dict(scores) if isinstance(scores, dict) else {},
            raw=dict(payload),
        )
    if event_name == "error":
        return ErrorEvent(
            error=str(payload.get("error", "unknown")),
            message=str(payload.get("message", "")),
            raw=dict(payload),
        )
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
