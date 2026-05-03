"""Tests for the streaming client helper (P3-STR-03)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from transduce.streaming import (
    ChunkEvent,
    ErrorEvent,
    StreamingClientError,
    VerdictEvent,
    parse_sse_events,
    stream_transform,
)

pytestmark = pytest.mark.unit


async def _aiter(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


def _build_chunk(text: str) -> list[str]:
    return [
        "event: chunk",
        f"data: {json.dumps({'text': text})}",
        "",
    ]


def _build_verdict(verdict: str, **extra: object) -> list[str]:
    payload = {"verdict": verdict, **extra}
    return [
        "event: verdict",
        f"data: {json.dumps(payload)}",
        "",
    ]


# ---------------------------------------------------------------------------
# parse_sse_events
# ---------------------------------------------------------------------------


async def test_parse_sse_events_yields_chunk_then_verdict() -> None:
    lines = [
        *_build_chunk("Hello"),
        *_build_chunk(" world"),
        *_build_verdict("accept", transformed="Hello world", scores={"cosine": 0.91}),
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]

    assert isinstance(events[0], ChunkEvent)
    assert events[0].text == "Hello"
    assert isinstance(events[1], ChunkEvent)
    assert events[1].text == " world"
    assert isinstance(events[2], VerdictEvent)
    assert events[2].verdict == "accept"
    assert events[2].transformed == "Hello world"
    assert events[2].rollback is False


async def test_parse_sse_events_verdict_reject_signals_rollback() -> None:
    lines = [
        *_build_chunk("partial"),
        *_build_verdict(
            "reject",
            transformed="partial",
            rejection_reason="cosine_similarity below threshold",
        ),
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]
    verdict = events[-1]

    assert isinstance(verdict, VerdictEvent)
    assert verdict.rollback is True
    assert verdict.rejection_reason == "cosine_similarity below threshold"


async def test_parse_sse_events_handles_carriage_returns() -> None:
    lines = [
        "event: chunk\r",
        f"data: {json.dumps({'text': 'Hi'})}\r",
        "\r",
        *_build_verdict("accept", transformed="Hi"),
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]

    assert isinstance(events[0], ChunkEvent)
    assert events[0].text == "Hi"


async def test_parse_sse_events_emits_tail_event_without_trailing_blank() -> None:
    lines = [
        "event: verdict",
        f"data: {json.dumps({'verdict': 'accept', 'transformed': 'ok'})}",
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]

    assert len(events) == 1
    assert isinstance(events[0], VerdictEvent)


async def test_parse_sse_events_drops_unknown_event_types() -> None:
    lines = [
        "event: heartbeat",
        f"data: {json.dumps({'now': 'whenever'})}",
        "",
        *_build_verdict("accept", transformed="ok"),
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]

    assert len(events) == 1
    assert isinstance(events[0], VerdictEvent)


async def test_parse_sse_events_yields_error_event() -> None:
    lines = [
        "event: error",
        f"data: {json.dumps({'error': 'concurrency_limit_exceeded', 'message': 'too many'})}",
        "",
    ]

    events = [event async for event in parse_sse_events(_aiter(lines))]

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, ErrorEvent)
    assert error.error == "concurrency_limit_exceeded"
    assert error.message == "too many"


# ---------------------------------------------------------------------------
# stream_transform (HTTP)
# ---------------------------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


def _sse_body(events: list[tuple[str, dict[str, object]]]) -> bytes:
    parts = []
    for name, data in events:
        parts.append(f"event: {name}\n")
        parts.append(f"data: {json.dumps(data)}\n\n")
    return "".join(parts).encode("utf-8")


async def test_stream_transform_yields_chunks_then_verdict_on_accept() -> None:
    body = _sse_body(
        [
            ("chunk", {"text": "Hello"}),
            ("chunk", {"text": " world"}),
            (
                "verdict",
                {
                    "verdict": "accept",
                    "transformed": "Hello world",
                    "scores": {"cosine": 0.91},
                    "diff": [{"op": "equal", "text": "Hello world"}],
                },
            ),
        ]
    )

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with _client(handler) as client:
        events = [
            event
            async for event in stream_transform(
                client=client,
                url="http://test/v1/transform/stream",
                text="hi",
                mode="dejargon",
            )
        ]

    chunk_texts = [e.text for e in events if isinstance(e, ChunkEvent)]
    verdicts = [e for e in events if isinstance(e, VerdictEvent)]
    assert chunk_texts == ["Hello", " world"]
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "accept"
    assert verdicts[0].rollback is False
    assert verdicts[0].transformed == "Hello world"
    payload = captured["body"]
    assert isinstance(payload, dict)
    assert payload["streaming"] == "advisory"


async def test_stream_transform_rolls_back_streamed_text_on_reject() -> None:
    body = _sse_body(
        [
            ("chunk", {"text": "questionable"}),
            (
                "verdict",
                {
                    "verdict": "reject",
                    "transformed": "questionable",
                    "rejection_reason": "negation_diff added",
                    "scores": {},
                    "diff": [],
                },
            ),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with _client(handler) as client:
        events = [
            event
            async for event in stream_transform(
                client=client,
                url="http://test/v1/transform/stream",
                text="hi",
                mode="dejargon",
            )
        ]

    accumulated = "".join(e.text for e in events if isinstance(e, ChunkEvent))
    verdict = next(e for e in events if isinstance(e, VerdictEvent))
    final_text = accumulated if not verdict.rollback else ""

    assert verdict.rollback is True
    assert final_text == ""
    assert verdict.rejection_reason == "negation_diff added"


async def test_stream_transform_non_200_raises_streaming_client_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content=json.dumps(
                {
                    "request_id": "x",
                    "error": "not_implemented",
                    "message": "strict not allowed",
                }
            ).encode(),
        )

    async with _client(handler) as client:
        with pytest.raises(StreamingClientError, match="returned 400"):
            async for _ in stream_transform(
                client=client,
                url="http://test/v1/transform/stream",
                text="hi",
                mode="dejargon",
            ):
                pass


async def test_stream_transform_passes_preserve_and_intensity_into_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read().decode())
        return httpx.Response(
            200,
            content=_sse_body([("verdict", {"verdict": "accept", "transformed": "ok"})]),
        )

    async with _client(handler) as client:
        async for _ in stream_transform(
            client=client,
            url="http://test/v1/transform/stream",
            text="hi",
            mode="dejargon",
            intensity=0.7,
            preserve=["entities", "numbers"],
        ):
            pass

    payload = captured["body"]
    assert isinstance(payload, dict)
    assert payload["intensity"] == 0.7
    assert payload["preserve"] == ["entities", "numbers"]


async def test_stream_transform_yields_error_event_when_server_emits_one() -> None:
    body = _sse_body(
        [
            (
                "error",
                {"error": "generation_failed", "message": "backend unreachable"},
            ),
        ]
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    async with _client(handler) as client:
        events = [
            event
            async for event in stream_transform(
                client=client,
                url="http://test/v1/transform/stream",
                text="hi",
                mode="dejargon",
            )
        ]

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, ErrorEvent)
    assert error.error == "generation_failed"
