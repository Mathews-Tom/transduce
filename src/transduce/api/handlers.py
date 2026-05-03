"""HTTP route handlers for the v1 API surface."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from http import HTTPStatus
from typing import Any

from litestar import Request, Response, get, post
from litestar.response import ServerSentEvent, ServerSentEventMessage
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from transduce.api.errors import request_id_for
from transduce.api.schemas import (
    BackendInfo,
    ErrorCode,
    ModeRef,
    StreamingMode,
    TransformError,
    TransformRequest,
    TransformResponse,
)
from transduce.api.state import TransduceState
from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
)
from transduce.backends.concurrency import ConcurrencyLimitExceededError
from transduce.backends.preconditions import enforce_min_model_b
from transduce.budget.budgeter import BudgetExceededError
from transduce.injection.scanner import InputInjectionDetectedError
from transduce.language.detector import LanguageNotSupportedError
from transduce.observability.attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_TRANSDUCE,
    SPAN_REQUEST,
    SPAN_SCAN,
    TRANSDUCE_COST_USD,
    TRANSDUCE_LANGUAGE,
    TRANSDUCE_MODE_ID,
    TRANSDUCE_RETRIES,
    TRANSDUCE_SCAN_MATCHED_PATTERN,
    TRANSDUCE_VERDICT,
)
from transduce.pipeline.streaming import (
    StreamChunkEvent,
    StreamVerdictEvent,
    stream_transform,
)
from transduce.registry.spec import ModeSpec, PreserveRule


def _state(request: Request[Any, Any, Any]) -> TransduceState:
    state = request.app.state.transduce_state
    if not isinstance(state, TransduceState):  # pragma: no cover — wiring contract
        raise RuntimeError("transduce_state not bound on app.state")
    return state


def _coerce_preserve(values: Iterable[PreserveRule]) -> tuple[PreserveRule, ...]:
    return tuple(values)


_MODE_RESPONSE_EXCLUDE = {"prompt_template"}


def _mode_to_dict(spec: ModeSpec) -> dict[str, Any]:
    """Serialise a ``ModeSpec`` for the catalog response.

    The prompt template is excluded — exposing it via ``GET /v1/modes`` would
    leak the v0 prompts. Mode introspection with the rendered prompt is the
    later ``POST /v1/modes/{id}/render`` surface (P4-INTRO-01).
    """
    return spec.model_dump(mode="json", exclude=_MODE_RESPONSE_EXCLUDE)


@post("/v1/transform")
async def post_transform(
    data: TransformRequest, request: Request[Any, Any, Any]
) -> TransformResponse:
    state = _state(request)
    request_id = data.request_id or request_id_for(request)
    max_retries = data.verification.max_retries if data.verification else None

    parent_attrs = {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_TRANSDUCE,
        GEN_AI_REQUEST_MODEL: state.backend.model,
        TRANSDUCE_MODE_ID: _mode_label(data.mode),
    }
    with state.span_emitter.span(SPAN_REQUEST, parent_attrs) as request_span:
        with state.span_emitter.span(SPAN_SCAN) as scan_span:
            injection_match = state.injection_scanner.scan(data.text)
            if injection_match is not None:
                scan_span.set_attribute(TRANSDUCE_SCAN_MATCHED_PATTERN, injection_match.category)
                state.metrics.injection_detected_total.labels(
                    category=injection_match.category
                ).inc()
                raise InputInjectionDetectedError(injection_match)
            scan_span.set_attribute(TRANSDUCE_SCAN_MATCHED_PATTERN, "clean")

        detected_language = state.language_detector.detect(data.text)
        request_span.set_attribute(TRANSDUCE_LANGUAGE, detected_language)
        mode_ids = [data.mode] if isinstance(data.mode, str) else list(data.mode)
        for mode_ref in mode_ids:
            mode_spec = state.registry.resolve(mode_ref)
            if detected_language not in mode_spec.supported_languages:
                state.metrics.language_unsupported_total.labels(
                    mode=mode_spec.id, lang=detected_language
                ).inc()
                raise LanguageNotSupportedError(
                    detected=detected_language,
                    supported=mode_spec.supported_languages,
                    mode_id=mode_spec.id,
                )
            enforce_min_model_b(
                mode_id=mode_spec.id,
                required_b=mode_spec.backend_requirements.min_model_b,
                backend_id=state.backend_id,
                backend_model=state.backend.model,
                declared_b=state.backend_model_size_b,
            )

        try:
            result = await state.orchestrator.transform(
                text=data.text,
                mode=data.mode,
                intensity=data.intensity,
                preserve=_coerce_preserve(data.preserve),
                max_retries=max_retries,
                request_id=request_id,
                language=detected_language,
            )
        except ConcurrencyLimitExceededError as exc:
            state.metrics.concurrency_rejections_total.labels(backend=exc.backend_id).inc()
            state.metrics.requests_total.labels(mode=_mode_label(data.mode), verdict="error").inc()
            raise
        except Exception:
            state.metrics.requests_total.labels(mode=_mode_label(data.mode), verdict="error").inc()
            raise

        response_mode: ModeRef | list[ModeRef]
        if isinstance(result.mode, tuple):
            response_mode = list(result.mode)
            metric_mode_label = "+".join(ref.id for ref in result.mode)
        else:
            response_mode = result.mode
            metric_mode_label = result.mode.id

        request_span.set_attribute(TRANSDUCE_VERDICT, "accept")
        request_span.set_attribute(TRANSDUCE_RETRIES, result.retries)
        request_span.set_attribute(TRANSDUCE_COST_USD, result.cost.usd_total)

        state.metrics.requests_total.labels(mode=metric_mode_label, verdict="accept").inc()
        state.metrics.generation_duration_ms.labels(
            backend=result.backend_used.provider, mode=metric_mode_label
        ).observe(result.timing.generate_ms)
        if result.cost.usd_total > 0.0:
            state.metrics.generation_cost_usd_total.labels(
                backend=result.backend_used.provider, mode=metric_mode_label
            ).inc(result.cost.usd_total)

        return TransformResponse(
            request_id=request_id,
            mode=response_mode,
            language=result.language,
            original=result.original,
            transformed=result.transformed,
            diff=list(result.diff),
            scores=result.scores,
            backend_used=result.backend_used,
            timing=result.timing,
            retries=result.retries,
            cost=result.cost,
            composite_score=result.composite_score,
        )


@post("/v1/transform/stream", status_code=HTTPStatus.OK)
async def post_transform_stream(
    data: TransformRequest, request: Request[Any, Any, Any]
) -> ServerSentEvent | Response[dict[str, Any]]:
    """Advisory streaming transform (P3-STR-01..02).

    Forwards each backend text-delta to the client as it arrives and
    emits a final ``verdict`` event after the verifier runs once on
    the accumulated text. Strict streaming returns 400
    ``not_implemented`` per ``docs/system-design.md``; ``streaming:
    off`` is rejected at this endpoint to keep the contract honest
    (the non-streaming endpoint exists for that case).
    """
    state = _state(request)
    request_id = data.request_id or request_id_for(request)

    if data.streaming is StreamingMode.STRICT:
        return _streaming_error_response(
            request_id=request_id,
            code=ErrorCode.NOT_IMPLEMENTED,
            message=(
                "strict verification with token streaming is architecturally "
                "incompatible; use POST /v1/transform for strict verification"
            ),
            status=HTTPStatus.BAD_REQUEST,
        )
    if data.streaming is not StreamingMode.ADVISORY:
        return _streaming_error_response(
            request_id=request_id,
            code=ErrorCode.VALIDATION_ERROR,
            message=(
                "POST /v1/transform/stream requires streaming=advisory; "
                "use POST /v1/transform for non-streaming requests"
            ),
            status=HTTPStatus.BAD_REQUEST,
        )
    if isinstance(data.mode, list):
        return _streaming_error_response(
            request_id=request_id,
            code=ErrorCode.NOT_IMPLEMENTED,
            message="compose chains are not supported on the streaming endpoint",
            status=HTTPStatus.BAD_REQUEST,
        )
    if not state.backend.capabilities.streaming:
        return _streaming_error_response(
            request_id=request_id,
            code=ErrorCode.NOT_IMPLEMENTED,
            message=(
                f"backend {state.backend.name!r} does not support streaming; use POST /v1/transform"
            ),
            status=HTTPStatus.BAD_REQUEST,
        )

    injection_match = state.injection_scanner.scan(data.text)
    if injection_match is not None:
        state.metrics.injection_detected_total.labels(category=injection_match.category).inc()
        raise InputInjectionDetectedError(injection_match)

    detected_language = state.language_detector.detect(data.text)
    mode_id = data.mode if isinstance(data.mode, str) else data.mode[0]
    mode_spec = state.registry.resolve(mode_id)
    if detected_language not in mode_spec.supported_languages:
        state.metrics.language_unsupported_total.labels(
            mode=mode_spec.id, lang=detected_language
        ).inc()
        raise LanguageNotSupportedError(
            detected=detected_language,
            supported=mode_spec.supported_languages,
            mode_id=mode_spec.id,
        )
    enforce_min_model_b(
        mode_id=mode_spec.id,
        required_b=mode_spec.backend_requirements.min_model_b,
        backend_id=state.backend_id,
        backend_model=state.backend.model,
        declared_b=state.backend_model_size_b,
    )

    return ServerSentEvent(
        _streaming_event_messages(
            request_id=request_id,
            data=data,
            spec=mode_spec,
            state=state,
            language=detected_language,
        )
    )


# Maps backend / orchestrator exceptions onto SSE error-event ErrorCodes
# so a streaming client can branch on cause the same way a non-streaming
# client branches on the HTTP status / `error` envelope. Once SSE has
# started streaming we cannot change the 200 response status, so the
# error-event payload is the only signal — keep it accurate.
_STREAM_ERROR_CODES: tuple[tuple[type[BaseException], ErrorCode], ...] = (
    (BudgetExceededError, ErrorCode.BUDGET_EXCEEDED),
    (BackendUnavailableError, ErrorCode.BACKEND_UNAVAILABLE),
    (GenerationTimeoutError, ErrorCode.TIMEOUT),
    (GenerationFailedError, ErrorCode.GENERATION_FAILED),
)


async def _streaming_event_messages(
    *,
    request_id: str,
    data: TransformRequest,
    spec: ModeSpec,
    state: TransduceState,
    language: str,
) -> AsyncIterator[ServerSentEventMessage]:
    """Convert :class:`stream_transform` events to SSE messages."""
    metric_mode_label = spec.id
    parent_attrs = {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_TRANSDUCE,
        GEN_AI_REQUEST_MODEL: state.backend.model,
        TRANSDUCE_MODE_ID: spec.id,
        TRANSDUCE_LANGUAGE: language,
    }
    with state.span_emitter.span(SPAN_REQUEST, parent_attrs) as request_span:
        try:
            async for event in stream_transform(
                text=data.text,
                spec=spec,
                backend=state.backend,
                verifier=state.verifier,
                intensity=data.intensity,
                preserve=_coerce_preserve(data.preserve),
                span_emitter=state.span_emitter,
            ):
                if isinstance(event, StreamChunkEvent):
                    yield ServerSentEventMessage(
                        data=json.dumps({"text": event.text}),
                        event="chunk",
                    )
                elif isinstance(event, StreamVerdictEvent):
                    payload = {
                        "request_id": request_id,
                        "mode": event.mode.model_dump(mode="json"),
                        "language": language,
                        "verdict": event.verdict,
                        "transformed": event.transformed,
                        "rejection_reason": event.rejection_reason,
                        "diff": list(event.diff),
                        "scores": event.scores,
                        "tokens_in": event.tokens_in,
                        "tokens_out": event.tokens_out,
                        "timing_ms": event.timing_ms,
                    }
                    request_span.set_attribute(TRANSDUCE_VERDICT, event.verdict)
                    yield ServerSentEventMessage(
                        data=json.dumps(payload),
                        event="verdict",
                    )
        except ConcurrencyLimitExceededError as exc:
            state.metrics.concurrency_rejections_total.labels(backend=exc.backend_id).inc()
            state.metrics.requests_total.labels(mode=metric_mode_label, verdict="error").inc()
            yield _stream_error_event(
                request_id=request_id,
                code=ErrorCode.CONCURRENCY_LIMIT_EXCEEDED,
                message=str(exc),
            )
            return
        except Exception as exc:
            state.metrics.requests_total.labels(mode=metric_mode_label, verdict="error").inc()
            code = _classify_stream_exception(exc)
            yield _stream_error_event(
                request_id=request_id,
                code=code,
                message=str(exc) or code.value,
            )
            return

        state.metrics.requests_total.labels(mode=metric_mode_label, verdict="accept").inc()


def _classify_stream_exception(exc: BaseException) -> ErrorCode:
    """Map a backend / orchestrator exception to its SSE error-event code."""
    for exc_type, code in _STREAM_ERROR_CODES:
        if isinstance(exc, exc_type):
            return code
    return ErrorCode.GENERATION_FAILED


def _streaming_error_response(
    *,
    request_id: str,
    code: ErrorCode,
    message: str,
    status: HTTPStatus,
) -> Response[dict[str, Any]]:
    envelope = TransformError(request_id=request_id, error=code, message=message)
    return Response(envelope.model_dump(mode="json"), status_code=status)


def _stream_error_event(
    *, request_id: str, code: ErrorCode, message: str
) -> ServerSentEventMessage:
    payload = {
        "request_id": request_id,
        "error": code.value,
        "message": message,
    }
    return ServerSentEventMessage(data=json.dumps(payload), event="error")


@get("/v1/modes")
async def list_modes(request: Request[Any, Any, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "modes": [_mode_to_dict(spec) for spec in _state(request).registry.list_modes()],
    }


@get("/v1/modes/{mode_id:str}")
async def get_mode(mode_id: str, request: Request[Any, Any, Any]) -> dict[str, Any]:
    return _mode_to_dict(_state(request).registry.resolve(mode_id))


@get("/v1/backends")
async def list_backends(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    backend_info = BackendInfo(provider=state.backend.name, model=state.backend.model)
    return {
        "default": state.config.backends.default,
        "backends": [backend_info.model_dump()],
    }


@get("/v1/scorers")
async def list_scorers(request: Request[Any, Any, Any]) -> dict[str, list[str]]:
    return {"scorers": [scorer.name for scorer in _state(request).verifier.scorers]}


@get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@get("/readyz")
async def readyz(request: Request[Any, Any, Any]) -> Response[dict[str, Any]]:
    state = _state(request)
    health = await state.backend.health()
    body: dict[str, Any] = {
        "backend": {"healthy": health.healthy, "detail": health.detail},
        "modes": [spec.id for spec in state.registry.list_modes()],
    }
    if not health.healthy:
        return Response(body, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
    return Response(body, status_code=HTTPStatus.OK)


@get("/metrics")
async def metrics(request: Request[Any, Any, Any]) -> Response[bytes]:
    payload = generate_latest(_state(request).metrics.registry)
    return Response(
        payload,
        media_type=CONTENT_TYPE_LATEST,
        status_code=HTTPStatus.OK,
    )


def _mode_label(mode: str | list[str]) -> str:
    if isinstance(mode, list):
        return "compose"
    return mode


__all__ = [
    "get_mode",
    "healthz",
    "list_backends",
    "list_modes",
    "list_scorers",
    "metrics",
    "post_transform",
    "post_transform_stream",
    "readyz",
]
