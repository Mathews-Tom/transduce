"""HTTP route handlers for the v0 API surface."""

from __future__ import annotations

from collections.abc import Iterable
from http import HTTPStatus
from typing import Any

from litestar import Request, Response, get, post
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from transduce.api.errors import request_id_for
from transduce.api.schemas import (
    BackendInfo,
    TransformRequest,
    TransformResponse,
)
from transduce.api.state import TransduceState
from transduce.backends.concurrency import ConcurrencyLimitExceededError
from transduce.injection.scanner import InputInjectionDetectedError
from transduce.language.detector import LanguageNotSupportedError
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

    injection_match = state.injection_scanner.scan(data.text)
    if injection_match is not None:
        state.metrics.injection_detected_total.labels(category=injection_match.category).inc()
        raise InputInjectionDetectedError(injection_match)

    detected_language = state.language_detector.detect(data.text)
    if isinstance(data.mode, str):
        mode_id_for_lang_check = data.mode
        mode_spec = state.registry.resolve(mode_id_for_lang_check)
        if detected_language not in mode_spec.supported_languages:
            state.metrics.language_unsupported_total.labels(
                mode=mode_id_for_lang_check, lang=detected_language
            ).inc()
            raise LanguageNotSupportedError(
                detected=detected_language,
                supported=mode_spec.supported_languages,
                mode_id=mode_id_for_lang_check,
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

    state.metrics.requests_total.labels(mode=result.mode.id, verdict="accept").inc()
    state.metrics.generation_duration_ms.labels(
        backend=result.backend_used.provider, mode=result.mode.id
    ).observe(result.timing.generate_ms)

    return TransformResponse(
        request_id=request_id,
        mode=result.mode,
        language=result.language,
        original=result.original,
        transformed=result.transformed,
        diff=list(result.diff),
        scores=result.scores,
        backend_used=result.backend_used,
        timing=result.timing,
        retries=result.retries,
        cost=result.cost,
    )


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
    "readyz",
]
