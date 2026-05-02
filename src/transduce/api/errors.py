"""Exception → HTTP response mapping for the API layer (P1-API-06).

Centralises the bridge from internal exception classes to the documented
``TransformError`` envelope. The mapping is data-driven so adding a new
exception class only requires one row, not a new handler function.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from litestar import Request, Response
from litestar.exceptions import ClientException, ValidationException

from transduce.api.schemas import ErrorCode, TransformError
from transduce.backends.base import (
    BackendUnavailableError,
    GenerationFailedError,
    GenerationTimeoutError,
)
from transduce.backends.concurrency import ConcurrencyLimitExceededError
from transduce.injection.scanner import InputInjectionDetectedError
from transduce.language.detector import LanguageNotSupportedError
from transduce.pipeline.orchestrator import (
    CompositionNotImplementedError,
    VerificationFailedError,
)
from transduce.registry.static import ModeNotFoundError, ModeVersionNotFoundError

_EXCEPTION_MAPPING: Mapping[type[BaseException], tuple[ErrorCode, int]] = {
    CompositionNotImplementedError: (ErrorCode.NOT_IMPLEMENTED, HTTPStatus.BAD_REQUEST),
    ModeVersionNotFoundError: (ErrorCode.MODE_VERSION_NOT_FOUND, HTTPStatus.NOT_FOUND),
    ModeNotFoundError: (ErrorCode.MODE_NOT_FOUND, HTTPStatus.NOT_FOUND),
    BackendUnavailableError: (
        ErrorCode.BACKEND_UNAVAILABLE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ),
    GenerationTimeoutError: (ErrorCode.TIMEOUT, HTTPStatus.GATEWAY_TIMEOUT),
    GenerationFailedError: (ErrorCode.GENERATION_FAILED, HTTPStatus.BAD_GATEWAY),
}


def request_id_for(request: Request[Any, Any, Any]) -> str:
    rid = getattr(request.state, "request_id", None)
    if isinstance(rid, str) and rid:
        return rid
    return uuid.uuid4().hex


def validation_exception_handler(
    request: Request[Any, Any, Any], exc: ValidationException
) -> Response[dict[str, Any]]:
    extras = exc.extra if isinstance(exc.extra, list) else []
    if _is_input_too_long(extras):
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.INPUT_TOO_LONG,
            message="text exceeds the configured maximum length",
            details={"errors": extras},
        )
    else:
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.VALIDATION_ERROR,
            message=exc.detail,
            details={"errors": extras} if extras else None,
        )
    return Response(envelope.model_dump(mode="json"), status_code=HTTPStatus.BAD_REQUEST)


def client_exception_handler(
    request: Request[Any, Any, Any], exc: ClientException
) -> Response[dict[str, Any]]:
    """Map malformed-JSON / generic 4xx Litestar errors onto the validation envelope."""
    envelope = TransformError(
        request_id=request_id_for(request),
        error=ErrorCode.VALIDATION_ERROR,
        message=exc.detail or "client request rejected",
    )
    return Response(envelope.model_dump(mode="json"), status_code=HTTPStatus.BAD_REQUEST)


def domain_exception_handler(
    request: Request[Any, Any, Any], exc: Exception
) -> Response[dict[str, Any]]:
    if isinstance(exc, VerificationFailedError):
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.VERIFICATION_FAILED,
            message=str(exc),
            last_candidate=exc.last_candidate,
            scores=exc.scores,
            details=({"failed_scorer": exc.rejection_reason} if exc.rejection_reason else None),
        )
        return Response(
            envelope.model_dump(mode="json"),
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    if isinstance(exc, InputInjectionDetectedError):
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.INPUT_INJECTION_DETECTED,
            message=str(exc),
            details={
                "matched_pattern": exc.match.pattern,
                "category": exc.match.category,
                "span": exc.match.span,
            },
        )
        return Response(
            envelope.model_dump(mode="json"),
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    if isinstance(exc, LanguageNotSupportedError):
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.LANGUAGE_NOT_SUPPORTED,
            message=str(exc),
            details={
                "detected": exc.detected,
                "supported": list(exc.supported),
                "mode_id": exc.mode_id,
            },
        )
        return Response(
            envelope.model_dump(mode="json"),
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )
    if isinstance(exc, ConcurrencyLimitExceededError):
        envelope = TransformError(
            request_id=request_id_for(request),
            error=ErrorCode.CONCURRENCY_LIMIT_EXCEEDED,
            message=str(exc),
            details={
                "backend_id": exc.backend_id,
                "limit": exc.limit,
                "retry_after_s": exc.retry_after_s,
            },
        )
        return Response(
            envelope.model_dump(mode="json"),
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
            headers={"Retry-After": f"{exc.retry_after_s:g}"},
        )
    for exception_type, (code, status) in _EXCEPTION_MAPPING.items():
        if isinstance(exc, exception_type):
            envelope = TransformError(
                request_id=request_id_for(request),
                error=code,
                message=str(exc) or code.value,
            )
            return Response(envelope.model_dump(mode="json"), status_code=status)
    raise exc


def _is_input_too_long(errors: list[dict[str, Any]]) -> bool:
    for entry in errors:
        location_field = entry.get("key") or entry.get("loc") or entry.get("source") or ""
        if not _location_targets_text(location_field):
            continue
        message = str(entry.get("message") or entry.get("msg") or "").lower()
        if "at most" in message or "too long" in message:
            return True
    return False


def _location_targets_text(location: object) -> bool:
    if isinstance(location, str):
        return location == "text"
    if isinstance(location, (list, tuple)):
        return any(part == "text" for part in location)
    return False


__all__ = [
    "client_exception_handler",
    "domain_exception_handler",
    "request_id_for",
    "validation_exception_handler",
]
