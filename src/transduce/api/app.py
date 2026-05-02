"""Litestar application factory for transduce v0 (P1-API-01..04)."""

from __future__ import annotations

import uuid
from typing import Any

from litestar import Litestar, Request
from litestar.exceptions import ClientException, ValidationException

from transduce.api.errors import (
    client_exception_handler,
    domain_exception_handler,
    validation_exception_handler,
)
from transduce.api.handlers import (
    get_mode,
    healthz,
    list_backends,
    list_modes,
    list_scorers,
    metrics,
    post_transform,
    readyz,
)
from transduce.api.state import TransduceMetrics, TransduceState
from transduce.backends.base import Backend
from transduce.backends.ollama import OllamaBackend
from transduce.config.schema import Config
from transduce.injection.scanner import InjectionScanner
from transduce.language.detector import LanguageDetector
from transduce.pipeline.orchestrator import Orchestrator
from transduce.registry.static import StaticRegistry, build_default_registry
from transduce.verification.base import Scorer
from transduce.verification.composite import CompositeVerifier
from transduce.verification.pipeline import VerifierPipeline


def attach_request_id(request: Request[Any, Any, Any]) -> None:
    """Stamp every inbound request with a server-side ``request_id``.

    Honours an inbound ``X-Request-ID`` header so upstream tracing IDs
    survive the round-trip; otherwise mints a fresh UUID hex string.
    """
    incoming = request.headers.get("x-request-id")
    request.state.request_id = incoming or uuid.uuid4().hex


def create_app(
    config: Config,
    *,
    backend: Backend | None = None,
    registry: StaticRegistry | None = None,
    scorers: list[Scorer] | None = None,
    composite_verifier: CompositeVerifier | None = None,
    metrics_state: TransduceMetrics | None = None,
    injection_scanner: InjectionScanner | None = None,
    language_detector: LanguageDetector | None = None,
) -> Litestar:
    """Build a Litestar app wired against ``config`` and optional overrides.

    Tests inject ``backend``, ``registry``, ``scorers``,
    ``composite_verifier``, and ``language_detector`` to avoid hitting
    fastembed, spaCy, and lingua; production wiring synthesises the real
    implementations from config (commit follow-ups: see CLI ``serve``).

    The default composite verifier reuses ``scorers`` with a threshold
    of ``cosine_min - 0.05`` per docs/system-design.md §Composite
    Verifier; operators can pass an explicit ``composite_verifier`` for
    a custom scorer set or threshold.
    """
    if scorers is None:
        raise ValueError("scorers must be supplied; production wiring builds them from config")

    resolved_registry = registry or build_default_registry()
    resolved_backend = backend or _build_default_backend(config)
    verifier = VerifierPipeline(scorers)
    resolved_composite = composite_verifier or CompositeVerifier(
        scorers=scorers,
        threshold=max(0.0, config.verification.default_cosine_min - 0.05),
    )
    orchestrator = Orchestrator(
        registry=resolved_registry,
        backend=resolved_backend,
        verifier=verifier,
        budget_config=config.budget,
        composite_verifier=resolved_composite,
        default_max_retries=config.verification.max_retries,
    )
    resolved_detector = language_detector or LanguageDetector(
        languages=config.language.languages,
        default=config.language.default,
        min_confidence=config.language.min_confidence,
    )

    app_state = TransduceState(
        config=config,
        registry=resolved_registry,
        backend=resolved_backend,
        verifier=verifier,
        orchestrator=orchestrator,
        metrics=metrics_state or TransduceMetrics.build(),
        injection_scanner=injection_scanner or InjectionScanner(),
        language_detector=resolved_detector,
    )

    async def shutdown(app_instance: Litestar) -> None:
        aclose = getattr(app_instance.state.transduce_state.backend, "aclose", None)
        if callable(aclose):
            await aclose()

    litestar_app = Litestar(
        route_handlers=[
            post_transform,
            list_modes,
            get_mode,
            list_backends,
            list_scorers,
            healthz,
            readyz,
            metrics,
        ],
        before_request=attach_request_id,
        exception_handlers={
            ValidationException: validation_exception_handler,
            ClientException: client_exception_handler,
            Exception: domain_exception_handler,
        },
        on_shutdown=[shutdown],
        debug=False,
    )
    litestar_app.state.transduce_state = app_state
    return litestar_app


def _build_default_backend(config: Config) -> Backend:
    """Build the default Ollama backend from ``config.backends`` selection.

    Phase-3 widens ``BackendEntry`` with cloud providers and optional
    endpoints; the v1 backend dispatch table replaces this helper. Until
    then the wiring narrows ``endpoint`` for the ollama path the
    validator already guarantees, and rejects anything else loudly.
    """
    default_id = config.backends.default
    for entry in config.backends.registry:
        if entry.id != default_id:
            continue
        if entry.provider != "ollama":
            raise RuntimeError(
                f"backend {entry.id!r}: provider {entry.provider!r} not yet wired in app factory"
            )
        if entry.endpoint is None:  # pragma: no cover — validator enforces this for ollama
            raise RuntimeError(f"backend {entry.id!r}: ollama provider requires an endpoint")
        return OllamaBackend(endpoint=entry.endpoint, model=entry.model)
    raise RuntimeError(f"backends.default {default_id!r} missing from registry at app build time")


__all__ = ["attach_request_id", "create_app"]
