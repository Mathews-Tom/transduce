"""Application-state container shared across API routes."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram

from transduce.backends.base import Backend
from transduce.config.schema import Config
from transduce.injection.scanner import InjectionScanner
from transduce.pipeline.orchestrator import Orchestrator
from transduce.registry.static import StaticRegistry
from transduce.verification.pipeline import VerifierPipeline


@dataclass
class TransduceState:
    """Concrete dependency bundle wired into the Litestar app."""

    config: Config
    registry: StaticRegistry
    backend: Backend
    verifier: VerifierPipeline
    orchestrator: Orchestrator
    metrics: TransduceMetrics
    injection_scanner: InjectionScanner


@dataclass
class TransduceMetrics:
    """Prometheus collectors emitted from the API surface."""

    registry: CollectorRegistry
    requests_total: Counter
    generation_duration_ms: Histogram
    verification_failures_total: Counter
    injection_detected_total: Counter

    @classmethod
    def build(cls) -> TransduceMetrics:
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests_total=Counter(
                "transduce_requests_total",
                "Total transduce transform requests by mode and verdict.",
                labelnames=("mode", "verdict"),
                registry=registry,
            ),
            generation_duration_ms=Histogram(
                "transduce_generation_duration_ms",
                "Generation latency in milliseconds keyed by backend and mode.",
                labelnames=("backend", "mode"),
                registry=registry,
            ),
            verification_failures_total=Counter(
                "transduce_verification_failures_total",
                "Verification rejections grouped by mode and failed scorer.",
                labelnames=("mode", "scorer"),
                registry=registry,
            ),
            injection_detected_total=Counter(
                "transduce_injection_detected_total",
                "Ingress injection scanner matches grouped by category.",
                labelnames=("category",),
                registry=registry,
            ),
        )
