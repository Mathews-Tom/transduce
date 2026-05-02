"""Pydantic schema for the v1 service configuration.

Mirrors docs/system-design.md §Configuration. The v1 release widens
``BackendEntry`` with cloud providers, ``api_key_env``, explicit cost
overrides, and per-backend timeouts (P3-BACK-01..P3-BACK-07); adds a
``budget`` section for the per-request cost guard (P3-BUDG-01); widens
``language`` with the lingua detector configuration (P3-LANG-01..02);
and lands an ``observability`` section as a config slot for the OTel
GenAI SemConv emission that ships in a follow-up branch (P3-OBS-*).

Each section sets ``extra="forbid"`` so typos surface as validation
errors at startup rather than silently degrading behaviour.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderName = Literal[
    "ollama",
    "anthropic",
    "openai_compat",
    "vllm",
    "llama_cpp",
    "litellm",
]
"""Names of every backend provider transduce ships an adapter for."""

_REMOTE_PROVIDERS: frozenset[str] = frozenset({"anthropic", "litellm"})
_ENDPOINT_PROVIDERS: frozenset[str] = frozenset({"ollama", "openai_compat", "vllm", "llama_cpp"})


class ServiceConfig(BaseModel):
    """Top-level service runtime parameters."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(
        default="0.0.0.0",  # noqa: S104  # nosec B104 — server bind default per docs/system-design.md §Configuration
        min_length=1,
    )
    port: int = Field(default=8080, ge=1, le=65_535)
    request_timeout_s: float = Field(default=30.0, gt=0.0)
    max_input_chars: int = Field(default=50_000, gt=0)
    max_retries_default: int = Field(default=3, ge=0, le=5)


class BackendEntry(BaseModel):
    """One backend registry entry covering local and cloud providers (P3-BACK-01..07).

    ``endpoint`` is required for local/self-hosted backends (ollama, vllm,
    llama_cpp, openai_compat) and optional for SaaS backends that derive
    their endpoint from the SDK (anthropic, litellm). ``api_key_env`` is
    required for cloud backends so secrets stay in environment variables
    per the security policy. ``cost_in_per_million_usd`` /
    ``cost_out_per_million_usd`` let an operator pin pricing for budget
    accounting; backends with built-in pricing tables ignore them when
    unset.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    provider: ProviderName
    model: str = Field(min_length=1)
    endpoint: str | None = Field(default=None, min_length=1)
    concurrency_limit: int = Field(default=1, ge=1)
    api_key_env: str | None = Field(default=None, min_length=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    cost_in_per_million_usd: float | None = Field(default=None, ge=0.0)
    cost_out_per_million_usd: float | None = Field(default=None, ge=0.0)
    model_size_b: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Backend model size in billions of parameters used by the "
            "min_model_b precondition (P3-BACK-09). Operators must declare "
            "this for any backend they want to serve modes with a min_model_b "
            "floor; the precondition raises 412 when the declared size is "
            "below the mode's requirement, or when the mode requires a floor "
            "and the backend leaves model_size_b unset."
        ),
    )
    prompt_alias: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Backend-side alias used by mode prompt-override dispatch (P3-BACK-08). "
            "When set, modes can supply an override under ``prompt.<mode>.<alias>``."
        ),
    )

    @model_validator(mode="after")
    def _validate_endpoint_and_auth(self) -> BackendEntry:
        if self.provider in _ENDPOINT_PROVIDERS and not self.endpoint:
            raise ValueError(
                f"backend {self.id!r}: provider {self.provider!r} requires an endpoint"
            )
        if self.provider in _REMOTE_PROVIDERS and not self.api_key_env:
            raise ValueError(
                f"backend {self.id!r}: provider {self.provider!r} requires api_key_env"
            )
        return self


class BackendsConfig(BaseModel):
    """Backends section: a default id and a non-empty registry."""

    model_config = ConfigDict(extra="forbid")

    default: str = Field(min_length=1)
    registry: list[BackendEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _default_must_be_registered(self) -> BackendsConfig:
        registered = {entry.id for entry in self.registry}
        if self.default not in registered:
            raise ValueError(
                f"backends.default {self.default!r} is not present in backends.registry "
                f"(known: {sorted(registered)})"
            )
        duplicates = [
            backend_id
            for backend_id, count in Counter(entry.id for entry in self.registry).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(f"duplicate backend ids in registry: {sorted(duplicates)}")
        return self


class VerificationConfig(BaseModel):
    """Verifier subsystem defaults for the v0.5 ensemble.

    Per-mode ``VerifierProfile`` overrides take precedence; these defaults
    apply to any mode that does not declare its own threshold.
    ``injection_scanner`` is the name of the regex pack the operator is
    running with; v0.5 ships ``regex-pack-v0.5`` in core.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    default_cosine_min: float = Field(default=0.85, ge=0.0, le=1.0)
    default_nli_min: float = Field(default=0.70, ge=0.0, le=1.0)
    default_hhem_min: float = Field(default=0.50, ge=0.0, le=1.0)
    reject_on_negation_diff: bool = True
    injection_scanner: str = Field(default="regex-pack-v0.5", min_length=1)
    max_retries: int = Field(default=3, ge=0, le=5)


class BudgetConfig(BaseModel):
    """Per-request cost guard (P3-BUDG-01..03).

    ``max_cost_per_request_usd`` caps the cumulative cost of a transform
    across retries. Per-request overrides via ``TransformRequest`` widen
    or tighten the cap when present. ``abort_on_non_improving_trend``
    short-circuits the retry loop when verifier scores fail to improve
    over ``non_improving_window`` consecutive attempts.
    """

    model_config = ConfigDict(extra="forbid")

    max_cost_per_request_usd: float = Field(default=0.05, ge=0.0)
    abort_on_non_improving_trend: bool = True
    non_improving_window: int = Field(default=3, ge=2)


class LanguageConfig(BaseModel):
    """Language detection and routing (P3-LANG-01..02).

    ``detector`` selects the implementation the ingress detector wraps.
    ``languages`` is the explicit set the detector loads at startup;
    keeping it small (e.g., ("en", "de", "fr")) keeps lingua's load
    footprint and per-call latency in budget. ``default`` is the
    fall-back returned when the detector cannot identify the input above
    the configured confidence floor.
    """

    model_config = ConfigDict(extra="forbid")

    default: str = Field(default="en", min_length=1)
    detector: Literal["lingua"] = "lingua"
    languages: tuple[str, ...] = Field(default=("en",), min_length=1)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _default_must_be_loaded(self) -> LanguageConfig:
        if self.default not in self.languages:
            raise ValueError(
                f"language.default {self.default!r} must be in language.languages "
                f"(loaded: {list(self.languages)})"
            )
        return self


class ObservabilityConfig(BaseModel):
    """OTel GenAI SemConv slot (P3-OBS-01..05).

    The v1 substrate lands the configuration surface so operators can
    declare their OTel posture without waiting for the emission code that
    ships in a follow-up branch. ``redact_text_in_spans`` is the
    enforcement flag that bans raw text and ``last_candidate`` from span
    attributes per docs/system-design.md §Observability;
    ``debug_include_text`` is the explicit opt-in that re-enables raw
    text for non-prod debugging.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    otel_endpoint: str | None = Field(default=None, min_length=1)
    semconv: Literal["gen_ai"] = "gen_ai"
    redact_text_in_spans: bool = True
    debug_include_text: bool = False

    @model_validator(mode="after")
    def _debug_text_requires_redaction_off(self) -> ObservabilityConfig:
        if self.debug_include_text and self.redact_text_in_spans:
            raise ValueError(
                "observability.debug_include_text=true requires "
                "observability.redact_text_in_spans=false"
            )
        return self


class ModePackageEntry(BaseModel):
    """One pinned mode package in the allowlist (P2-PLG-01).

    The same package name may appear multiple times in the allowlist with
    different versions; the multi-version dispatch loader exposes both
    revisions simultaneously and clients pin via ``mode: "id@version"``
    (P3-VER-01..02).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    path: str = Field(min_length=1, description="Filesystem path to the package root.")
    signed_by: str | None = Field(default=None, min_length=1)


class ModesConfig(BaseModel):
    """Mode-loader configuration (P2-PLG-01..P2-PLG-03, P3-VER-01..02).

    ``source`` defaults to ``allowlist``. Setting it to ``auto`` enables
    legacy entry-point discovery and emits a startup warning per the
    dev plan; production deployments must keep it on ``allowlist``.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["allowlist", "auto"] = "allowlist"
    enforce_signing: bool = False
    packages: list[ModePackageEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_duplicate_name_version(self) -> ModesConfig:
        seen: set[tuple[str, str]] = set()
        for pkg in self.packages:
            key = (pkg.name, pkg.version)
            if key in seen:
                raise ValueError(
                    f"duplicate mode package entry: {pkg.name} {pkg.version} "
                    "(distinct versions of the same package are allowed; identical pairs are not)"
                )
            seen.add(key)
        return self


class Config(BaseModel):
    """Full v1 service configuration."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    modes: ModesConfig = Field(default_factory=ModesConfig)
    backends: BackendsConfig
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
