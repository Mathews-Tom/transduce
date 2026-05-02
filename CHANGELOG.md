# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.0.1] - 2026-05-02

First public preview shipping the v0 surface from `.docs/development-plan.md`. Local-first transformation primitive with cosine + preservation verification, single-mode dispatch, and three seed modes.

### Added

- `POST /v1/transform` Litestar route with Pydantic v2 request/response models, request-id propagation via `X-Request-ID`, and a structured `TransformError` envelope mapping internal exceptions to stable `ErrorCode` values (`mode_not_found`, `backend_unavailable`, `verification_failed`, `input_too_long`, `generation_failed`, `not_implemented`, `timeout`, `validation_error`).
- Catalog endpoints `GET /v1/modes`, `GET /v1/modes/{id}`, `GET /v1/backends`, and `GET /v1/scorers`; health endpoints `GET /healthz` and `GET /readyz` with readiness probing the configured backend; Prometheus metrics at `GET /metrics` with `transduce_requests_total`, `transduce_generation_duration_ms`, and `transduce_verification_failures_total`.
- Five-stage pipeline orchestrator (resolve → generate → verify → retry → diff) with targeted-feedback retry — the next prompt carries the failed scorer name, rejection reason, and offending span (CRITIC-style external feedback per docs/system-design.md §Verification Subsystem). Compose chains return `not_implemented`; max retries default 3, hard ceiling 5.
- `OllamaBackend` adapter against `/api/generate` and `/api/tags` with httpx async client; failures map onto the `BackendUnavailableError`, `GenerationTimeoutError`, and `GenerationFailedError` exception hierarchy.
- Static mode registry seeded with three in-tree modes: `dejargon` (cosine_min 0.85, min_model_b 14), `register.casual` (cosine_min 0.82), and `length.normalize` (cosine_min 0.78). The default backend model in `transduce.example.yaml` is `gemma4`. All modes ship Jinja2 prompt templates compiled at registry construction.
- Verification ensemble: `CosineSimilarityScorer` (fastembed `bge-small-en-v1.5`), `EntityPreservationScorer` (spaCy `en_core_web_sm`, exact substring match), `NumberPreservationScorer` (decimal-aware with magnitude-word collapse — `94B` ≡ `94 billion`), `UrlPreservationScorer`. Sequential pipeline short-circuits on first reject and surfaces the failing scorer name and span.
- Word-level diff via `diff-match-patch` with semantic cleanup; structured `DiffOp` operations let clients render however they need.
- YAML config loader with `${VAR:-default}` env-var substitution and Pydantic schema validation; `transduce.example.yaml` mirrors the v0 schema exactly.
- CLI entry point `transduce serve --config <path>` builds the production scorer set and runs uvicorn against the Litestar app.
- Integration test suite under `tests/integration/test_transform_real_ollama.py`; auto-skips when Ollama is unreachable.

### Project metadata

- Bumped version from `0.0.0` to `0.0.1`.
- Runtime dependencies pinned: `pydantic>=2.10,<3`, `litestar[standard]>=2.20,<3`, `httpx>=0.27,<1`, `jinja2>=3.1,<4`, `diff-match-patch>=20241021`, `fastembed>=0.5,<1`, `spacy>=3.7,<4`, `pyyaml>=6.0,<7`, `prometheus-client>=0.20,<1`, `uvicorn[standard]>=0.30,<1`, `click>=8.1,<9`.

### Deferred

- Integration suite running against Ollama in CI is staged for a follow-up; the test code is committed and runs locally when `OLLAMA_HOST` is reachable.
- End-to-end suite against a `docker compose` stack (transduce + Ollama) is staged for a follow-up.
- Performance gate (p50 < 2s, p99 < 5s) requires the CI integration job to land first.

[Unreleased]: https://github.com/Mathews-Tom/transduce/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Mathews-Tom/transduce/releases/tag/v0.0.1
