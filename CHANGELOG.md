# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-05-03

First production release of the v1 substrate. Lands the streaming, observability, and eval-corpus deliverables that were deferred from `v1.0.0-rc1`. The substrate from rc1 (five backend adapters, compose chains, per-request cost guard, language detection, multi-version mode dispatch, five additional seed modes) ships unchanged.

### Added

- **OTel GenAI SemConv span emission (P3-OBS-01..04).** `transduce.observability` ships `SpanEmitter` and `build_tracer_provider` so the orchestrator and HTTP handlers emit a parent `gen_ai.client.request` span with `transduce.scan`, `transduce.generate`, `transduce.verify`, `transduce.compose`, and `transduce.diff` children. Standard `gen_ai.*` attributes (system, request model, input/output tokens) plus `transduce.*` extensions (mode id, language, verdict, retries, cost, per-scorer values) are emitted on every request. Raw text is banned from span attributes by default; `transduce.text.sha256_8` and `transduce.text.length` are the privacy-default surface, with an opt-in `debug.include_text=true` flag gated by the config validator and a stderr warning at startup.
- **Streaming generate Protocol method (P3-STR-01 substrate).** `Backend` Protocol gains `stream() -> AsyncIterator[StreamChunk]`. The four production adapters — Ollama (NDJSON), Anthropic (SDK events), OpenAI-compat (SSE), LiteLLM router (async iterator) — project their native streaming protocols onto the unified `StreamChunk` union. `BackendCapabilities.streaming` flips to `true` for each.
- **Advisory SSE transform endpoint (P3-STR-01..02).** `POST /v1/transform/stream` returns `text/event-stream` with `event: chunk` per backend text-delta and a final `event: verdict` carrying the verifier outcome, transformed text, diff, scores, and timing. `streaming=advisory` is the only accepted mode at this endpoint; `strict` returns 400 `not_implemented`, `off` returns 400 `validation_error`, compose chains return 400 `not_implemented`, and a backend without `capabilities.streaming` returns 400 `not_implemented`.
- **Streaming client rollback helper (P3-STR-03).** `transduce.streaming` ships `parse_sse_events` (low-level) and `stream_transform` (high-level httpx wrapper) with `ChunkEvent`, `VerdictEvent`, and `ErrorEvent` types. `VerdictEvent.rollback is True` when the verifier rejects, giving callers a one-line discard signal for partially-rendered text. The TypeScript reference SDK ships separately under the `armory` repo.
- **`transduce-faithfulness` v0.2 multilingual subset (eval-suite expansion).** New `tests/eval/transduce_faithfulness_v0_2.jsonl` with 300 records: 202 from v0.1 (English, all six failure-mode categories) plus 98 hand-curated German and French records covering negation, antonym, and fact-drift. Every record declares `language`. The 500-record target stays the v1.5 floor.
- **`transduce-composition` v0.1 corpus.** New `tests/eval/transduce_composition_v0_1.jsonl` with 104 records across `faithful_chain`, `drift_accumulated`, and `intensity_overshoot` categories. Each record carries `(original, stage_1, stage_2, expected_composite_verdict)` and exercises the composite verifier's end-to-end drift detection (P3-COMP-02 / N10).
- **`transduce-injection-resilience` v0.1 expansion to 200 prompts.** Doubles the injection corpus (150 attacks + 50 benign controls) without adding new attack categories; the 3:1 attack:benign ratio is asserted by the structural test gate.
- **`docs/observability.md`.** Documents the span hierarchy, `gen_ai.*` and `transduce.*` attribute reference, redaction policy, OTLP collector recipes (Jaeger, Phoenix, Langfuse, OTLP HTTP), and operator-question trace queries.
- **`docs/api/openapi-v1.json` (frozen).** The v1 OpenAPI surface is checkpointed; `tools/freeze_openapi.py` regenerates it and `test_openapi_v1_frozen_matches_app_routes` blocks silent drift.

### Changed

- `StreamingMode` enum gains `ADVISORY` and `STRICT` members alongside the existing `OFF`. `OFF` remains the default for `POST /v1/transform`.
- `Backend` Protocol gains `stream`. Existing adapters and test stubs add their `stream` implementation; `SemaphoreBackend` holds its permit for the lifetime of the inner stream so a slow client cannot starve the backend permit pool.
- CLI `serve` builds and installs the OTel TracerProvider when `observability.enabled=true` and emits a stderr warning when `debug.include_text=true`.
- Project `version` bumped from `0.5.0` to `1.0.0`. Trove classifier moves from `Development Status :: 2 - Pre-Alpha` to `Development Status :: 4 - Beta`.

### Deferred to v1.5

- Native-speaker review of the multilingual faithfulness pairs and category expansion to tense, number, and entity in non-English locales (target: 500-record corpus).
- Hugging Face dataset hosting for `transduce-faithfulness` (`P3` exit criterion).
- Live Anthropic / vLLM integration suites (CI provisioning of test keys with budget caps).
- Public release announcement coordinated with the published benchmark.
- Per-mode AUROC / detection-rate gates on the eval corpora — currently structural-only under `@pytest.mark.unit`; the slow eval gate ships in v1.5 with `@pytest.mark.eval`.

## [0.5.0] - 2026-05-02

Second public preview shipping the v0.5 ensemble verifier, the Spotlighting injection fence and ingress scanner, and the sha256-pinned mode allowlist with manifest-only modes. Replaces the cosine-only verifier from v0 with an ordered ensemble (cosine → negation diff → bidirectional NLI → HHEM → preservation rules → mode-specific) and surfaces all per-scorer outputs on the response.

### Added

- `NegationDiffScorer` (P2-VER-01) — deterministic floor for the ensemble. Tokenises both texts, extracts negation cues with simple lexical scope, and rejects when the multiset of non-quoted cues differs between original and candidate. Catches the classic `did → did not` flip that cosine reads as a near-paraphrase.
- `BidirectionalNLIScorer` (P2-VER-02) — primary faithfulness signal. Wraps an injectable `Entailer` callable so unit tests run without the heavy NLI model; production wiring constructs a MiniCheck-Flan-T5-Large entailer via `build_minicheck_entailer` per ADR-0003. Both directions (`original ⊨ candidate` and reverse) must clear the threshold for accept; the scorer publishes both direction scores in `details`.
- `HHEMScorer` (P2-VER-03) — Vectara HHEM-2.1 cross-encoder as a complementary factuality signal sourced from a different training distribution than MiniCheck-FT5. Same injectable-callable pattern.
- `DatePreservationScorer` (P2-VER-04) — opt-in scorer for modes that declare `preserve.dates`. Extracts ISO dates, fiscal markers (Q3 2025, fiscal year 2026), month-year forms, and a small set of relative time markers, normalises whitespace and the leading determiner `the`, then rejects when any token from the original is missing from the candidate.
- `LengthDeltaScorer` (P2-VER-05) — bound-aware length check protecting against truncation and injection-style padding. Default band is 0.4x–2.0x of the original length; modes override via `VerifierProfile.length_min_ratio` / `length_max_ratio`.
- Spotlighting fence (P2-INJ-01) — per-request 16-byte nonce wrapping user input inside `<<<USER_TEXT_*>>>` / `<<<END_*>>>` sentinels. The orchestrator instructs the model to refuse instructions that appear inside the fence; nonce regeneration on collision is bounded and fails loudly rather than silently degrading.
- Ingress injection scanner (P2-INJ-02) — regex pack covering ignore-previous-instructions, role-flip, system-prompt-leak, fence-breakout, and exfiltration patterns. Returns HTTP 422 `input_injection_detected` (P2-INJ-03) before any prompt is rendered. SECURITY.md documents the scanner's limits explicitly (P2-INJ-04).
- `transduce_injection_detected_total{category}` Prometheus counter for fleet-wide observability of scanner hits.
- Allowlist mode loader (P2-PLG-01..P2-PLG-04) — the `modes` configuration section pins packages by sha256, refuses to load tampered packages, disables auto-discovery by default, and ships a manifest-only mode loader (TOML + Jinja) that never executes Python at registry load. A reference manifest mode `formal-to-warm` ships under `tests/fixtures/manifest_modes/`.
- Subprocess sandbox for Python plugin scorers (P2-PLG-05) — `multiprocessing.Process` worker with exact-name and fnmatch-glob environment filtering (`*_TOKEN`, `*_SECRET`); workers run under a budget and exceeding it terminates the child.
- sigstore signature surface (P2-PLG-06) — `signed_by` identity surfaced via `/v1/modes`; enforcement remains opt-in in v0.5 per ADR-0004 and flips default-on in v2.
- Targeted-feedback retry refinement (P2-VER-08) — the retry-prompt feedback names the failing scorer, the offending span, and a scorer-specific guidance sentence (CRITIC-style external feedback per Gou et al., ICLR 2024).
- ADR-0002 (manifest format: TOML), ADR-0003 (NLI model: MiniCheck-Flan-T5-Large with DeBERTa-v3 fallback), ADR-0004 (sigstore signing, optional in v0.5 and default-on in v2).
- `tests/eval/transduce_faithfulness_v0_1.jsonl` (200+ records across negation, antonym, tense, number, entity, fact-drift) and `tests/eval/injection_attacks_v0_1.jsonl` (75 attacks + 25 benign paraphrase controls). Structural shape is enforced under the unit suite; full per-mode AUROC and detection-rate gates run under `@pytest.mark.eval` with the v1.5 harness.

### Changed

- `VerifierProfile` widens with `nli_min`, `hhem_min`, `reject_on_negation_diff`, `preserve_dates`, `length_min_ratio`, and `length_max_ratio` (P2-VER-07). Defaults match docs/system-design.md §Verification Subsystem.
- `ScoreResult` gains an opaque `details: dict[str, Any]` field so multi-value scorers (bidirectional NLI publishing forward and backward; negation diff publishing added/removed cue lists) surface their structured outputs without breaking the single-float `Scorer` protocol.
- `PreserveRule` adds `DATES`.
- `ErrorCode` adds `input_injection_detected` and `mode_hash_mismatch`.
- The orchestrator's prompt template now receives `fence_open` and `fence_close` Jinja variables in addition to the existing `input`, `intensity`, `preserve` set; existing seed prompts continue to work without changes because the fence is applied to `input` automatically.

### BREAKING CHANGE

- `VerificationScores.verdict` literal field is removed (P2-MIG-02). The HTTP status code (200 vs 422) now carries the accept/reject signal at the response level. Clients that read `response["scores"]["verdict"]` must branch on status or on `rejection_reason is None`. See `docs/migration-v0-to-v0.5.md`.

### Project metadata

- Bumped version from `0.0.1` to `0.5.0`.
- Added migration guide at `docs/migration-v0-to-v0.5.md`.
- `transduce.example.yaml` now reflects the `modes` allowlist schema, the v0.5 verification defaults (`default_nli_min`, `default_hhem_min`, `reject_on_negation_diff`, `injection_scanner`), and points at the documented sha256 helper.

### Deferred

- Production wiring of `BidirectionalNLIScorer` and `HHEMScorer` in `transduce serve` ships once operators install the optional `transformers`, `torch`, and `minicheck` dependencies. The unit and structural tests run without them; the integration suite with real model weights runs under `@pytest.mark.slow` and is excluded from the default CI run.
- Full `transduce-faithfulness` AUROC and injection-detection-rate gates run nightly under `@pytest.mark.eval`; the structural-shape tests in this release verify the corpus loads and meets the dev-plan minimum-record contract.

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

[Unreleased]: https://github.com/Mathews-Tom/transduce/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Mathews-Tom/transduce/releases/tag/v0.5.0
[0.0.1]: https://github.com/Mathews-Tom/transduce/releases/tag/v0.0.1
