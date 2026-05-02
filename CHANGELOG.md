# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
