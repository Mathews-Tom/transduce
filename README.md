# transduce

Change the form. Conserve the signal.

A backend service for mode-driven, ensemble-verified text transformations. Local-first via Ollama. Plugin-extensible. Apache 2.0.

> **Status:** v0.0.1 preview. The cosine + preservation verifier is the floor; the full ensemble lands in v0.5.

## Quick start

```bash
# 1. Install dependencies via uv
uv sync

# 2. Pull a model into Ollama
ollama pull gemma4

# 3. Install the spaCy English NER model
uv run python -m spacy download en_core_web_sm

# 4. Start the service
uv run transduce serve --config transduce.example.yaml
```

```bash
# Issue a transform request
curl -s http://localhost:8080/v1/transform \
  -H 'content-type: application/json' \
  -d '{"text": "We synergize verticals.", "mode": "dejargon", "intensity": 0.6}' \
  | jq
```

## v0 capabilities

- `POST /v1/transform` with three seed modes: `dejargon`, `register.casual`, `length.normalize`.
- Verifier ensemble (cosine + entity + number + URL preservation) with first-fail short-circuit and targeted-feedback retry.
- Word-level diff returned as structured ops; clients render the change-set.
- `GET /v1/modes`, `/v1/modes/{id}`, `/v1/backends`, `/v1/scorers`, `/healthz`, `/readyz`, `/metrics`.
- Ollama backend adapter; Phase 3 adds Anthropic, vLLM, llama.cpp, OpenAI-compat, and LiteLLM.

## What's next

See `.docs/development-plan.md` for the phase-by-phase roadmap. Phase 2 (v0.5) replaces cosine-only verification with the full ensemble (NLI + HHEM + negation diff), adds the allowlist plugin loader, and ships the Spotlighting injection fence.

## Documentation

- `docs/overview.md` — the framing case.
- `docs/system-design.md` — components, request lifecycle, configuration.
- `docs/pitch.md` — the original pressure-test.
- `.docs/development-plan.md` — phased deliverable matrix and exit criteria.
- `.docs/adr/` — recorded architectural decisions.
