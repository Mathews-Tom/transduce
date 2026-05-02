# Migration guide — v0.0.1 → v0.5.0

This release introduces the v0.5 ensemble verifier, the Spotlighting injection fence and ingress scanner, and the sha256-pinned mode allowlist. It contains breaking API and configuration changes that operators must address during the upgrade. Pre-1.0 software permits breaking changes with notice; this is the notice.

## Breaking changes

### `VerificationScores.verdict` removed (P2-MIG-02)

The `verdict` literal field has been removed from the `VerificationScores` schema. The HTTP status code now carries the accept/reject signal:

- HTTP 200 (or 201 for `POST`) — verifier accepted; the response body's `scores` reflects the accepted run.
- HTTP 422 with `error: verification_failed` — verifier rejected after exhausting retries; the error envelope's `scores.rejection_reason` names the failing scorer.

Clients that read `response["scores"]["verdict"]` should instead branch on the HTTP status. Clients that rendered "accept" / "reject" UI from the field should switch to either status code or `rejection_reason is None`.

### New error codes

Two new `ErrorCode` values land in v0.5:

- `input_injection_detected` (HTTP 422) — emitted by the ingress scanner when a request payload matches a documented injection pattern. The error envelope's `details` carries `category`, `matched_pattern`, and `span`.
- `mode_hash_mismatch` (HTTP 503 surfaced via `/readyz`) — emitted at startup when an allowlisted package's computed sha256 differs from its `transduce.yaml` pin.

Clients that switch on error codes must add handlers for these.

### `VerificationScores` schema widened

Successful responses now include the full ensemble outcome:

```json
{
  "cosine": 0.91,
  "nli_forward": 0.88,
  "nli_backward": 0.85,
  "hhem": 0.74,
  "negation_diff": {"added": [], "removed": []},
  "preserved": {"entities": true, "numbers": true, "urls": true, "dates": true},
  "mode_specific": {},
  "topical_similarity": 0.91,
  "rejection_reason": null
}
```

`nli_forward`, `nli_backward`, and `hhem` may be `null` when the ensemble short-circuits before those scorers run. `negation_diff` is always present (defaults to empty added/removed lists when the scorer is not configured). Clients that strictly schema-validated the v0 response shape need to widen their schema.

## Configuration migration

### Add the `modes` allowlist

`transduce.yaml` now requires (or strongly encourages) a `modes` section. The default is empty — service starts but resolves no modes — so operators must list packages explicitly:

```yaml
modes:
  source: allowlist
  enforce_signing: false              # v2 flips this default; ADR-0004
  packages:
    - name: transduce-mode-formal-to-warm
      version: "1.0.0"
      sha256: "<64-hex-character pin>"
      path: ./packages/transduce-mode-formal-to-warm
      signed_by: "release@determ-ai"  # optional, surfaced via /v1/modes
```

Compute the sha256 of a package directory (or single file) with:

```bash
uv run python -c \
  "from transduce.registry.allowlist import compute_package_sha256; \
   from pathlib import Path; \
   print(compute_package_sha256(Path('packages/transduce-mode-formal-to-warm')))"
```

`source: auto` opts back into entry-point discovery and emits a startup warning; production deployments must keep `source: allowlist`.

### Add new `verification` defaults

```yaml
verification:
  enabled: true
  default_cosine_min: 0.85
  default_nli_min: 0.70
  default_hhem_min: 0.50
  reject_on_negation_diff: true
  injection_scanner: regex-pack-v0.5
  max_retries: 3
```

Per-mode `VerifierProfile` declarations override these defaults; the verifier ensemble runs cosine → negation diff → bidirectional NLI → HHEM → preservation rules → mode-specific scorers in that order, short-circuiting on the first failure.

## Behavioural changes

- **Spotlighting fence on every request.** The orchestrator wraps user input inside a per-request 16-byte nonce sentinel before rendering the prompt. Mode authors do not need to update existing prompts; the fence is applied transparently. Authors who want explicit control reference `{{ fence_open }}` and `{{ fence_close }}` in their Jinja template.
- **Ingress injection scanner.** `POST /v1/transform` rejects requests that match the documented injection patterns (`ignore previous instructions`, role-flip phrases, system-prompt-leak markers, fence-breakout sentinels, exfiltration verbs) with HTTP 422 `input_injection_detected`. SECURITY.md documents what the scanner does and does not do.
- **Targeted-feedback retry.** When the ensemble rejects a candidate, the next prompt now carries a scorer-specific guidance sentence ("do not add or remove negation cues"; "preserve every named entity from the source verbatim"). Retry-2 acceptance rates should improve over retry-1 on non-pathological inputs.

## Optional ensemble dependencies

The bidirectional NLI scorer (MiniCheck-Flan-T5-Large) and HHEM cross-encoder ship in core but their model weights are not bundled. Operators who want the full v0.5 ensemble install:

```bash
uv add transformers torch
uv add minicheck   # source: pip install git+https://github.com/Liyan06/MiniCheck.git
```

Then construct the production scorer set in `transduce serve` by composing `BidirectionalNLIScorer(entail=build_minicheck_entailer())` and `HHEMScorer(scorer=build_hhem_scorer())` alongside the existing cosine and preservation scorers.

Operators who do not install the optional dependencies can keep running the v0 cosine + preservation scorer set; the verifier will report `nli_forward: null`, `nli_backward: null`, `hhem: null` in the response.

## Reference checklist

- [ ] Update API clients to branch on HTTP status, not `scores.verdict`.
- [ ] Add `mode_hash_mismatch` and `input_injection_detected` to the client's `ErrorCode` switch.
- [ ] Widen the response schema in any client-side validators.
- [ ] Add the `modes` section to `transduce.yaml` with explicit packages.
- [ ] (Optional) Install `transformers`, `torch`, and `minicheck` for the full ensemble.
- [ ] Re-run the integration suite against a staging deployment before promoting to production.
