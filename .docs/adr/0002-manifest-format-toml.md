# ADR-0002 — Mode manifest format is TOML

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-05-02 |
| Deciders | @Mathews-Tom |
| Tags | registry, plugins, manifest |

## Context

The v0.5 release introduces manifest-only modes as the primary contribution path: a mode is a small declarative file plus a Jinja prompt template inside an allow-listed package, and the registry never executes Python at load time (`docs/system-design.md` §Mode Registry, dev-plan deliverable P2-PLG-04). The manifest carries the mode id, version, description, intensity range, preservation defaults, backend requirements, verifier profile, and supported languages.

The dev-plan Appendix C tags this Q-01 ("Manifest format: TOML vs YAML") for resolution before Phase 2 exit. Manifests are read at startup, written by mode authors, and reviewed during PR audit by operators who add packages to the allowlist with sha256 pins. Format choice affects: parser availability in stdlib, schema clarity for non-Python contributors, error-message quality on malformed input, and whether the same parser already runs elsewhere in the toolchain.

## Decision

Mode manifests use TOML (`mode.toml`), parsed with the standard-library `tomllib` module. YAML remains the format for service configuration (`transduce.yaml`) where multi-document support and anchor reuse pay for the extra dependency, but mode manifests are intentionally one-document, flat-section files where TOML's stricter grammar produces clearer errors and zero added dependencies.

Manifest schema lives in a Pydantic model (`ManifestSpec`) loaded from the TOML dict; validation runs at registry load and any failure refuses the package with `ERR_MODE_HASH_MISMATCH`-class error semantics (the package is rejected even if the sha256 is correct, because the manifest is malformed).

## Alternatives considered

- **YAML (`mode.yaml`)** — rejected because the manifest has no document-stream, no anchors, no merge-keys, and no need for the multi-line scalar gymnastics that justify YAML elsewhere. Adding `pyyaml` to the registry path widens the attack surface for a parser CVE class that has hit the Python ecosystem repeatedly. Operators already bear `pyyaml` in `transduce.yaml`, but reusing it here couples the mode loader to a config format whose Phase-3 evolution may diverge.
- **JSON (`mode.json`)** — rejected because mode authors review and edit manifests by hand; JSON's lack of comments and trailing-comma intolerance makes hand-editing brittle. Stdlib parser, but worse author UX.
- **Python `pyproject.toml` `[tool.transduce.mode]` section** — rejected because it conflates packaging metadata with mode declaration. A mode package can ship multiple manifests under one `pyproject.toml`; the format also forces the mode loader to parse build metadata, expanding the registry's reach into packaging concerns.
- **Inline manifest in `pyproject.toml` with a separate `prompt.j2`** — same objection as above plus an awkward two-file convention with no parent directory grouping the pair.

## Consequences

### Positive

- Zero added dependencies in the registry path: `tomllib` ships with Python 3.11+ and the project requires 3.12+.
- TOML's grammar surfaces malformed manifests at parse time with line-and-column precision, so operators see the bad field before sha256 verification or schema validation runs.
- Manifest format is independent of `transduce.yaml`, so Phase 3+ config changes do not cascade into mode-package compatibility.
- The format choice mirrors `pyproject.toml`'s established convention in the Python packaging ecosystem, lowering author surprise.

### Negative

- TOML's spec disallows true `null` literals; absent values must use omission or sentinel strings. Pydantic handles this cleanly via `Optional[T] = None`, but contributors coming from YAML may need a one-line convention note in the mode-author guide.
- Multi-line strings in TOML use triple-quoted heredocs; long descriptions or `signed_by` URIs read slightly less naturally than YAML block scalars. Acceptable trade-off given the manifest field set is small.

### Neutral

- The Jinja prompt template ships beside the manifest as `prompt.j2` (or `prompts/<id>.j2` for multi-prompt packages). The choice of TOML for the manifest does not constrain prompt-template format.

## Compliance and verification

- `tests/unit/registry/test_manifest.py::test_manifest_only_mode_loads_without_python` exercises the happy path against a fixture manifest under `tests/fixtures/manifest_modes/`.
- `tests/unit/registry/test_manifest.py::test_manifest_invalid_jinja_raises_at_load_time` confirms Jinja syntax errors fail at registry-load, not at first prompt render.
- The mode-author contributor guide (Phase 3 deliverable) will reference this ADR and the fixture as the canonical example.
- A repository grep for `yaml.safe_load.*mode` in `src/transduce/registry/` is a CI assertion that the registry path remains YAML-free.

## References

- `.docs/development-plan.md` Appendix C Q-01 — Manifest format
- `docs/system-design.md` §Mode Registry — Manifest-only modes
- Python `tomllib` documentation — <https://docs.python.org/3/library/tomllib.html>
- TOML 1.0.0 specification — <https://toml.io/en/v1.0.0>
