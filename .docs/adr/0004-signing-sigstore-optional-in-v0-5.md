# ADR-0004 — Mode signing uses sigstore, optional in v0.5, default-on in v2

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-05-02 |
| Deciders | @Mathews-Tom |
| Tags | security, supply-chain, signing |

## Context

The v0.5 mode-allowlist loader pins packages by sha256 (P2-PLG-02) and forbids auto-discovery of `transduce.modes` entry points (P2-PLG-03). sha256 pinning prevents tampered wheels from loading, but it does not establish *who* produced the wheel. The dev-plan target for v2 is signed-mode-only enforcement (P5-SIG-01); v0.5 is the staging step where the surface lands but does not block.

Q-03 in dev-plan Appendix C asks: sigstore vs in-house signing for v0.5? The choice constrains the signing surface that ships in v0.5 (`/v1/modes` exposing `signed_by`), the verification path that lights up under `modes.enforce_signing: true`, and the migration story for v2 when the flag flips default-on.

Two signing approaches considered:

1. **sigstore-python** — keyless OIDC-based signing built on the public Rekor transparency log and Fulcio short-lived certificate authority. Verifiable by anyone with no key distribution, no key rotation, no HSM. Ecosystem support across PyPI, GitHub releases, container registries.
2. **In-house GPG / age signing with a project-managed public key** — traditional approach; requires distributing the project's public key out-of-band, rotating on compromise, and providing a verification CLI.

## Decision

Mode signing uses sigstore (`sigstore-python`) and the verification path lands in v0.5 as **optional**: the loader records `signed_by` in `/v1/modes` responses and stores signature status in the registry, but `modes.enforce_signing: false` is the default. v2 (P5-SIG-01) flips the default to `true`; unsigned packages refuse to load unless an operator sets `unsigned_modes: allow` with a startup warning.

In-house GPG/age signing is rejected for both v0.5 and v2.

## Alternatives considered

- **In-house GPG/age signing** — rejected because it forces transduce to operate a public-key distribution channel, manage rotation on compromise, and ship a verification CLI. Each is a non-trivial maintenance burden for a project whose moat is verification methodology, not key infrastructure. Operators would still have to trust *transduce's* key-distribution pipeline; the trust boundary moves but does not shrink.
- **No signing surface in v0.5** — rejected because the dev plan ships `/v1/modes` `signed_by` field and Phase 2's eval suite assumes signature status is queryable. Deferring the surface means v2 has to add a breaking schema change to a client-facing endpoint.
- **sigstore default-on in v0.5** — rejected because sigstore-python's verification API is still maturing (the `verify` function shape changed twice between 2024 and early 2026 per the upstream changelog). Default-on enforcement during a maturing API surface risks breaking honest deployments. Defer enforcement to v2 when the API has stabilized; in the interim surface signature status without blocking.
- **TUF (The Update Framework)** — rejected as overkill. TUF's threshold-signing and target-rotation primitives are valuable for operating-system distros and large software delivery networks, not for a per-package mode allowlist where sha256 pins already provide tamper-evidence.

## Consequences

### Positive

- Zero key distribution: operators verify a package against the public Rekor log and Fulcio root, both run by the OpenSSF.
- Signature status is observable in `/v1/modes` from v0.5 onward, so dashboards and review tooling can flag unsigned packages without waiting for v2.
- v2 enforcement is a config-flag flip, not a code-level migration: `modes.enforce_signing: true` is the only delta, and the rejection-path code already exists from v0.5.
- Aligns with PyPI's own sigstore adoption (PEP 740) so transduce mode authors can reuse their existing CI signing setup.

### Negative

- sigstore-python's verification API is not yet 1.0; v0.5 pins to a specific release range (`sigstore>=3,<4` documented in `pyproject.toml`) and an upstream API break would require a follow-up commit. The risk register row R-02 documents this; the mitigation is the optional-in-v0.5 phasing.
- Network access to Rekor and Fulcio is required during signature verification at registry-load. Air-gapped deployments cannot use online verification; the v2 cutover ships an offline-bundle mode (Rekor inclusion proof + Fulcio cert chain bundled with the package) per the v2 deliverable list.
- Signature absence is **not** a CI-blocking gate in v0.5; operators who want to enforce gating must opt in explicitly. This is the documented trade-off for shipping the surface alongside a maturing verification library.

### Neutral

- The `signed_by` field schema (`<identity>@<provider>`, e.g., `release@determ-ai`) follows sigstore's identity convention and matches the example in `transduce.example.yaml`.

## Compliance and verification

- `tests/unit/registry/test_signing.py` covers signature-status surface, signed/unsigned distinction, and the optional vs enforced behavior under both `enforce_signing` flag values.
- `tests/integration/test_signing_real.py` (gated `@pytest.mark.integration and slow`) verifies a real sigstore bundle against a fixture mode package.
- The risk register entry R-02 ("sigstore tooling immature for Python") tracks the upstream API stability; if sigstore-python ships a breaking change before v2, a follow-up ADR documents the migration.
- v2's enforcement flip (P5-SIG-01) is gated by a CI assertion: `transduce.example.yaml` must show `enforce_signing: true` at v2 and the gate file under `tests/security/test_signing_enforcement.py` blocks the v2 release if it does not.

## References

- `.docs/development-plan.md` Appendix C Q-03 — Signing approach
- `.docs/development-plan.md` §Risk register R-02 — sigstore tooling immature
- `docs/system-design.md` §Mode Registry — Allowlist + sigstore identity
- sigstore-python documentation — <https://docs.sigstore.dev/python/>
- PEP 740 — Index support for digital attestations — <https://peps.python.org/pep-0740/>
