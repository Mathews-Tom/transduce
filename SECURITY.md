# Security policy

## Threat model

transduce is a backend service that accepts arbitrary text from clients and forwards rendered prompts to model backends. Operators are responsible for:

- Network exposure (default deployment is local-only; bearer-token middleware is opt-in).
- Backend credential management (API keys read from environment; never hardcoded).
- Mode allowlist curation (sha256-pinned packages; no auto-discovery).
- Multi-tenant isolation (deferred to v2; single-tenant assumed in v0–v1.5).

transduce is **not a safety boundary against adversarial input authors.** The Spotlighting fence and ingress scanner are defense-in-depth layers. Operators serving untrusted input must add their own controls.

### What Spotlighting and the ingress scanner do, and what they do not

| Layer | Does | Does not |
|---|---|---|
| Spotlighting fence (per-request 16-byte nonce wrapping user input inside `<<<USER_TEXT_*>>>` / `<<<END_*>>>` sentinels) | Tells the model which characters are user-supplied and instructs the prompt template to refuse instructions inside the fence. Reduces success rates against well-described attack categories. | Prevent a sufficiently-trained adversary from crafting input that the model still treats as instructions. Provide cryptographic isolation. |
| Ingress regex scanner | Catches well-known patterns (role-flip, "ignore previous instructions," system-prompt-leak phrasings, fence-breakout markers, exfiltration verbs). Returns `INPUT_INJECTION_DETECTED` (HTTP 422) before the prompt is rendered. | Catch novel paraphrases that do not match the documented patterns. Replace a model-side or downstream injection-aware judge. |

Hostile input authors that need a guarantee should not deploy transduce as the only line of defense. Operators serving untrusted input should layer at least one of: (a) an authoritative model judge invoked after generation, (b) a sandboxed downstream surface that cannot act on the model's output, or (c) a stricter pattern set tuned to the deployment's threat model.

## Supported versions

| Version | Status |
|---|---|
| `0.x` | Pre-release; security fixes applied to the latest minor only |
| `1.x` | Supported once released; latest minor receives fixes |
| `< 0.x` | Unsupported |

## Reporting a vulnerability

Do **not** open a public GitHub issue.

Email the maintainer at `tommathews2007@gmail.com` with:

- A description of the issue and its impact.
- Steps to reproduce, including config snippet and request payload if relevant.
- Affected version(s) and deployment topology.
- Your name or handle for credit (or request anonymity).

Acknowledgement target: 72 hours. Initial assessment: 7 days. Fix timeline depends on severity per the table below.

## Severity and response

| Severity | Definition | Fix target |
|---|---|---|
| Critical | Remote code execution, credential disclosure, allowlist bypass | 72 hours |
| High | Injection-fence bypass, mode-isolation failure, unauthenticated cost amplification | 7 days |
| Medium | Denial of service, log injection, observability data leak | 30 days |
| Low | Information disclosure with no privilege impact | next minor release |

## Disclosure

After a fix is released, the maintainers publish a security advisory on the GitHub Security Advisories tab with CVE assignment for issues at High severity or above. Reporters are credited unless anonymity was requested.

## Out of scope

- Findings against modes or scorers loaded from third-party allowlists — report to the package author.
- Findings against backend providers (Ollama, vLLM, Anthropic, etc.) — report to the upstream project.
- Findings that require operator misconfiguration disabling default protections (e.g., `unsigned_modes: allow`, `debug.include_text: true` in production).
