# Security policy

## Threat model

transduce is a backend service that accepts arbitrary text from clients and forwards rendered prompts to model backends. Operators are responsible for:

- Network exposure (default deployment is local-only; bearer-token middleware is opt-in).
- Backend credential management (API keys read from environment; never hardcoded).
- Mode allowlist curation (sha256-pinned packages; no auto-discovery).
- Multi-tenant isolation (deferred to v2; single-tenant assumed in v0–v1.5).

transduce is **not a safety boundary against adversarial input authors.** The Spotlighting fence and ingress scanner are defense-in-depth layers. Operators serving untrusted input must add their own controls.

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
