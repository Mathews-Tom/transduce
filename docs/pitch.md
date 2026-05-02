# transduce — testing the waters

> **Change the form. Conserve the signal.**
>
> A backend service for verifiable, mode-driven text transformations.
> Local-first via Ollama. Plugin-extensible. Open source.

---

Sharing this with a few of you to gut-check before I commit time. Read posture: skeptical. I'd rather hear "this is a wrapper around a prompt, don't" now than after a month of building.

## The observation

Sinceerly went viral last month — Chrome extension that "humanizes" AI-written emails. Single mode, paid SaaS, ~200 LOC of plumbing around one Anthropic call. The tech is unimpressive. The market response was not.

That's a signal worth pulling on, but not in the direction the meme points. The interesting layer isn't the humanizer, it's the fact that **every product now ships its own bespoke text transformation pipeline** — Notion's "make shorter," Slack's "rewrite tone," Gmail's smart suggestions, Linear's PR rewriter, every CMS, every CRM, every writing tool. None share infrastructure. None expose their prompts. None verify outputs. None let users plug in local models.

This is the market condition that produces commodity infrastructure: capability duplicated everywhere with poor differentiation. The abstraction wins.

## What I want to build

A backend service that does one thing: transform text through a named mode, with verification.

- **One verb: transform.** No UI. No CRUD. No history. Stateless API.
- **Modes are plugins.** `dejargon`, `register.casual`, `voice-match`, `length.normalize`, `tone.us-to-uk`, `simplify.grade-8`, `formal-to-warm`. Discovered via an explicit allowlist — never auto-loaded from `pip install`. Catalog grows without forking core.
- **Local-first as a privacy choice, not the assumed cost win.** Ollama is the reference backend for embedded/single-user deployments. vLLM is the reference for multi-tenant. Cloud APIs (Anthropic, OpenAI-compat, LiteLLM router) are first-class adapters — at low volumes, Haiku 4.5 is cheaper than a local A10G.
- **Verified — by an ensemble, not by cosine.** Every transformation runs through a scorer pipeline: bidirectional NLI (AlignScore / MiniCheck-FT5), HHEM cross-encoder, negation-token diff, embedding cosine as a coarse filter, plus mode-specific scorers and entity/number/URL preservation. Retry tightens the prompt with the *specific scorer that failed*, not a generic "try harder."
- **Apache 2.0** under `determ-ai/transduce`.

The mental model is **Pandoc for prose transformations**. Unix-philosophy primitive. Becomes a default by being honest about scope.

## Why this is worth doing instead of just shipping a clone

Three reasons.

**One — the addressable surface is 50–100× bigger than the meme product.** Sinceerly's Chrome extension addresses email composition. The same primitive serves CMS plugins, accessibility tools, IDE assistants, localization pipelines, customer support tools, social media drafters. Same backend, same verification ensemble, same diff infrastructure.

**Two — local-first is the right default for sensitive text, not the right default for cost.** People who care about transformations on email drafts, internal docs, customer correspondence are exactly the ones uncomfortable shipping content to OpenAI or Anthropic. Ollama and vLLM made local 14B–32B inference viable on commodity hardware. The application layer that uses them well is missing. But at <50K req/day, cloud Haiku is materially cheaper than a dedicated GPU. The honest framing is "local-first when privacy or volume justifies it"; the dishonest framing is "local-first beats cloud on cost." We're picking honest.

**Three — verification is the genuinely defensible technical contribution, *if* it's actually verification.** Cosine similarity on bge-small misses negation flips, antonym swaps, tense shifts, and number perturbations — the failure classes that matter most. The literature has moved to NLI-grounded scoring (AlignScore, MiniCheck, HHEM, FENICE atomic-claim decomposition). A primitive built around an ensemble verifier with mode-specific scorers, a composite verifier across compose chains, and retry-with-targeted-feedback is a meaningfully better artifact, and the methodology — once published — is sticky.

## Why this and not the alternatives

| I could instead… | But | So |
|---|---|---|
| Ship an OSS clone of Sinceerly | Single mode, narrow market, ethical drag | No |
| Build a humanizer SaaS | Sinceerly already has the mindshare; closed-source losing race | No |
| Wrap it in a generic LangChain abstraction | Too broad, no opinion, no verification, no local-first stance, no plugin security | No |
| Build a fine-tuned humanizer model | Costly, ages poorly, single use case | No |
| Build the transformation primitive | Category creation, low ethical drag, plays into local-first wave, slots into existing portfolio (`attest`, `anneal`, `armory`) | **Yes** |

## Honest weaknesses

I'd rather you see these from me than discover them yourselves.

- **No technical moat.** Two weeks of competent work clones the API surface. Defensibility is the verification *methodology* (an ensemble that catches negation flips and silent fact drift) plus the eval benchmark (`transduce-faithfulness`) becoming standards. That's a 12–18 month positioning play, not a feature race.
- **Plugin security is non-negotiable from v1.** Python entry-point auto-discovery is arbitrary code execution at process start — exactly the supply-chain shape that produced PyPI's March 2024 mass-malware halt. Modes must be explicit, allow-listed, sha256-pinned. Custom Python scorers run in a subprocess sandbox with secrets stripped. Manifest-only modes (TOML + Jinja) are the primary contribution path; Python plugins are the reserved escape hatch.
- **Prompt injection is a real threat model and transduce is *not* a safety boundary against hostile input authors.** A user pasting in attacker-crafted text can subvert the mode prompt unless the input is fenced (Spotlighting / structured queries) and an injection scanner runs at ingress. We do that, and we say so loudly.
- **Plugin ecosystem cold start is real.** I have to ship 6–8 polished modes in core or the plugin story doesn't bootstrap. Falsifiable test: at month 6, count unique non-author contributors with a mode used by ≥10 deployments. If that number is <3, the catalog moat thesis has falsified itself, and the project pivots to "small curated catalog + verification methodology" without pretending otherwise.
- **Detection arms race compresses humanizer modes specifically.** Humanize is *not* in the seed modes for v1; it ships as a third-party plugin. The brand is register adjustment for human readers, not detection evasion.
- **Local model quality ceiling under 14B.** The verification ensemble catches failures harder than cosine alone, which means small models will be rejected more often. `min_model_b` is enforced as a 412 precondition per request, not as documentation. Per-mode minimum-model recommendations are tested, not asserted.
- **Streaming-vs-verification is an architectural tension, not a v1.5 polish item.** Token streaming + post-hoc verification cannot both be true. v1 ships an advisory verification mode (stream now, verify after, return verdict as metadata) so interactive clients aren't blocked for 2–8s on every request.

## Scope and non-scope

| In scope | Out of scope |
|---|---|
| HTTP API (Litestar) with optional MCP server façade | Web UI / dashboard |
| Allowlisted mode plugins (manifest-first, signed) | Auto-discovery from PyPI |
| Backends: Ollama, vLLM, llama.cpp, Anthropic, OpenAI-compat, LiteLLM router | Native model fine-tuning |
| Verification ensemble: NLI + HHEM + negation diff + cosine + entity/number/URL preservation | Detection-evasion guarantees |
| Composite verifier across compose chains | Document storage / history |
| Word-level diff (semantic-cleanup post-processed) | Browser extension as part of core |
| Embeddable as Python library | Multi-tenant auth in core (v2) |
| OTel GenAI SemConv (`gen_ai.*`) + Prometheus | PII / raw-text in span attributes (banned) |
| Cost budget per request; concurrency semaphore per backend | Unbounded retry loops |
| Streaming with advisory verification in v1 | Streaming with strict verification (architecturally incompatible) |
| Reference Chrome ext + CLI as separate repos | `humanize.*` modes in core |

## Time budget

Halved from the original draft after pressure-testing. The original v0.5 bundled retry loop + eval harness + two backends + mode versioning into 14 days, which was 2–3× optimistic.

| Phase | Scope | Effort |
|---|---|---|
| v0 | API skeleton, Ollama backend, 3 seed modes (`dejargon`, `register.casual`, `length.normalize`), cosine + entity/number scorers, word diff | Weekend |
| v0.5 | Verification ensemble (add NLI scorer + negation-diff + HHEM); retry-with-targeted-feedback; Spotlighting fence; injection scanner at ingress; allowlisted plugin loader | 2–3 weeks |
| v1 | OpenAPI spec, Docker Compose, contribution docs, Anthropic + vLLM backends, mode versioning with multi-version dispatch, composite verifier, streaming-with-advisory-verify, cost budget, language detection, 8 seed modes (no humanize) | 5–6 weeks |
| v1.5 | Batch endpoints, MCP server façade, public benchmark suite (`transduce-faithfulness`), eval harness wired to `attest`, attention-probe scorer for local backends | 8 weeks |

I can have v0 in a public repo this weekend if I commit Saturday morning.

## What I need from you

Specific questions, in order of how much they'd change my decision:

1. **Is the reframe (transformation primitive vs humanizer) actually defensible in the market, or am I rationalizing scope expansion to feel better about cloning Sinceerly?** Push hard on this. The current plan ships *no* humanize mode in core, so the headline use case is gone — does the rest still pull?
2. **Manifest-only modes (TOML + Jinja) vs Python entry-points-with-sandbox** — manifest is safer and lowers the barrier for non-Python contributors but caps mode expressiveness. Worth the trade-off?
3. **Is the verification ensemble (NLI + HHEM + negation diff + cosine + preservation regexes) the right substance, or is it too much for a v0.5?** The minimum-credible verifier is "cosine + NLI"; everything else is incremental. Pick the floor.
4. **Do you see any production use case in your stack where this would slot in?** Concrete examples are worth more than hypothetical ones. Even "no, but X tool would benefit" helps.
5. **What's the most likely failure mode in 6 months — abandoned (no users), captured (someone Apache-2.0s a competitor), commoditized (HuggingFace ships a pipeline), or compromised (plugin supply-chain incident)?**
6. **Naming** — `transduce` lands well for the FP/engineering audience; possibly cerebral for broader market. Strong opinions?
7. **Anything I'm not seeing.**

Reply in-thread or DM. Not looking for validation. Looking for the thing I haven't thought of.

— Tom
