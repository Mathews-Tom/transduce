# transduce — System Overview

> **Tagline:** Change the form. Conserve the signal.
> **Repo (proposed):** `github.com/determ-ai/transduce`
> **License:** Apache 2.0
> **Status:** Pre-implementation, design phase

---

## What

`transduce` is a backend service that performs **mode-driven, ensemble-verified text transformations** with pluggable model backends. It exposes a small HTTP API (with an optional MCP server façade) that any client — browser extensions, CMS plugins, CRMs, CLI tools, IDE assistants, internal apps — can call to transform text through a named mode.

Examples of seed modes: `dejargon`, `register.casual`, `voice-match`, `length.normalize`, `style.match`, `tone.us-to-uk`, `simplify.grade-8`, `formal-to-warm`. Modes are declarative specifications loaded via an explicit allowlist — never auto-discovered from `pip install`. The catalog is unbounded; modes get added by editing the allowlist, not by forking core.

The defining design choices:

- **Backend-only.** No UI, no integrations, no opinions about where the text comes from or where it goes. One verb: transform.
- **Local-first as a privacy choice.** Ollama is the reference backend for embedded / single-user deployments. vLLM is the reference for multi-tenant. Cloud APIs (Anthropic, OpenAI-compat, LiteLLM router) are first-class adapters — local-first is honest about being a privacy/data-residency choice, not an automatic cost win.
- **Ensemble verification.** Every transformation runs through a scorer pipeline: bidirectional NLI (AlignScore or MiniCheck-FT5), HHEM cross-encoder, negation-token diff, embedding cosine as a coarse filter, plus mode-specific scorers and entity/number/URL/date preservation. The retry loop tightens the prompt with the *specific scorer that failed*, not generic feedback.
- **Plugin-allowlisted.** Modes are loaded only from a configured allowlist with sha256 pinning. Manifest-only modes (TOML + Jinja) are the primary contribution path; Python plugins are the reserved escape hatch and run in a subprocess sandbox with secrets stripped.
- **Injection-aware at ingress.** User text is fenced inside a per-request nonce sentinel before rendering into the prompt template (Spotlighting). An injection scanner runs at ingress and the verifier checks output for instruction-drift artifacts. Transduce is *not* a safety boundary against hostile input authors and the docs say so loudly.
- **Open source.** Apache 2.0 from day one.

The conceptual analogy: **Pandoc for prose transformations**. Unix-philosophy primitive. No UI. Becomes a default by being honest about scope.

---

## Why

Three converging pressures define the moment.

### 1. Polish is no longer a signal

LLM writing is everywhere. The implicit social contract — that well-written text indicates effort, attention, or care — has broken. Recipients increasingly read polished prose as *less* trustworthy, not more. Sinceerly went viral on this exact observation, but its product addresses ~1% of the actual surface (one mode, one channel, one model, paid SaaS, no verification).

### 2. Every product is reinventing this layer badly

Notion's "make shorter," Slack's "rewrite tone," Gmail's smart suggestions, Linear's PR description rewriter, every AI writing assistant ships its own bespoke transformation pipeline. None share infrastructure. None expose their prompts. None verify outputs beyond "does it look reasonable." None let users plug in local models. This is exactly the market condition that produces commodity infrastructure: when a capability is duplicated everywhere with poor differentiation, the abstraction wins.

### 3. Local-first is the right default for sensitive text

Email drafts, PR reviews, internal docs, customer correspondence — exactly the cases where users want transformation but don't want to ship the content to OpenAI or Anthropic. Ollama, vLLM, and llama.cpp made local inference viable for 14B–32B class models on commodity hardware. The missing piece is *the application layer that uses them well*. Honest framing: at <50K req/day cloud Haiku is cheaper; local-first is justified by privacy, data residency, or volume — not by per-request cost at hobby scale.

`transduce` is the primitive that fits underneath all three.

---

## How

Architecturally simple. The work happens in seven stages:

1. **Receive** — Client posts text + mode + parameters
2. **Detect** — Language detection at ingress (fasttext-langid); reject 415 if the mode does not declare support
3. **Scan** — Injection scanner over input; reject 422 with `INPUT_INJECTION_DETECTED` on hit
4. **Resolve** — Mode registry yields prompt template, intensity policy, preservation hooks, verifier profile, supported language set
5. **Generate** — Selected backend produces transformed text. User text is fenced inside a per-request nonce sentinel inside the rendered prompt
6. **Verify** — Ensemble of scorers (NLI + HHEM + negation diff + cosine + preservation rules + mode-specific). Retry up to N times with *targeted* prompt tightening, where the failure context names the exact scorer that rejected. Composite verifier compares final output against original input across compose chains
7. **Diff and return** — Word-level diff (Myers + semantic cleanup) computed against original. Response includes original, modified, diff, per-scorer scores, mode metadata, language, retry count, cost

The complexity lives in the *quality* of each stage — the prompts, the scorer ensemble, the retry strategy, the backend adapters — not in the architecture. This is exactly the part that benefits from being open: prompts iterate via PR, modes get contributed by domain experts, scorers get benchmarked publicly against `transduce-faithfulness`.

Stack: Python 3.12+, Litestar, Pydantic v2, httpx, fastembed (cosine), Hugging Face transformers (NLI/HHEM), diff-match-patch-python (maintained fork). Single-binary deploy via Docker. Embeddable as a library for in-process use.

---

## Market Research

### Direct comparables

| Product | Positioning | Open? | Local? | API-first? | Verifier | Modes |
|---|---|---|---|---|---|---|
| Sinceerly | Email humanizer | No | No | No | None | 3 |
| QuillBot | Paraphraser SaaS | No | No | Limited | Unknown | ~5 |
| Wordtune | Tone rewriter | No | No | Limited | Unknown | ~6 |
| Grammarly | Grammar/style correction | No | No | Limited | Rule-based | N/A |
| Hemingway Editor | Readability rules | No | Yes | No | Rule-based | N/A |
| Humanize AI / GPTHuman / etc. | Detection evasion SaaS | No | No | Some | None | 1–3 |
| **transduce** | **Transformation primitive** | **Yes** | **Yes** | **Yes** | **Ensemble (NLI + HHEM + negation + cosine + rules)** | **Unbounded (allowlisted plugin)** |

Every existing product is a *destination* — a SaaS or app with its own UI. None positions as infrastructure. None is local-first. None ships an ensemble verifier.

### Adjacent infrastructure

| Project | Purpose | Why it isn't this |
|---|---|---|
| LiteLLM | Model API abstraction | Routes calls; doesn't transform. transduce uses it as a backend adapter |
| LangChain | Generic LLM framework | Too broad, deep coupling, no transformation primitive |
| Outlines / Instructor | Structured output | Constrains generation, not transformation |
| Guardrails / NeMo Guardrails / LLM Guard | Output validation, input/output rails | Validates, doesn't transform-with-validation. transduce can use LLM Guard for the injection scanner |
| Ragas / DeepEval / promptfoo | Eval & observability | Adjacent, complementary, not overlapping. transduce's `attest` integration is the analogue |
| Vectara HHEM | Hallucination evaluation model | Used as a scorer inside transduce's verifier ensemble |
| dbt adapter pattern | Database adapter abstraction | Architectural reference for transduce's backend adapter layer |

Closest spiritual analogues from other domains: **Pandoc** (document format conversion), **ImageMagick** (image transformations), **ffmpeg** (media transformations). All three succeeded as Unix-philosophy primitives that other tools built on. None has a UI. All became defaults.

### Demand signals

- **Sinceerly** went viral within weeks (BizTech Weekly, Mashable, Business Insider, Republic World, Fast Company) on a single mode of this primitive.
- **r/LocalLLaMA**, **r/ChatGPT** see recurring threads asking "how do I make this less AI-sounding," "anyone have prompts for tone shifting," "best local model for paraphrasing."
- **HuggingFace** has dozens of humanization datasets and fine-tuned models with no coordinating service layer.
- **Browser extension stores** show dozens of "AI humanizer" extensions, all paid, all opaque, all single-mode.

Demand is real. The supply side is a fragmented mess of closed SaaS and one-off scripts. That is the gap. The framing risk is brand contamination by anti-detection SEO — addressed by deliberately *not* shipping `humanize.*` in core.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Verification gives false confidence (cosine misses negation, antonyms, tense flips) | **Critical** | Ensemble verifier with NLI (AlignScore/MiniCheck) as primary signal; negation-token diff as deterministic floor; rename API field from `verdict: accept` to `topical_similarity` so clients confront the limit |
| Plugin supply-chain compromise via PyPI | **Critical** | Default-deny entry-point loading; allowlist with sha256 pins in config; sigstore-signed core modes; manifest-only modes as primary path; Python scorers run in subprocess sandbox with secrets stripped |
| Prompt injection via user input | **Critical** | Spotlighting fence with per-request nonce; ingress injection scanner (LLM Guard / prompt-armor); `INPUT_INJECTION_DETECTED` error code; documented as not-a-safety-boundary against hostile input authors |
| Detection arms race compresses humanizer modes | High | `humanize.*` not in core; ships as third-party plugin; reposition register/tone modes as register adjustment for human readers, not detection evasion |
| Composite drift across mode chains | High | `CompositionVerifier` runs end-to-end against original input; multiplicative intensity composition; preservation rules union across stages |
| Streaming vs verification tension | High | v1 ships advisory verification mode (stream now, verify after, return verdict as metadata) — strict verification stays non-streaming |
| Cost runaway via retry loop | High | Per-request `max_cost_per_request_usd` budget; non-improving-trend early exit; `transduce_generation_cost_usd_total` metric per backend/mode |
| Concurrent storm on single Ollama instance | High | Per-backend concurrency semaphore returning 429; vLLM documented as production multi-tenant backend |
| Privacy leak via OTel spans | Medium | Raw text and `last_candidate` banned from span attributes; sha256[:8] + length only; opt-in `debug.include_text` for non-prod |
| Mode versioning collision on pip upgrade | Medium | Multi-version dispatch via vendored import paths or explicit version routers; advisory `@version` until then |
| Multi-language silent failure | Medium | Language detection at ingress; `ModeSpec.supported_languages`; multilingual embedder (`bge-m3`) when mode declares non-English support |
| Local model quality ceiling (sub-14B fails on tight constraints) | Medium | `min_model_b` enforced as 412 precondition per request, not just documentation; per-mode minimum-model floors tested in `transduce-faithfulness` |
| Plugin ecosystem cold start | High | Ship 8 well-designed modes in core (no humanize); document the contribution path with manifest-only examples; falsifiable test at month 6: <3 unique non-author contributors with ≥10-deployment uptake → catalog moat thesis dropped |
| Ethical perception drag ("evade detection" association) | Medium | `humanize.*` removed from core seed modes; reframe explicitly as transformation primitive for register/tone/length/style; refuse to ship anti-detection-tuned modes in core |
| Brand confusion with Clojure transducers | Low | Different domain, different audience; FP context is mildly positive |
| Anthropic / OpenAI ship native equivalents | Medium | Stay focused on local-first, the catalog, and the verifier methodology; cloud is a backend, not a competitor |
| Commoditization by HuggingFace pipeline | Medium | Differentiate on verification ensemble + plugin security + operational ergonomics |
| Roadmap slip from over-scoped milestones | Medium | v0.5 halved from original draft (2–3 weeks instead of 2); eval harness moved to v1.5 as the spine, not a v0.5 sidecar |

---

## Possible Moat — Honest Assessment

There is **no architectural moat** here. Any competent team could clone the API surface in two weeks. The defensible elements, in order:

1. **Verification methodology as the standard.** An ensemble verifier (NLI + HHEM + negation diff + preservation rules) that catches negation flips and silent fact drift is materially better than the cosine-only competition. If `transduce`'s scorer protocol becomes how text-transformation quality is measured, every adjacent tool either adopts or competes against it. This is the primary moat.
2. **`transduce-faithfulness` benchmark ownership.** A public benchmark for transformation tasks (preservation × style strength × negation robustness × injection resistance) anchors mindshare and gives third parties something to publish against.
3. **Plugin security model as a positive differentiator.** "Allowlist + sha256 + sandbox + manifest-first" is the OWASP-aligned posture. Competitors will ship `pip install transduce-mode-X` auto-discovery and eventually have an incident; transduce avoids it by construction.
4. **Mode catalog as an early-stage network asset.** The first 50 well-designed, eval-benchmarked modes are useful, but only marginally defensible — modes are easy to fork. The catalog is a contribution magnet, not a moat. Falsifiable test: at month 6, count non-author contributors with ≥10-deployment mode uptake. If <3, drop the catalog moat thesis explicitly.
5. **Mindshare as the default primitive.** Pandoc has no technical moat either. It is the default because it was first, stayed honest about scope, and never tried to be more than a primitive.

This is **category creation through methodology, not moat construction**. The win condition is being the answer when someone asks "what's the standard way to do verified text transformation in our stack." That is a 12–18 month positioning play, not a feature race.

---

## Examples — what gets built on top

| Surface | Mode | Use |
|---|---|---|
| CMS plugin | `dejargon` | Pre-publish jargon-density reduction with KPI preservation |
| IDE assistant | `voice-match` | Match commit message style to repo history |
| Localization pipeline | `tone.us-to-uk` | Regional register shift |
| Customer support tool | `formal-to-warm` | Response humanization with audit trail |
| Accessibility tool | `simplify.grade-8` | Reading-level normalization |
| Social media drafter | `length.normalize:280` | Twitter/X compression |
| Personal writing app | `style.match:<sample>` | Voice transfer from author samples |
| Documentation site | `audience.shift:beginner` | Readership adaptation |
| Editorial tool | `de-passive` | Active voice enforcement |
| Sales tool | `compress.elevator-pitch` | Pitch normalization |
| Browser extension | `register.casual` | Compose-pane register adjustment (third-party `humanize.*` modes available out-of-core for users who want them) |

Each is a thin client. The transformation logic, the ensemble verification, the local model orchestration — all in `transduce`.

---

## Roadmap (high-level)

Halved from the original draft after pressure-testing against realistic engineering effort.

| Phase | Scope | Time |
|---|---|---|
| **v0** | API skeleton, Ollama backend, 3 seed modes (`dejargon`, `register.casual`, `length.normalize`), cosine + entity/number/URL preservation scorers, word diff | Weekend |
| **v0.5** | Verification ensemble (NLI scorer + negation-diff + HHEM); retry-with-targeted-feedback; Spotlighting fence on prompt template; ingress injection scanner; allowlisted plugin loader with sha256 pinning | 2–3 weeks |
| **v1** | OpenAPI spec, Docker Compose, contribution docs, Anthropic + vLLM backends, mode versioning with multi-version dispatch, composite verifier across compose chains, streaming-with-advisory-verify, per-request cost budget, language detection + `supported_languages`, OTel GenAI SemConv alignment, 8 seed modes (no humanize) | 5–6 weeks |
| **v1.5** | Batch endpoints, MCP server façade, public `transduce-faithfulness` benchmark, eval harness wired to `attest`, attention-probe scorer for local backends, mode-introspection endpoint (`POST /v1/modes/{id}/render`) | 8 weeks |
| **v2** | Mode marketplace metadata, multi-tenant config, signed-mode-only enforcement default, optional Rust extraction for verifier hot path (only if profiling justifies) | 16 weeks |

---

## Connection to existing portfolio

| Project | Relationship |
|---|---|
| `attest` | Provides the eval substrate. Mode quality is measured via attest assertions against the `transduce-faithfulness` benchmark. CI fails on >2% per-scorer drop |
| `anneal` | Mode prompt optimization is a clean AEA triplet — Artifact = transformed text, Eval = ensemble scorer outputs, Agent = prompt + model variant. Compatible with DSPy-style compilation |
| `archex` | Independent. Possible cross-pollination: `transduce` as a mode within `archex` for code-comment register normalization |
| `armory` | Reference clients live here as Claude Code skills |
| `SIEVE` | Verifier protocol mirrors SIEVE's selective-execution discipline; both reject on quality floors rather than accepting low-confidence output |

`transduce` is the first `determ-ai` project that is *unambiguously a service*, not a framework or a skill bundle. It tests whether the org can ship and operate infrastructure, not just primitives — and whether the verification methodology bet pays off as a category-defining standard.
