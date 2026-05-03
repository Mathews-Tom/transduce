# transduce — Observability

This document specifies the OpenTelemetry surface transduce emits and the operator-facing knobs that control it. It complements `docs/system-design.md` §Observability with the per-attribute reference and the collector setup recipes.

> **Status:** v1. The OTel `gen_ai.*` Semantic Conventions are still listed as experimental upstream; transduce pins to a tested SemConv release range and surfaces drift via the `transduce.*` extension namespace so operator dashboards stay stable.

---

## Configuration

The `observability` section of `transduce.yaml`:

```yaml
observability:
  enabled: false              # default off; flip to true to install the global tracer
  otel_endpoint: null         # optional OTLP HTTP collector URL; null = capture only, no export
  semconv: gen_ai             # only value supported in v1
  redact_text_in_spans: true  # privacy default; sha256_8 + length only
  debug_include_text: false   # opt-in for non-prod debugging; requires redact_text_in_spans=false
```

When `enabled` is false the service runs against the OpenTelemetry global no-op tracer. Span emission is a near-zero-cost no-op and the metrics counters under `/metrics` continue to function — observability and metrics are independent surfaces.

The validator forbids `debug_include_text=true` together with `redact_text_in_spans=true`. The runtime treats that combination defensively as redact-only, but the config will refuse to load before that path matters.

The CLI prints a stderr warning whenever `debug_include_text=true` is set so misconfigured non-prod deployments are visible at startup rather than discovered through a leaked span.

---

## Span hierarchy

Every request emits one parent span with at most five children:

```text
gen_ai.client.request
├── transduce.scan
├── transduce.generate           (one per generate attempt; up to max_retries+1)
├── transduce.verify             (one per generate attempt)
├── transduce.compose            (compose chains only; wraps the per-stage spans)
│   ├── transduce.generate       (per stage, per attempt)
│   ├── transduce.verify         (per stage, per attempt)
│   └── transduce.verify         (composite verifier — the end-to-end check)
└── transduce.diff
```

The streaming endpoint emits the same `transduce.generate`, `transduce.verify`, and `transduce.diff` spans as the non-streaming endpoint. There is exactly one generate span per stream because advisory streaming is single-attempt by contract (P3-STR-01..03).

---

## Standard `gen_ai.*` attributes

The OTel GenAI Semantic Conventions namespace these to model-call telemetry. transduce emits them on `gen_ai.client.request` and on every `transduce.generate` span.

| Attribute | Type | Source | Notes |
|---|---|---|---|
| `gen_ai.system` | string | constant `"transduce"` | Identifies transduce as the request-issuing system; the underlying provider name appears on the backend `BackendInfo` and as the metric label |
| `gen_ai.request.model` | string | `backend.model` | Operator-configured model alias (no hardcoded model strings in transduce code) |
| `gen_ai.usage.input_tokens` | int | `GenerationResult.tokens_in` / `StreamFinal.tokens_in` | Set after the backend returns; zero for backends that omit usage |
| `gen_ai.usage.output_tokens` | int | `GenerationResult.tokens_out` / `StreamFinal.tokens_out` | Same as above |
| `gen_ai.response.finish_reasons` | string[] | reserved for future backends that surface finish reasons | Not emitted in v1; ships when the backend protocol gains the field |

Risk register entry **R-05** ("OTel SemConv breaking changes") tracks this surface. If upstream renames any of these keys, the rename is one line in `src/transduce/observability/attributes.py` and the constants ripple through every emission site.

---

## `transduce.*` extension attributes

Transduce-specific extensions sit under the `transduce.` namespace so a collector can grep them without a SemConv conflict.

### Per-request (`gen_ai.client.request` parent span)

| Attribute | Type | Notes |
|---|---|---|
| `transduce.mode.id` | string | The mode-id from the request body (or `"compose"` for a chain) |
| `transduce.language` | string | ISO-639-1 code returned by the language detector |
| `transduce.verdict` | string | `"accept"` on success; the parent span is not set on failure paths — the error envelope carries the rejection |
| `transduce.retries` | int | Number of retries the orchestrator performed (0 means accepted on first attempt) |
| `transduce.cost_usd` | float | Total USD cost across attempts (zero for local backends) |

### `transduce.scan`

| Attribute | Type | Notes |
|---|---|---|
| `transduce.scan.matched_pattern` | string | The injection-scanner category that matched, or `"clean"` for no hit |

### `transduce.generate`

| Attribute | Type | Notes |
|---|---|---|
| `transduce.mode.id` | string | Mode the generate is rendering for (per-stage in compose chains) |
| `transduce.mode.version` | string | Resolved mode version |
| `transduce.attempt` | int | 1-indexed attempt count within the orchestrator's retry loop |
| `transduce.cost_usd` | float | This attempt's USD cost, ignoring earlier attempts (reserved; populated in v1.5) |

### `transduce.verify`

| Attribute | Type | Notes |
|---|---|---|
| `transduce.mode.id` | string | Same as the generate span this verify follows |
| `transduce.mode.version` | string | Same |
| `transduce.verdict` | string | `"accept"` or `"reject"` for this verify call |
| `transduce.scorer.cosine` | float | Set when the cosine scorer ran; absent when an earlier scorer short-circuited |
| `transduce.scorer.nli_forward` | float | `original ⊨ candidate` direction score |
| `transduce.scorer.nli_backward` | float | `candidate ⊨ original` direction score |
| `transduce.scorer.hhem` | float | HHEM-2.1 cross-encoder score |
| `transduce.scorer.negation_diff_count` | int | Sum of added + removed negation cues; zero on accept |
| `transduce.rejection_reason` | string | Set only when `transduce.verdict == "reject"`; copies `failed_scorer` |

### `transduce.compose`

| Attribute | Type | Notes |
|---|---|---|
| `transduce.compose.stages` | int | Number of modes in the chain |
| `transduce.compose.drift_total` | float | `1 - composite_aggregate_score`; higher means more drift introduced across the chain |

### `transduce.diff`

| Attribute | Type | Notes |
|---|---|---|
| `transduce.diff.ops_count` | int | Number of `equal`/`insert`/`delete` operations the diff produced |
| `transduce.diff.chars_changed` | int | Total characters across non-`equal` operations |

---

## Redaction policy

Raw request and candidate text are **banned** from span attributes by default. The `SpanEmitter.text_attributes` helper emits two safe attributes for any text it accepts:

- `transduce.text.sha256_8` — first 8 hex chars of `sha256(text.encode("utf-8"))`. 32 bits is enough to disambiguate concurrent requests in a trace UI without leaking content.
- `transduce.text.length` — character length.

The `transduce.text.value` attribute carries raw text **only** when both `redact_text_in_spans=false` and `debug_include_text=true`. The pairing is enforced by the config validator, so a non-prod deployment that wants raw text in spans must opt in twice and accept the stderr warning at startup.

> **Privacy contract:** transduce never logs raw text by default. If a downstream tool needs full text for debugging, run a single non-production session with `debug_include_text=true` and discard the trace store afterwards.

---

## Metrics

Prometheus counters and histograms exposed via `GET /metrics`:

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `transduce_requests_total` | counter | `mode`, `verdict` | `verdict ∈ {accept, error}`; compose chains use `+`-joined mode ids |
| `transduce_generation_duration_ms` | histogram | `backend`, `mode` | Generate-stage latency in ms |
| `transduce_verification_failures_total` | counter | `mode`, `scorer` | Increments per scorer rejection (note: orchestrator stops at the first reject, so this is also "first-failing-scorer count") |
| `transduce_retry_count` | counter | `mode` | Total retries across all requests (reserved; emitted in v1.5) |
| `transduce_generation_cost_usd_total` | counter | `backend`, `mode` | Cumulative USD spend per backend+mode |
| `transduce_concurrency_rejections_total` | counter | `backend` | Increments when `SemaphoreBackend` returns 429 |
| `transduce_injection_detected_total` | counter | `category` | Increments on injection-scanner hits, label = matched category |
| `transduce_language_unsupported_total` | counter | `mode`, `lang` | Increments on `LANGUAGE_NOT_SUPPORTED` 415s |

Metrics emit unconditionally — they are not gated by `observability.enabled`.

---

## Collector setup

### OTLP HTTP (vendor-neutral)

Set the environment variable `TRANSDUCE_OTEL_ENDPOINT` and the config:

```yaml
observability:
  enabled: true
  otel_endpoint: ${TRANSDUCE_OTEL_ENDPOINT:-http://otel-collector:4318}
```

Any OTLP-capable backend (Jaeger 1.55+, Tempo, Honeycomb, New Relic, Datadog, AWS X-Ray with the OTel collector) accepts spans on port 4318 (HTTP) or 4317 (gRPC; transduce ships HTTP only).

### Jaeger (local development)

```yaml
# docker-compose.observability.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.62
    ports:
      - "16686:16686"   # UI
      - "4318:4318"     # OTLP HTTP

  transduce:
    environment:
      TRANSDUCE_OTEL_ENDPOINT: "http://jaeger:4318/v1/traces"
```

Open `http://localhost:16686` and search for service `transduce`.

### Phoenix (LLM-aware)

Phoenix is OTel-native and understands `gen_ai.*` SemConv out of the box.

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:9
    ports:
      - "6006:6006"
      - "4318:4318"

  transduce:
    environment:
      TRANSDUCE_OTEL_ENDPOINT: "http://phoenix:4318/v1/traces"
```

The Phoenix UI groups spans by `gen_ai.system` and surfaces token-cost panels keyed on `gen_ai.usage.*`.

### Langfuse

```yaml
services:
  langfuse-otel-collector:
    image: otel/opentelemetry-collector:0.110.0
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes:
      - ./otel-config.yaml:/etc/otelcol/config.yaml:ro
    ports:
      - "4318:4318"
```

```yaml
# otel-config.yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

exporters:
  otlphttp/langfuse:
    endpoint: https://cloud.langfuse.com/api/public/otel
    headers:
      authorization: "Basic ${LANGFUSE_AUTH}"

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp/langfuse]
```

Set `LANGFUSE_AUTH` to a base64-encoded `<public_key>:<secret_key>` pair from your Langfuse project settings.

### In-process testing (no collector)

Set `enabled: true` with `otel_endpoint: null`. The TracerProvider is built without an exporter, so spans are produced but never leave the process. This is the configuration the integration suite uses to assert span attributes via an `InMemorySpanExporter`.

---

## Querying common operator questions

Once spans flow into a collector, these searches answer the day-to-day operational questions:

| Question | Trace query |
|---|---|
| Which modes are rejecting most often? | Filter by `transduce.verdict = "reject"`, group by `transduce.mode.id` |
| Are we paying retries for a specific backend? | Filter by `gen_ai.system = "transduce"`, group by `gen_ai.request.model`, sum `transduce.retries` |
| How much are we spending on cloud calls? | Filter `gen_ai.client.request`, group by `gen_ai.request.model`, sum `transduce.cost_usd` |
| Did the injection scanner block anything today? | Filter spans named `transduce.scan` where `transduce.scan.matched_pattern != "clean"` |
| Which scorer is the bottleneck? | Filter `transduce.verify` spans, sort by duration |
| Compose chain accumulating drift? | Filter `transduce.compose` spans, sort by `transduce.compose.drift_total` desc |

---

## What this design deliberately excludes

| Excluded | Why |
|---|---|
| Raw text or `last_candidate` in span attributes | Privacy by default; opt-in only for non-prod debugging |
| Logging of API keys, bearer tokens, or any value reaching the backend | The backend adapter layer never logs the request body; metric and span attributes never embed secrets |
| Inline trace context propagation across multi-tenant boundaries | Multi-tenancy ships in v2; v1 traces a single tenant per process |
| Auto-instrumentation of Litestar internals | We surface only the spans documented above; auto-instrumentation would surface internal Litestar frames that obscure the verifier and backend semantics |
| `gen_ai.prompt` / `gen_ai.completion` raw-text attributes | Forbidden by the redaction policy; the SemConv extension that adds them is opt-in upstream and stays opt-in here |
