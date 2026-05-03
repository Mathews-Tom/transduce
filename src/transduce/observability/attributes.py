"""OTel attribute names for transduce spans (P3-OBS-01..03).

The ``gen_ai.*`` namespace mirrors the OTel GenAI Semantic Conventions
(experimental as of late 2025). The ``transduce.*`` namespace covers
transduce-specific extensions documented in
``docs/system-design.md`` §Observability and ``docs/observability.md``.

Constants live here so the orchestrator and the HTTP handlers reference
the same string and grep finds every emission site at once. Risk R-05
(SemConv churn) is mitigated by funnelling every name through this
module — a future SemConv rename surfaces as a one-line edit here.
"""

from __future__ import annotations

from typing import Final

# ----- Standard gen_ai.* attributes ----------------------------------------
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_FINISH_REASONS: Final[str] = "gen_ai.response.finish_reasons"

# ----- transduce.* extensions ---------------------------------------------
TRANSDUCE_MODE_ID: Final[str] = "transduce.mode.id"
TRANSDUCE_MODE_VERSION: Final[str] = "transduce.mode.version"
TRANSDUCE_LANGUAGE: Final[str] = "transduce.language"
TRANSDUCE_VERDICT: Final[str] = "transduce.verdict"
TRANSDUCE_RETRIES: Final[str] = "transduce.retries"
TRANSDUCE_COST_USD: Final[str] = "transduce.cost_usd"
TRANSDUCE_ATTEMPT: Final[str] = "transduce.attempt"

TRANSDUCE_TEXT_SHA256_8: Final[str] = "transduce.text.sha256_8"
TRANSDUCE_TEXT_LENGTH: Final[str] = "transduce.text.length"
TRANSDUCE_TEXT_VALUE: Final[str] = "transduce.text.value"
"""Raw text, emitted only when ``debug.include_text=true``."""

TRANSDUCE_SCAN_MATCHED_PATTERN: Final[str] = "transduce.scan.matched_pattern"

TRANSDUCE_SCORER_COSINE: Final[str] = "transduce.scorer.cosine"
TRANSDUCE_SCORER_NLI_FORWARD: Final[str] = "transduce.scorer.nli_forward"
TRANSDUCE_SCORER_NLI_BACKWARD: Final[str] = "transduce.scorer.nli_backward"
TRANSDUCE_SCORER_HHEM: Final[str] = "transduce.scorer.hhem"
TRANSDUCE_SCORER_NEGATION_DIFF_COUNT: Final[str] = "transduce.scorer.negation_diff_count"
TRANSDUCE_REJECTION_REASON: Final[str] = "transduce.rejection_reason"

TRANSDUCE_COMPOSE_STAGES: Final[str] = "transduce.compose.stages"
TRANSDUCE_COMPOSE_DRIFT_TOTAL: Final[str] = "transduce.compose.drift_total"

TRANSDUCE_DIFF_OPS_COUNT: Final[str] = "transduce.diff.ops_count"
TRANSDUCE_DIFF_CHARS_CHANGED: Final[str] = "transduce.diff.chars_changed"

# ----- Span names -----------------------------------------------------------
SPAN_REQUEST: Final[str] = "gen_ai.client.request"
SPAN_SCAN: Final[str] = "transduce.scan"
SPAN_GENERATE: Final[str] = "transduce.generate"
SPAN_VERIFY: Final[str] = "transduce.verify"
SPAN_COMPOSE: Final[str] = "transduce.compose"
SPAN_DIFF: Final[str] = "transduce.diff"

# ----- Provider name surfaced as gen_ai.system -----------------------------
GEN_AI_SYSTEM_TRANSDUCE: Final[str] = "transduce"
