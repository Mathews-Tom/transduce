# ADR-0003 — Bidirectional NLI uses MiniCheck-Flan-T5-Large

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-05-02 |
| Deciders | @Mathews-Tom |
| Tags | verification, nli, model-selection |

## Context

The v0.5 verifier ensemble adds a bidirectional natural-language-inference (NLI) scorer between the cosine pre-filter and the HHEM cross-encoder. The scorer must check both `original ⊨ candidate` and `candidate ⊨ original` to catch silent fact drift that cosine does not see (`docs/system-design.md` §Verification Subsystem, dev-plan deliverable P2-VER-02). The dev plan Appendix C tags this Q-02 ("NLI model: AlignScore vs MiniCheck vs ensemble") for resolution before Phase 2 exit.

The performance gate sets a 50 ms p99 budget on GPU and a 500 ms p99 budget on CPU per single 500-character input. The acceptance bar for `transduce-faithfulness` v0.1 is AUROC >0.85 per mode and negation-flip recall ≥99%. The model choice has to clear these bars at a size that fits a CPU-only deployment topology.

Three candidate model families considered:

1. **AlignScore** — Liu et al., 2023; trained on a unified curation of seven NLI/QA/summarization-faithfulness datasets; ~125M parameters in the small variant. Public benchmark AUROC on summary faithfulness 0.78–0.82.
2. **MiniCheck-Flan-T5-Large** — Tang et al., 2024; 770M parameters; trained specifically for fact-checking summary outputs against source text; reports AUROC 0.86–0.89 on LLM-AggreFact and competitive scores against GPT-4-as-judge at ~$1 per 100K labels.
3. **DeBERTa-v3 NLI heads** (e.g., `cross-encoder/nli-deberta-v3-base`) — general-purpose NLI fine-tunes; smaller (180M) and faster but trained on MNLI/SNLI distributions that do not match the summary-faithfulness use case as tightly.

## Decision

The default bidirectional NLI scorer uses **MiniCheck-Flan-T5-Large** (`lytang/MiniCheck-Flan-T5-Large` on Hugging Face). The scorer wraps an injectable `Entailer = Callable[[str, str], float]` callable so unit tests run without the model and integration tests run against the real weights via `build_minicheck_entailer`. The DeBERTa NLI head is supported as a fallback under `verification.nli_model: cross-encoder/nli-deberta-v3-small` for CPU-only deployments where the 500 ms p99 budget cannot be hit with the larger model.

## Alternatives considered

- **AlignScore (small)** — rejected as the default. Lower AUROC on the summary-faithfulness distribution than MiniCheck (per Tang et al., 2024 Table 3), and weaker performance on tense/quantifier perturbations which is exactly what Phase 2 must catch. Kept available as a configurable alternative for ops who already have AlignScore weights cached.
- **DeBERTa-v3-base NLI head as default** — rejected because MNLI/SNLI training distributions favor short, single-sentence inferences. Long-context summary faithfulness is closer to MiniCheck's training set. Documented as the supported CPU-budget fallback with a startup warning that AUROC drops ~5 points relative to MiniCheck.
- **Ensemble (MiniCheck + AlignScore + GPT-4-judge)** — rejected as default because the ensemble is the *whole* Phase 2 verifier; running three NLI heads inside the NLI step doubles latency without measurable AUROC lift on the eval corpus pilot data. The verifier ensemble already aggregates NLI with HHEM, negation diff, cosine, and preservation rules — a second NLI ensemble would be circular.
- **GPT-4-as-judge as the NLI step** — rejected for cost reasons (Phase 2 budget gate is $0.05 per request) and because Phase 2 ships local-first defaults; cloud-only NLI defeats the topology principle.

## Consequences

### Positive

- AUROC ≥0.85 per mode on `transduce-faithfulness` v0.1 is achievable with the documented model and a 770M-parameter weight is small enough to ship via Hugging Face hub (≈1.5 GB) on operator machines.
- The injectable-callable boundary means unit-test runtimes stay under the 100 ms-per-test budget; only the integration suite pays the model-load cost.
- Choice aligns with Phase 4's `transduce-faithfulness` benchmark, which uses MiniCheck-FT5 as a published baseline.

### Negative

- 770M parameters on CPU exceeds the 500 ms p99 budget on commodity hardware without ONNX export or quantization. The performance gate explicitly anticipates this: documented mitigation is GPU recommendation or DeBERTa-v3-small fallback.
- Hugging Face hub dependency introduces a model-download step on first integration test run; CI bakes the weights into the test container layer to amortize.
- MiniCheck-FT5's license (Apache 2.0) is compatible with transduce's license but operators must surface it in their compliance review when distributing Docker images.

### Neutral

- Long inputs (>512 tokens after tokenization) are chunked deterministically by the scorer, with each chunk pair scored independently and the minimum direction-score returned. This keeps the entailment semantics monotone — a single failing chunk fails the whole pair — without inventing a custom long-context strategy.

## Compliance and verification

- `tests/unit/verification/test_nli.py` exercises the scorer against a stub entailer covering negation, hallucinated qualifiers, paraphrases, and chunked long inputs.
- Integration coverage under `@pytest.mark.integration and slow` runs the real MiniCheck weights against a 50-pair sample of `transduce-faithfulness` and asserts AUROC ≥0.85.
- The performance budget is verified by `@pytest.mark.perf` benchmarks runnable via `uv run pytest -m perf`. Failure of the CPU budget triggers the documented fallback flow per `docs/system-design.md` §Verification Subsystem latency table.
- Configuration documents the fallback in `transduce.example.yaml` `verification.nli_model` so operators can switch without code changes.

## References

- `.docs/development-plan.md` Appendix C Q-02 — NLI model selection
- `docs/system-design.md` §Verification Subsystem — Default scorers table
- Tang et al., 2024 — "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents" — <https://arxiv.org/abs/2404.10774>
- Liu et al., 2023 — "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function" — <https://arxiv.org/abs/2305.16739>
- MiniCheck-Flan-T5-Large model card — <https://huggingface.co/lytang/MiniCheck-Flan-T5-Large>
