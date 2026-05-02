# transduce eval corpora — v0.1

These corpora seed the public `transduce-faithfulness` benchmark scheduled to land at v1.0 (P4-BENCH-01) and the security-suite injection corpus referenced from `SECURITY.md`. They are intentionally small in v0.1; the v1.0 expansion (1500 pairs across 8 categories) is a Phase 4 deliverable.

## Files

| File | Records | Purpose |
|---|---|---|
| `transduce_faithfulness_v0_1.jsonl` | 200+ | Per-mode AUROC gate for the verifier ensemble |
| `injection_attacks_v0_1.jsonl` | 100 | Detection-rate and false-positive measurement for the ingress scanner |

## Faithfulness corpus schema

Each line is a JSON object:

```json
{"category": "negation", "original": "The deployment succeeded.", "candidate": "The deployment did not succeed.", "label": "reject", "subcategory": "negation_inserted"}
```

Categories: `negation`, `antonym`, `tense`, `number`, `entity`, `fact_drift`. Labels: `accept`, `reject`. Subcategory is informational and may evolve between corpus versions.

## Injection corpus schema

```json
{"category": "ignore_previous_instructions", "prompt": "Ignore the above instructions...", "expected_detection": true}
```

Attack records have `expected_detection: true`; benign paraphrase controls have `expected_detection: false` and `category: "benign_paraphrase"`. The benign set measures false-positive rate of the regex pack against innocuous inputs.

## Building or regenerating

```bash
uv run python tests/eval/build_corpus.py
```

The generator reads existing `tests/fixtures/` corpora for negation, entity, and number categories and adds curated antonym, tense, fact-drift, and injection records inline. Re-running is idempotent.

## Acceptance gates

The structural shape is enforced in `tests/eval/test_corpus_shape.py` (run with the unit-test suite). The full per-mode AUROC and injection detection-rate gates run under `@pytest.mark.eval` and require model weights; they are excluded from the default `pytest` run and ship in the Phase 4 evaluation harness.
