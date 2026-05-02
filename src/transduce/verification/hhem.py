"""HHEM cross-encoder scorer (P2-VER-03).

Vectara HHEM-2.1 is the published hallucination-evaluation cross-encoder.
The scorer wraps an injectable ``HhemFactualityScorer`` callable so unit
tests run without the cross-encoder weights; production wiring constructs
a real instance via :func:`build_hhem_scorer`.

Single-direction by design: HHEM scores ``probability of factuality``
of the candidate against the original. Bidirectional checking is the
NLI scorer's job; HHEM lands as a complementary signal sourced from a
different training distribution. The default threshold (0.50) matches
the published Vectara guidance for hallucination flagging.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from transduce.verification.base import ScoreResult

HhemFactualityScorer = Callable[[str, str], float]
"""Source-claim factuality callable returning a probability in [0, 1]."""

_DEFAULT_THRESHOLD: Final[float] = 0.50
_DEFAULT_MODEL: Final[str] = "vectara/hallucination_evaluation_model"


class HHEMScorer:
    """Reject when HHEM factuality probability falls below ``threshold``."""

    name = "hhem_factuality"

    def __init__(
        self,
        *,
        scorer: HhemFactualityScorer,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold}")
        self._scorer = scorer
        self._threshold = threshold

    def score(self, original: str, candidate: str) -> ScoreResult:
        if not original or not candidate:
            raise ValueError("hhem scorer requires non-empty original and candidate")
        probability = self._scorer(original, candidate)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"hhem scorer returned out-of-range probability {probability}")
        if probability >= self._threshold:
            return ScoreResult(name=self.name, value=probability, verdict="accept")
        return ScoreResult(
            name=self.name,
            value=probability,
            verdict="reject",
            rejection_reason=(
                f"hhem factuality {probability:.3f} below threshold {self._threshold:.3f}"
            ),
        )


def build_hhem_scorer(
    model_name: str = _DEFAULT_MODEL,
    *,
    revision: str,
) -> HhemFactualityScorer:  # pragma: no cover - exercised in slow integration; covers IO at startup
    """Build a Vectara HHEM-2.1 factuality scorer for production use.

    Lazy-imports ``transformers`` so unit tests skip the weight load.
    Operators bear the one-time download cost on first call. Missing
    weights or missing optional dependency raises immediately.

    ``revision`` is required and pins the Hugging Face Hub commit; an
    attacker who compromises the repo cannot inject a different model
    without changing the pin in the operator's bootstrap. Bandit B615
    enforces this at CI time.
    """
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    model.eval()

    def score_pair(original: str, candidate: str) -> float:
        import torch  # type: ignore[import-not-found]

        with torch.no_grad():
            tokens = tokenizer(
                original,
                candidate,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            logits = model(**tokens).logits
            probability = float(torch.sigmoid(logits).flatten()[0])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"hhem returned out-of-range probability {probability}")
        return probability

    return score_pair


__all__ = [
    "HHEMScorer",
    "HhemFactualityScorer",
    "build_hhem_scorer",
]
