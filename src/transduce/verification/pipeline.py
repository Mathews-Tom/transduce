"""Sequential verifier pipeline with first-fail short-circuit (P1-VER-05).

Runs an ordered list of scorers and returns either the first failing
scorer's result (with all preceding accepts attached for diagnostics) or
an aggregate accept once every scorer passes. Order matters: cosine acts
as the coarse pre-filter, preservation scorers run after to catch fine-
grained drift, and the v0.5 NLI/HHEM scorers slot in between (P2-VER-02,
P2-VER-03) per docs/system-design.md §Verification Subsystem.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from transduce.verification.base import Scorer, ScoreResult


@dataclass(frozen=True)
class PipelineOutcome:
    """Aggregate outcome of a verifier pipeline run."""

    verdict: Literal["accept", "reject"]
    results: Sequence[ScoreResult]
    failed_scorer: str | None = None
    rejection_reason: str | None = None
    span: str | None = None


class VerifierPipeline:
    """Run scorers in order; short-circuit and return on the first reject."""

    def __init__(self, scorers: Iterable[Scorer]) -> None:
        scorers_list = list(scorers)
        if not scorers_list:
            raise ValueError("VerifierPipeline requires at least one scorer")
        self._scorers = tuple(scorers_list)

    @property
    def scorers(self) -> tuple[Scorer, ...]:
        return self._scorers

    def run(self, original: str, candidate: str) -> PipelineOutcome:
        results: list[ScoreResult] = []
        for scorer in self._scorers:
            result = scorer.score(original, candidate)
            results.append(result)
            if result.verdict == "reject":
                return PipelineOutcome(
                    verdict="reject",
                    results=tuple(results),
                    failed_scorer=result.name,
                    rejection_reason=result.rejection_reason,
                    span=result.span,
                )
        return PipelineOutcome(verdict="accept", results=tuple(results))
