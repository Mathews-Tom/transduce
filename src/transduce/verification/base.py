"""Shared types for the verification subsystem.

The ``Scorer`` Protocol is realised by every scorer in the v0.5 ensemble:
cosine (coarse pre-filter), negation diff (deterministic floor),
bidirectional NLI (faithfulness signal), HHEM (cross-encoder factuality),
and the preservation/length scorers (deterministic checks). The v1 release
widens with optional Lookback/SelfCheck/LLM-judge scorers (P4-PROBE-01..03).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ScoreResult(BaseModel):
    """Outcome of a single scorer run.

    ``details`` carries opaque per-scorer metadata that does not fit on the
    protocol's single-float ``value`` (e.g., the bidirectional NLI scorer
    publishes its forward and backward direction scores under
    ``details["forward"]`` and ``details["backward"]``; the negation diff
    scorer publishes added/removed cue lists). Consumers that want a
    typed view import the scorer's helper, e.g., ``diff_negation_cues``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=1.0)
    verdict: Literal["accept", "reject"]
    rejection_reason: str | None = None
    span: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class Scorer(Protocol):
    """Common surface every verification scorer implements."""

    name: str

    def score(
        self, original: str, candidate: str
    ) -> ScoreResult:  # pragma: no cover — Protocol method
        """Compare ``candidate`` against ``original`` and return a structured verdict."""
        ...
