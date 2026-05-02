"""Shared types for the verification subsystem.

The ``Scorer`` Protocol is justified at v0 by three concrete implementations
(``EntityPreservationScorer``, ``NumberPreservationScorer``,
``UrlPreservationScorer``) plus the cosine scorer landing in the same
release. The v0.5 release widens the protocol surface with bidirectional
NLI and HHEM scorers (P2-VER-02, P2-VER-03).
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ScoreResult(BaseModel):
    """Outcome of a single scorer run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=1.0)
    verdict: Literal["accept", "reject"]
    rejection_reason: str | None = None
    span: str | None = None


class Scorer(Protocol):
    """Common surface every verification scorer implements."""

    name: str

    def score(
        self, original: str, candidate: str
    ) -> ScoreResult:  # pragma: no cover — Protocol method
        """Compare ``candidate`` against ``original`` and return a structured verdict."""
        ...
