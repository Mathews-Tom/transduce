"""Length delta scorer (P2-VER-05).

Bound-aware length check that protects against two distinct failure modes:
truncation (candidate too short to be a faithful paraphrase) and
injection-style padding (candidate >2x original, suggesting the model was
nudged into producing system-prompt leakage or appended instructions).

The default lower bound is ``0.4 x original length`` and upper bound is
``2.0 x original length``; modes that opt into wider ranges (e.g.,
``length.normalize:280``) override via ``LengthDeltaScorer`` constructor
arguments.
"""

from __future__ import annotations

from transduce.verification.base import ScoreResult


class LengthDeltaScorer:
    """Reject when candidate length falls outside the configured bounds."""

    name = "length_delta"

    def __init__(
        self,
        *,
        min_ratio: float = 0.4,
        max_ratio: float = 2.0,
        absolute_min_chars: int = 1,
    ) -> None:
        if min_ratio < 0:
            raise ValueError(f"min_ratio must be non-negative, got {min_ratio}")
        if max_ratio <= 0:
            raise ValueError(f"max_ratio must be positive, got {max_ratio}")
        if min_ratio > max_ratio:
            raise ValueError(f"min_ratio {min_ratio} cannot exceed max_ratio {max_ratio}")
        if absolute_min_chars < 0:
            raise ValueError(f"absolute_min_chars must be non-negative, got {absolute_min_chars}")
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._absolute_min_chars = absolute_min_chars

    def score(self, original: str, candidate: str) -> ScoreResult:
        if not original:
            raise ValueError("length scorer requires a non-empty original")
        original_len = len(original)
        candidate_len = len(candidate)
        ratio = candidate_len / original_len
        lower_bound_chars = max(
            self._absolute_min_chars,
            int(original_len * self._min_ratio),
        )
        upper_bound_chars = int(original_len * self._max_ratio)
        if candidate_len < lower_bound_chars:
            return ScoreResult(
                name=self.name,
                value=0.0,
                verdict="reject",
                rejection_reason=(
                    f"candidate length {candidate_len} below lower bound "
                    f"{lower_bound_chars} ({ratio:.2f}x original)"
                ),
                span=str(candidate_len),
            )
        if candidate_len > upper_bound_chars:
            return ScoreResult(
                name=self.name,
                value=0.0,
                verdict="reject",
                rejection_reason=(
                    f"candidate length {candidate_len} above upper bound "
                    f"{upper_bound_chars} ({ratio:.2f}x original)"
                ),
                span=str(candidate_len),
            )
        return ScoreResult(name=self.name, value=1.0, verdict="accept")


__all__ = ["LengthDeltaScorer"]
