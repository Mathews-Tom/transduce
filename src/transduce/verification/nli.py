"""Bidirectional NLI scorer (P2-VER-02).

The scorer wraps an injectable ``Entailer`` callable so unit tests exercise
the bidirectional logic without loading the heavy NLI model. Production
wiring constructs a MiniCheck-Flan-T5-Large entailer via
:func:`build_minicheck_entailer` per ADR-0003.

Bidirectionality is the point. Catching ``original ⊨ candidate`` alone misses
hallucinated qualifiers that the candidate adds (the candidate would still
be entailed by the original); catching ``candidate ⊨ original`` alone misses
silent drops where the candidate is a strict subset. Both directions must
clear the threshold for accept.

Long inputs are chunked deterministically: tokenised by sentence (period or
newline boundaries) and concatenated until each chunk fits the configured
character budget. Each candidate chunk is scored against the corresponding
original chunk; the minimum direction-score over chunks decides the verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from transduce.verification.base import ScoreResult

Entailer = Callable[[str, str], float]
"""Premise-hypothesis entailment callable returning a probability in [0, 1]."""

_DEFAULT_THRESHOLD: Final[float] = 0.70
_DEFAULT_MAX_CHARS: Final[int] = 1500
_DEFAULT_MODEL: Final[str] = "lytang/MiniCheck-Flan-T5-Large"


@dataclass(frozen=True)
class _DirectionScore:
    """Outcome of a single direction over the chunked input."""

    minimum: float
    failing_chunk: str | None


class BidirectionalNLIScorer:
    """Reject when either entailment direction falls below ``threshold``."""

    name = "bidirectional_nli"

    def __init__(
        self,
        *,
        entail: Entailer,
        threshold: float = _DEFAULT_THRESHOLD,
        max_chunk_chars: int = _DEFAULT_MAX_CHARS,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold}")
        if max_chunk_chars <= 0:
            raise ValueError(f"max_chunk_chars must be positive, got {max_chunk_chars}")
        self._entail = entail
        self._threshold = threshold
        self._max_chunk_chars = max_chunk_chars

    def score(self, original: str, candidate: str) -> ScoreResult:
        if not original or not candidate:
            raise ValueError("nli scorer requires non-empty original and candidate")
        forward = self._direction_score(original, candidate)
        if forward.minimum < self._threshold:
            return self._reject(forward, direction="forward")
        backward = self._direction_score(candidate, original)
        if backward.minimum < self._threshold:
            return self._reject(backward, direction="backward")
        aggregate = min(forward.minimum, backward.minimum)
        return ScoreResult(name=self.name, value=aggregate, verdict="accept")

    def _direction_score(self, premise: str, hypothesis: str) -> _DirectionScore:
        premise_chunks = list(_chunk_text(premise, self._max_chunk_chars))
        hypothesis_chunks = list(_chunk_text(hypothesis, self._max_chunk_chars))
        pairs = zip(premise_chunks, hypothesis_chunks, strict=False)
        minimum = 1.0
        failing_chunk: str | None = None
        for premise_chunk, hypothesis_chunk in pairs:
            score = self._entail(premise_chunk, hypothesis_chunk)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"entailer returned out-of-range score {score} for chunk pair")
            if score < minimum:
                minimum = score
                failing_chunk = hypothesis_chunk
        if len(premise_chunks) != len(hypothesis_chunks):
            mismatch = self._entail(premise, hypothesis)
            if not 0.0 <= mismatch <= 1.0:
                raise ValueError(f"entailer returned out-of-range score {mismatch} for full input")
            if mismatch < minimum:
                minimum = mismatch
                failing_chunk = hypothesis
        return _DirectionScore(minimum=minimum, failing_chunk=failing_chunk)

    def _reject(self, direction_score: _DirectionScore, *, direction: str) -> ScoreResult:
        return ScoreResult(
            name=self.name,
            value=direction_score.minimum,
            verdict="reject",
            rejection_reason=(
                f"nli {direction} entailment {direction_score.minimum:.3f} "
                f"below threshold {self._threshold:.3f}"
            ),
            span=direction_score.failing_chunk,
        )


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` on sentence boundaries into chunks of at most ``max_chars``."""
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sentence
            continue
        current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current.strip())
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter on `.`, `!`, `?`, and newline boundaries."""
    pieces: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in ".!?\n":
            pieces.append("".join(buffer).strip())
            buffer = []
    if buffer:
        tail = "".join(buffer).strip()
        if tail:
            pieces.append(tail)
    return [piece for piece in pieces if piece]


def build_minicheck_entailer(
    model_name: str = _DEFAULT_MODEL,
) -> Entailer:  # pragma: no cover - exercised in slow integration; covers IO at startup
    """Build a MiniCheck-Flan-T5-Large entailer for production use.

    Lazy-imports ``transformers`` and the MiniCheck factory so unit tests
    avoid the model-load cost. Operators bear the one-time download on
    first call. Failures (missing model, missing dependency) raise
    immediately.
    """
    from minicheck.minicheck import MiniCheck  # type: ignore[import-not-found]

    scorer = MiniCheck(model_name=model_name, enable_prefix_caching=False)

    def entail(premise: str, hypothesis: str) -> float:
        _, raw_probs, *_ = scorer.score(docs=[premise], claims=[hypothesis])
        if not raw_probs:
            raise RuntimeError("minicheck returned an empty probability list")
        probability = float(raw_probs[0])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"minicheck returned out-of-range probability {probability}")
        return probability

    return entail


__all__ = [
    "BidirectionalNLIScorer",
    "Entailer",
    "build_minicheck_entailer",
]
