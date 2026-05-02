"""Cosine similarity scorer (P1-VER-01).

Uses an injectable embedding callable so unit tests exercise the cosine
arithmetic without downloading the fastembed ONNX model. Production wiring
constructs the scorer with :func:`build_fastembed_embedder` which lazy-loads
``BAAI/bge-small-en-v1.5`` once at startup per docs/system-design.md
§Verification Subsystem.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from transduce.verification.base import ScoreResult

Embedder = Callable[[str], Sequence[float]]

_DEFAULT_MODEL: Final[str] = "BAAI/bge-small-en-v1.5"


class CosineSimilarityScorer:
    """Reject when cosine similarity between the two embeddings falls below threshold."""

    name = "cosine_similarity"

    def __init__(self, *, embed: Embedder, threshold: float = 0.85) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold}")
        self._embed = embed
        self._threshold = threshold

    def score(self, original: str, candidate: str) -> ScoreResult:
        if not original or not candidate:
            raise ValueError("cosine scorer requires non-empty original and candidate")
        original_vec = list(self._embed(original))
        candidate_vec = list(self._embed(candidate))
        similarity = _cosine_similarity(original_vec, candidate_vec)
        clamped = max(0.0, min(1.0, similarity))
        if similarity >= self._threshold:
            return ScoreResult(name=self.name, value=clamped, verdict="accept")
        return ScoreResult(
            name=self.name,
            value=clamped,
            verdict="reject",
            rejection_reason=f"cosine {similarity:.3f} below threshold {self._threshold:.3f}",
        )


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimensions differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("zero-norm embedding; refusing to compute similarity")
    return dot / (norm_a * norm_b)


def build_fastembed_embedder(
    model_name: str = _DEFAULT_MODEL,
) -> Embedder:  # pragma: no cover — exercised in integration; covers IO at startup
    """Build a fastembed-backed embedder for production use.

    Lazy-imports fastembed and instantiates the ONNX model once. Operators
    bear the one-time download cost on first call. Failures (missing
    network, missing model) raise immediately.
    """
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name)

    def embed(text: str) -> list[float]:
        vector = next(iter(model.embed([text])))
        return list(vector)

    return embed
