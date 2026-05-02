"""Word-level diff for transform responses (P1-DIFF-01).

Wraps the maintained ``diff-match-patch`` library with semantic cleanup to
factor out coincidental commonalities. The result is a list of structured
``DiffOp`` operations the API layer can serialise; clients render however
they like (yellow highlights, inline strikethrough, two-pane). Returning
structured operations keeps the service free of presentation concerns per
docs/system-design.md §Diff Generator.
"""

from __future__ import annotations

from typing import Final, Literal

from diff_match_patch import diff_match_patch  # type: ignore[import-untyped]

from transduce.api.schemas import DiffOp

_OpLabel = Literal["equal", "insert", "delete"]
_OP_LABELS: Final[dict[int, _OpLabel]] = {
    0: "equal",
    -1: "delete",
    1: "insert",
}


def compute_diff(original: str, candidate: str) -> list[DiffOp]:
    """Compute the word-level diff between original and candidate text.

    Runs ``diff_main`` followed by ``diff_cleanupSemantic`` so adjacent
    insert/delete spans are grouped into human-readable chunks rather than
    character-level noise.

    Args:
        original: The source text submitted in the request.
        candidate: The transformed text returned by the backend.

    Returns:
        Ordered list of ``DiffOp`` instances representing the change-set.
        Empty-text segments are dropped so consumers can render directly
        without filtering.
    """
    engine = diff_match_patch()
    raw_diffs: list[tuple[int, str]] = engine.diff_main(original, candidate)
    engine.diff_cleanupSemantic(raw_diffs)
    return [DiffOp(op=_OP_LABELS[op], text=text) for op, text in raw_diffs if text]
