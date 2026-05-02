"""Unit tests for the spotlight fence (P2-INJ-01)."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from transduce.injection.fence import build_fence

pytestmark = pytest.mark.unit


def test_fence_wraps_input_with_nonce() -> None:
    fence = build_fence("hello world")

    wrapped = fence.wrap("hello world")

    assert wrapped.startswith(fence.open_marker)
    assert wrapped.endswith(fence.close_marker)
    assert "hello world" in wrapped


def test_fence_open_and_close_share_the_same_nonce() -> None:
    fence = build_fence("hello world")

    pattern = re.compile(r"<<<USER_TEXT_([0-9a-f]+)>>>")
    open_match = pattern.match(fence.open_marker)
    close_match = re.match(r"<<<END_([0-9a-f]+)>>>", fence.close_marker)

    assert open_match is not None
    assert close_match is not None
    assert open_match.group(1) == close_match.group(1) == fence.nonce


def test_fence_nonce_is_32_hex_characters() -> None:
    fence = build_fence("hello world")

    assert len(fence.nonce) == 32
    assert re.fullmatch(r"[0-9a-f]+", fence.nonce) is not None


def test_fence_nonce_never_in_input() -> None:
    """The nonce must not collide with the user input."""
    for _ in range(20):
        fence = build_fence("benign user input")
        assert fence.nonce not in "benign user input"


def test_fence_regenerates_nonce_on_collision() -> None:
    """If the first nonce collides with the input, ``build_fence`` retries."""
    colliding = "deadbeefdeadbeefdeadbeefdeadbeef"
    sequence = iter([colliding, "feedfacefeedfacefeedfacefeedface"])

    def stub_token_hex(_n_bytes: int) -> str:
        return next(sequence)

    with patch("transduce.injection.fence.secrets.token_hex", side_effect=stub_token_hex):
        fence = build_fence(f"prefix {colliding} suffix")

    assert fence.nonce == "feedfacefeedfacefeedfacefeedface"


def test_fence_raises_when_collision_budget_exhausted() -> None:
    colliding = "deadbeefdeadbeefdeadbeefdeadbeef"

    def always_collide(_n_bytes: int) -> str:
        return colliding

    with (
        patch("transduce.injection.fence.secrets.token_hex", side_effect=always_collide),
        pytest.raises(RuntimeError, match="non-colliding"),
    ):
        build_fence(colliding)
