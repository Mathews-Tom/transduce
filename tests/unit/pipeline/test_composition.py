"""Unit tests for compose-chain helpers (P3-COMP-03..04)."""

from __future__ import annotations

import pytest

from transduce.pipeline.composition import per_stage_intensity, preservation_union
from transduce.registry.spec import PreserveRule

pytestmark = pytest.mark.unit


def test_per_stage_intensity_single_stage_returns_global() -> None:
    assert per_stage_intensity(global_intensity=0.6, n_stages=1) == pytest.approx(0.6)


def test_per_stage_intensity_three_stages_at_zero_six_returns_root_distribution() -> None:
    stage_intensity = per_stage_intensity(global_intensity=0.6, n_stages=3)
    expected = 1.0 - (1.0 - 0.6) ** (1.0 / 3)

    assert stage_intensity == pytest.approx(expected)


def test_per_stage_intensity_composes_back_to_global_for_three_stages() -> None:
    stage_intensity = per_stage_intensity(global_intensity=0.5, n_stages=3)
    composed = 1.0 - (1.0 - stage_intensity) ** 3

    assert composed == pytest.approx(0.5)


def test_per_stage_intensity_zero_global_returns_zero_for_any_n() -> None:
    assert per_stage_intensity(global_intensity=0.0, n_stages=5) == pytest.approx(0.0)


def test_per_stage_intensity_one_global_returns_one_for_any_n() -> None:
    assert per_stage_intensity(global_intensity=1.0, n_stages=4) == pytest.approx(1.0)


def test_per_stage_intensity_rejects_global_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="global_intensity"):
        per_stage_intensity(global_intensity=1.5, n_stages=3)


def test_per_stage_intensity_rejects_zero_stages() -> None:
    with pytest.raises(ValueError, match="n_stages"):
        per_stage_intensity(global_intensity=0.5, n_stages=0)


def test_preservation_union_combines_disjoint_sets() -> None:
    union = preservation_union(
        [
            (PreserveRule.ENTITIES,),
            (PreserveRule.URLS,),
        ]
    )

    assert set(union) == {PreserveRule.ENTITIES, PreserveRule.URLS}


def test_preservation_union_deduplicates_overlapping_sets() -> None:
    union = preservation_union(
        [
            (PreserveRule.ENTITIES, PreserveRule.NUMBERS),
            (PreserveRule.NUMBERS, PreserveRule.URLS),
        ]
    )

    assert set(union) == {
        PreserveRule.ENTITIES,
        PreserveRule.NUMBERS,
        PreserveRule.URLS,
    }


def test_preservation_union_preserves_first_seen_order() -> None:
    union = preservation_union(
        [
            (PreserveRule.URLS, PreserveRule.ENTITIES),
            (PreserveRule.NUMBERS, PreserveRule.URLS),
        ]
    )

    assert union == (PreserveRule.URLS, PreserveRule.ENTITIES, PreserveRule.NUMBERS)


def test_preservation_union_returns_empty_for_empty_input() -> None:
    union = preservation_union([])

    assert union == ()
