"""Compose-chain helpers: intensity composition and preservation union (P3-COMP-03..04).

The compose-chain orchestrator delegates two concerns to the helpers
in this module so they can be unit-tested in isolation and reused by
the v1.5 batch endpoint:

- :func:`per_stage_intensity` distributes a global intensity across N
  stages multiplicatively. The closed-form ``1 - (1 - g) ** (1/n)``
  guarantees every stage receives an intensity at most equal to the
  global setting and that the composed effect approaches the global
  setting as n grows. Single-stage chains short-circuit to the global
  intensity verbatim.
- :func:`preservation_union` returns the union of preservation rules
  across every stage in the chain so a downstream stage cannot drop a
  rule the upstream stage required (P3-COMP-04).
"""

from __future__ import annotations

from collections.abc import Iterable

from transduce.registry.spec import PreserveRule


def per_stage_intensity(*, global_intensity: float, n_stages: int) -> float:
    """Return the per-stage intensity that composes to ``global_intensity``.

    Args:
        global_intensity: Caller-supplied intensity in ``[0.0, 1.0]``.
        n_stages: Number of stages in the chain. Must be ``>= 1``.

    Returns:
        Per-stage intensity ``1 - (1 - g) ** (1/n)``. For a single
        stage this is exactly ``global_intensity``.
    """
    if not 0.0 <= global_intensity <= 1.0:
        raise ValueError(f"global_intensity must be within [0.0, 1.0], got {global_intensity}")
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    if n_stages == 1:
        return global_intensity
    if global_intensity == 1.0:
        return 1.0
    return float(1.0 - (1.0 - global_intensity) ** (1.0 / n_stages))


def preservation_union(rule_sets: Iterable[Iterable[PreserveRule]]) -> tuple[PreserveRule, ...]:
    """Return the union of preservation rules across every stage's set.

    The union preserves the registry-load order of rules: if stage 1
    declared ``(ENTITIES, NUMBERS)`` and stage 2 declared
    ``(URLS, ENTITIES)``, the union is ``(ENTITIES, NUMBERS, URLS)``.
    """
    seen: dict[PreserveRule, None] = {}
    for rules in rule_sets:
        for rule in rules:
            seen.setdefault(rule, None)
    return tuple(seen.keys())


__all__ = ["per_stage_intensity", "preservation_union"]
