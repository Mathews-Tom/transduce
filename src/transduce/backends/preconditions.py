"""Backend selection preconditions (P3-BACK-09).

The dev plan ships ``min_model_b`` as a per-mode requirement
(``ModeSpec.backend_requirements.min_model_b``) and demands enforcement
as a 412 precondition per request, not as a doc note. Operators
declare the model size of each backend in
``BackendEntry.model_size_b``; the API handler consults this module
before the orchestrator runs so the precondition fails before any
generation cost is paid.

When the mode has no floor (``min_model_b == 0.0``) the precondition
is a no-op. When the mode has a floor and the backend's
``model_size_b`` is ``None``, the precondition fails loudly — the
operator opted into a model-size-sensitive mode without declaring the
backend's size, which is a configuration error that should surface
immediately rather than silently route around the gate.
"""

from __future__ import annotations


class BackendMinModelNotMetError(RuntimeError):
    """Raised when the configured backend cannot meet the mode's ``min_model_b`` (P3-BACK-09)."""

    def __init__(
        self,
        *,
        mode_id: str,
        backend_id: str,
        backend_model: str,
        required_b: float,
        actual_b: float | None,
    ) -> None:
        if actual_b is None:
            message = (
                f"mode {mode_id!r} requires a backend with model_size_b >= {required_b:g}B; "
                f"backend {backend_id!r} ({backend_model}) leaves model_size_b unset"
            )
        else:
            message = (
                f"mode {mode_id!r} requires model_size_b >= {required_b:g}B; "
                f"backend {backend_id!r} ({backend_model}) declares {actual_b:g}B"
            )
        super().__init__(message)
        self.mode_id = mode_id
        self.backend_id = backend_id
        self.backend_model = backend_model
        self.required_b = required_b
        self.actual_b = actual_b


def enforce_min_model_b(
    *,
    mode_id: str,
    required_b: float,
    backend_id: str,
    backend_model: str,
    declared_b: float | None,
) -> None:
    """Raise :class:`BackendMinModelNotMetError` when the precondition fails."""
    if required_b <= 0.0:
        return
    if declared_b is None or declared_b < required_b:
        raise BackendMinModelNotMetError(
            mode_id=mode_id,
            backend_id=backend_id,
            backend_model=backend_model,
            required_b=required_b,
            actual_b=declared_b,
        )


__all__ = ["BackendMinModelNotMetError", "enforce_min_model_b"]
