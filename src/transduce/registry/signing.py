"""Signature status surface for mode packages (P2-PLG-06).

v0.5 ships the *surface* — operators see whether a package is signed
and by whom — without enforcing signature presence. ADR-0004 documents
why enforcement is deferred to v2: sigstore-python's verification API
is still maturing, and forcing default-on enforcement during that
window would break honest deployments. ``enforce_signing: true`` in
config opts a deployment in early; unsigned packages are then refused
unless ``unsigned_modes: allow`` is also set.

The actual sigstore verification call lazy-imports ``sigstore`` so
operators who do not need signature checks keep their dependency
footprint minimal. Production wiring builds a verifier from the
operator's trust root; v0.5 ships an in-memory stub the unit tests
exercise to lock the surface contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

SignatureVerifier = Callable[[str, str], bool]
"""Verifier callable returning True when the signature for the given
package identifier is valid for the named identity."""


class SignatureStatus(StrEnum):
    """Outcome of a signature check, surfaced via /v1/modes."""

    SIGNED = "signed"
    UNSIGNED = "unsigned"
    INVALID = "invalid"


@dataclass(frozen=True)
class SignatureRecord:
    """Per-package signature outcome attached to the registry."""

    package_name: str
    status: SignatureStatus
    signed_by: str | None


class SignatureEnforcementError(RuntimeError):
    """Raised when ``enforce_signing`` is true and a package is unsigned or invalid."""


def check_signature(
    *,
    package_name: str,
    expected_identity: str | None,
    verifier: SignatureVerifier | None,
) -> SignatureRecord:
    """Compute a signature record for a single package.

    When the operator has not declared an identity (``signed_by`` is
    None), the record is ``UNSIGNED``. When an identity is declared but
    the verifier callable is None (sigstore extra not installed),
    the record is also ``UNSIGNED`` so the surface is observable but
    not blocking. When a verifier is supplied and rejects the identity,
    the record is ``INVALID``.
    """
    if expected_identity is None:
        return SignatureRecord(
            package_name=package_name,
            status=SignatureStatus.UNSIGNED,
            signed_by=None,
        )
    if verifier is None:
        return SignatureRecord(
            package_name=package_name,
            status=SignatureStatus.UNSIGNED,
            signed_by=expected_identity,
        )
    is_valid = verifier(package_name, expected_identity)
    if is_valid:
        return SignatureRecord(
            package_name=package_name,
            status=SignatureStatus.SIGNED,
            signed_by=expected_identity,
        )
    return SignatureRecord(
        package_name=package_name,
        status=SignatureStatus.INVALID,
        signed_by=expected_identity,
    )


def enforce(record: SignatureRecord, *, enforce_signing: bool, allow_unsigned: bool) -> None:
    """Raise when enforcement is on and the record fails the policy.

    ``allow_unsigned=True`` opts an enforced deployment back into
    accepting unsigned packages with a startup warning (handled by the
    caller); ``INVALID`` records always raise regardless.
    """
    if record.status == SignatureStatus.INVALID:
        raise SignatureEnforcementError(
            f"package {record.package_name!r} signature is invalid for "
            f"declared identity {record.signed_by!r}"
        )
    if enforce_signing and record.status == SignatureStatus.UNSIGNED and not allow_unsigned:
        raise SignatureEnforcementError(
            f"package {record.package_name!r} is unsigned and "
            "modes.enforce_signing is true; set unsigned_modes: allow "
            "to opt this deployment back in"
        )


__all__ = [
    "SignatureEnforcementError",
    "SignatureRecord",
    "SignatureStatus",
    "SignatureVerifier",
    "check_signature",
    "enforce",
]
