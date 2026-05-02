"""Unit tests for the signature surface (P2-PLG-06)."""

from __future__ import annotations

import pytest

from transduce.registry.signing import (
    SignatureEnforcementError,
    SignatureStatus,
    check_signature,
    enforce,
)

pytestmark = pytest.mark.unit


def test_signature_unsigned_when_no_identity_declared() -> None:
    record = check_signature(
        package_name="x",
        expected_identity=None,
        verifier=None,
    )

    assert record.status is SignatureStatus.UNSIGNED
    assert record.signed_by is None


def test_signature_unsigned_when_verifier_unavailable() -> None:
    record = check_signature(
        package_name="x",
        expected_identity="release@determ-ai",
        verifier=None,
    )

    assert record.status is SignatureStatus.UNSIGNED
    assert record.signed_by == "release@determ-ai"


def test_signature_signed_when_verifier_returns_true() -> None:
    record = check_signature(
        package_name="x",
        expected_identity="release@determ-ai",
        verifier=lambda *_: True,
    )

    assert record.status is SignatureStatus.SIGNED


def test_signature_invalid_when_verifier_returns_false() -> None:
    record = check_signature(
        package_name="x",
        expected_identity="release@determ-ai",
        verifier=lambda *_: False,
    )

    assert record.status is SignatureStatus.INVALID


def test_signature_enforce_invalid_always_raises() -> None:
    record = check_signature(
        package_name="x",
        expected_identity="release@determ-ai",
        verifier=lambda *_: False,
    )

    with pytest.raises(SignatureEnforcementError, match="invalid"):
        enforce(record, enforce_signing=False, allow_unsigned=True)


def test_signature_enforce_unsigned_blocks_when_strict() -> None:
    record = check_signature(
        package_name="x",
        expected_identity=None,
        verifier=None,
    )

    with pytest.raises(SignatureEnforcementError, match="unsigned"):
        enforce(record, enforce_signing=True, allow_unsigned=False)


def test_signature_enforce_unsigned_allowed_with_explicit_flag() -> None:
    record = check_signature(
        package_name="x",
        expected_identity=None,
        verifier=None,
    )

    enforce(record, enforce_signing=True, allow_unsigned=True)


def test_signature_enforce_passes_signed_record() -> None:
    record = check_signature(
        package_name="x",
        expected_identity="release@determ-ai",
        verifier=lambda *_: True,
    )

    enforce(record, enforce_signing=True, allow_unsigned=False)
