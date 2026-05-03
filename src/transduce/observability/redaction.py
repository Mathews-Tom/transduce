"""Privacy-by-default text redaction for span attributes (P3-OBS-03).

Raw text and intermediate ``last_candidate`` strings are banned from
OTel span attributes by default; the helper here projects a string to a
``(sha256_8, length)`` pair so spans carry enough signal to correlate
without leaking content. The opt-in ``debug.include_text=true`` flag
re-enables raw text for non-prod debugging only and is gated by the
config validator (see ``ObservabilityConfig._debug_text_requires_redaction_off``).
"""

from __future__ import annotations

import hashlib

_SHA256_PREFIX_LEN: int = 8


def sha256_8(text: str) -> str:
    """Return the first 8 hex chars of the sha256 digest of ``text``.

    The 8-char prefix is the convention from
    ``docs/system-design.md`` §Observability and gives 32 bits of
    entropy — enough to disambiguate concurrent requests in a single
    operator's trace UI without shipping bytes to a collector.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_SHA256_PREFIX_LEN]
