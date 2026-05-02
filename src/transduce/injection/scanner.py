"""Ingress prompt-injection scanner (P2-INJ-02, P2-INJ-03).

A regex-pack pre-generation scan over user input. Detects role-flip
phrases, system-prompt leak markers, "ignore the above" imperatives, and
common evasion phrasings. Defence-in-depth only: SECURITY.md states
explicitly that this is not a safety boundary against hostile input
authors. It is not a model judge; it is fast pattern-matching designed
for the documented v0.5 latency budget (<30 ms p99 on a 50 KB input).

Coverage targets per dev-plan P2-INJ-02 and the eval-suite v0.1
acceptance: detection >80% on the 100-prompt injection-attack corpus
with false-positive rate <5% on benign paraphrase inputs. Pattern
inventory is documented inline so additions can be reviewed against
the same eval gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

InjectionCategory = str


@dataclass(frozen=True)
class InjectionMatch:
    """Result of a positive scan."""

    category: InjectionCategory
    pattern: str
    span: str


@dataclass(frozen=True)
class _PatternRule:
    category: InjectionCategory
    pattern: re.Pattern[str]


def _rule(category: InjectionCategory, regex: str) -> _PatternRule:
    return _PatternRule(category=category, pattern=re.compile(regex, re.IGNORECASE))


_RULES: Final[tuple[_PatternRule, ...]] = (
    _rule(
        "ignore_previous_instructions",
        r"\b(?:please\s+)?ignore\s+(?:(?:all|any|the)\s+)?"
        r"(?:above|prior|previous|preceding|earlier)\s+"
        r"(?:instructions?|prompts?|directives?|rules?|guidelines?)\b",
    ),
    _rule(
        "ignore_previous_instructions",
        r"\bdisregard\s+(?:(?:all|any|the)\s+)?"
        r"(?:above|prior|previous|preceding|earlier)\s+"
        r"(?:instructions?|prompts?|directives?|rules?)\b",
    ),
    _rule(
        "role_flip",
        r"\byou\s+are\s+now\s+"
        r"(?:a|an|the)?\s*"
        r"(?:dan|do\s+anything\s+now|jailbroken|developer\s+mode|"
        r"unrestricted|uncensored|root|admin)\b",
    ),
    _rule(
        "role_flip",
        r"\bact\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?"
        r"(?:a|an|the)?\s*"
        r"(?:dan|jailbroken|admin|root|developer|"
        r"unrestricted|uncensored)\b",
    ),
    _rule(
        "role_flip",
        r"\bpretend\s+(?:that\s+)?you\s+(?:are|have)\s+"
        r"(?:no|been\s+given\s+no)\s+(?:rules|restrictions|limits|guidelines)\b",
    ),
    _rule(
        "system_prompt_leak",
        r"\b(?:reveal|print|show|output|display|share)\s+"
        r"(?:the|your)\s+"
        r"(?:system\s+)?(?:prompt|instructions?|rules?|directives?)\b",
    ),
    _rule(
        "system_prompt_leak",
        r"\bwhat\s+(?:are|were)\s+your\s+"
        r"(?:system\s+)?(?:instructions?|prompts?|directives?)\b",
    ),
    _rule(
        "fence_breakout",
        r"<<<(?:end|user_text|system|admin|override)_[A-Za-z0-9_]+>>>",
    ),
    _rule(
        "fence_breakout",
        r"</?(?:system|admin|developer|override)>",
    ),
    _rule(
        "exfiltration",
        r"\b(?:send|email|post|upload|exfiltrate|leak|transmit)\s+"
        r"(?:the\s+)?(?:secret|key|token|credential|password|api[_\s-]*key)\b",
    ),
)


class InjectionScanner:
    """Run the regex pack over an input and return the first match (if any)."""

    name = "regex-pack-v0.5"

    def scan(self, text: str) -> InjectionMatch | None:
        for rule in _RULES:
            match = rule.pattern.search(text)
            if match is not None:
                return InjectionMatch(
                    category=rule.category,
                    pattern=rule.pattern.pattern,
                    span=match.group(0),
                )
        return None


class InputInjectionDetectedError(RuntimeError):
    """Raised when the ingress scanner matches a known injection pattern."""

    def __init__(self, match: InjectionMatch) -> None:
        super().__init__(f"input injection detected: category={match.category} span={match.span!r}")
        self.match = match


__all__ = [
    "InjectionMatch",
    "InjectionScanner",
    "InputInjectionDetectedError",
]
