"""Prompt-injection hardening for untrusted document text."""

from __future__ import annotations

import re

# Patterns that often try to override system instructions when embedded in docs.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions?\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(a|an|in)\b"),
    re.compile(r"(?i)\bsystem\s*:\s*"),
    re.compile(r"(?i)\bassistant\s*:\s*"),
    re.compile(r"(?i)\bnew\s+instructions?\s*:"),
    re.compile(r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(user|system)\b"),
    re.compile(r"(?i)\boverride\s+(the\s+)?system\s+prompt\b"),
    re.compile(r"(?i)<\s*/?\s*system\s*>"),
]


def harden_against_injection(text: str) -> str:
    """Neutralize common instruction-override patterns in retrieved/ingested text."""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered]", cleaned)
    return cleaned


def wrap_untrusted_context(text: str, source_label: str = "document") -> str:
    """Wrap untrusted content so models treat it as data, not instructions."""
    hardened = harden_against_injection(text)
    return (
        f'<untrusted_context source="{source_label}">\n'
        f"{hardened}\n"
        f"</untrusted_context>"
    )
