"""Optional PII redaction for ingest and logging."""

from __future__ import annotations

import re

# Conservative patterns — redacts common PII shapes without heavy NLP deps.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\w)"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_pii(text: str) -> str:
    """Replace common PII patterns with placeholders."""
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    text = _SSN.sub("[REDACTED_SSN]", text)
    text = _CREDIT_CARD.sub("[REDACTED_CARD]", text)
    text = _PHONE.sub("[REDACTED_PHONE]", text)
    text = _IP.sub("[REDACTED_IP]", text)
    return text
