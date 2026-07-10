"""Tests for PII redaction."""

from raggit.ingestion.pii import redact_pii


def test_redact_email() -> None:
    text = "Contact me at user@example.com please."
    assert "[REDACTED_EMAIL]" in redact_pii(text)
    assert "user@example.com" not in redact_pii(text)


def test_redact_phone() -> None:
    text = "Call 555-123-4567 for details."
    assert "[REDACTED_PHONE]" in redact_pii(text)


def test_redact_ssn() -> None:
    text = "SSN 123-45-6789"
    assert "[REDACTED_SSN]" in redact_pii(text)


def test_redact_credit_card() -> None:
    text = "Card 4111 1111 1111 1111"
    assert "[REDACTED_CARD]" in redact_pii(text)


def test_redact_ip() -> None:
    text = "Server at 192.168.1.1"
    assert "[REDACTED_IP]" in redact_pii(text)
