"""Tests for chunk cleaning."""

from raggit.ingestion.cleaner import clean_chunk


def test_clean_chunk_collapses_whitespace() -> None:
    raw = "  hello   world  \n\n\n  foo  bar  "
    cleaned = clean_chunk(raw)
    assert cleaned == "hello world\n\nfoo bar"


def test_clean_chunk_fixes_hyphenation() -> None:
    raw = "implemen-\ntation details"
    cleaned = clean_chunk(raw)
    assert cleaned == "implementation details"
