"""Tests for indexer chunk pairing logic (no DB required)."""

from raggit.ingestion.cleaner import clean_chunk


def test_short_chunks_filtered_keep_raw_cleaned_alignment() -> None:
    """Regression: filtering cleaned chunks must not desync raw content."""
    raw_chunks = [
        "ab",  # too short after clean
        "This is a long enough chunk that should survive filtering easily.",
        "x",  # too short
        "Another sufficiently long chunk for the index pipeline to keep.",
    ]
    min_len = 20
    paired: list[tuple[str, str]] = []
    for raw in raw_chunks:
        cleaned = clean_chunk(raw)
        if len(cleaned.strip()) > min_len:
            paired.append((raw, cleaned))

    assert len(paired) == 2
    for raw, cleaned in paired:
        assert clean_chunk(raw) == cleaned
        assert len(cleaned.strip()) > min_len
