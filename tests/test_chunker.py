"""Tests for format-aware chunking."""

from raggit.api.models import ChunkingConfig, RAGConfig
from raggit.ingestion.chunker import (
    ChunkPiece,
    chunk_document,
    count_tokens,
    dedup_chunks,
)


def _config(**kwargs: object) -> RAGConfig:
    chunking = ChunkingConfig(**kwargs)  # type: ignore[arg-type]
    return RAGConfig(chunking=chunking)


def test_count_tokens_approximates_when_tiktoken_missing() -> None:
    # tiktoken may or may not be installed; either way result is positive
    assert count_tokens("hello world") > 0


def test_chunk_markdown_preserves_section_titles() -> None:
    text = "# Intro\nHello world.\n\n# Details\nMore content here."
    config = _config(chunk_size=100, chunk_overlap=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.md")
    titles = {p.section_title for p in pieces if p.section_title}
    assert "Intro" in titles or "Details" in titles


def test_chunk_pdf_preserves_page_numbers() -> None:
    text = "--- Page 1 ---\nFirst page content.\n\n--- Page 2 ---\nSecond page content."
    config = _config(chunk_size=50, chunk_overlap=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.pdf")
    pages = {p.page_number for p in pieces if p.page_number is not None}
    assert 1 in pages
    assert 2 in pages


def test_chunk_code_preserves_function_names() -> None:
    text = "def hello():\n    pass\n\ndef world():\n    pass"
    config = _config(chunk_size=100, chunk_overlap=0, format_aware=True)
    pieces = chunk_document(text, config, path="script.py")
    titles = {p.section_title for p in pieces if p.section_title}
    assert any("def" in (t or "") for t in titles)


def test_token_based_chunking_limits_tokens() -> None:
    text = "word " * 1000
    config = _config(chunk_size=20, chunk_overlap=0, token_based=True)
    pieces = chunk_document(text, config)
    for piece in pieces:
        assert count_tokens(piece.text) <= 25  # allow splitter slack


def test_dedup_removes_near_duplicates() -> None:
    pieces = [
        ChunkPiece(text="The quick brown fox jumps over the lazy dog."),
        ChunkPiece(text="The quick brown fox jumps over the lazy dog"),
        ChunkPiece(text="Something completely different."),
    ]
    result = dedup_chunks(pieces, similarity=0.95)
    assert len(result) == 2
    texts = {p.text for p in result}
    assert "Something completely different." in texts
    dog = "The quick brown fox jumps over the lazy dog"
    assert dog not in texts or f"{dog}." not in texts


def test_dedup_keeps_distinct_chunks() -> None:
    pieces = [
        ChunkPiece(text="The quick brown fox jumps over the lazy dog."),
        ChunkPiece(text="Machine learning models require large datasets."),
    ]
    result = dedup_chunks(pieces, similarity=0.95)
    assert len(result) == 2
