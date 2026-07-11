"""Tests for format-aware, word-based chunking."""

from raggit.api.models import ChunkingConfig, RAGConfig
from raggit.ingestion.chunker import (
    ChunkPiece,
    chunk_document,
    count_words,
    dedup_chunks,
)


def _config(**kwargs: object) -> RAGConfig:
    chunking = ChunkingConfig(**kwargs)  # type: ignore[arg-type]
    return RAGConfig(chunking=chunking)


def test_count_words_counts_whitespace_separated_words() -> None:
    assert count_words("hello world") == 2
    assert count_words("") == 0
    assert count_words("one two three four five") == 5


def test_chunk_markdown_preserves_section_titles() -> None:
    text = "# Intro\nHello world.\n\n# Details\nMore content here."
    config = _config(max_words_per_chunk=100, chunk_overlap_words=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.md")
    titles = {p.section_title for p in pieces if p.section_title}
    assert "Intro" in titles or "Details" in titles


def test_chunk_pdf_preserves_page_numbers() -> None:
    text = "--- Page 1 ---\nFirst page content.\n\n--- Page 2 ---\nSecond page content."
    config = _config(max_words_per_chunk=50, chunk_overlap_words=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.pdf")
    pages = {p.page_number for p in pieces if p.page_number is not None}
    assert 1 in pages
    assert 2 in pages


def test_chunk_pdf_detects_numbered_headings() -> None:
    text = (
        "--- Page 1 ---\n"
        "3.2.2 Scaled Dot-Product Attention\n"
        "We scale the dot products by one over the square root of d_k.\n\n"
        "3.2.3 Multi-Head Attention\n"
        "Instead of performing a single attention function we use multiple heads."
    )
    config = _config(max_words_per_chunk=100, chunk_overlap_words=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.pdf")
    titles = {p.section_title for p in pieces if p.section_title}
    assert "3.2.2 Scaled Dot-Product Attention" in titles
    assert "3.2.3 Multi-Head Attention" in titles


def test_chunk_code_preserves_function_names() -> None:
    text = "def hello():\n    pass\n\ndef world():\n    pass"
    config = _config(max_words_per_chunk=100, chunk_overlap_words=0, format_aware=True)
    pieces = chunk_document(text, config, path="script.py")
    titles = {p.section_title for p in pieces if p.section_title}
    assert any("def" in (t or "") for t in titles)


def test_word_based_chunking_limits_words() -> None:
    text = "word " * 1000
    config = _config(max_words_per_chunk=20, chunk_overlap_words=0)
    pieces = chunk_document(text, config)
    for piece in pieces:
        assert count_words(piece.text) <= 20


def test_structural_unit_kept_when_under_limit() -> None:
    text = "# Section\n" + "word " * 50
    config = _config(max_words_per_chunk=1024, chunk_overlap_words=0, format_aware=True)
    pieces = chunk_document(text, config, path="doc.md")
    assert len(pieces) == 1
    assert pieces[0].section_title == "Section"


def test_recursive_split_respects_max_words() -> None:
    # A paragraph with 200 words, max 50 -> should split recursively.
    text = "Paragraph start. " + "word " * 200
    config = _config(max_words_per_chunk=50, chunk_overlap_words=0)
    pieces = chunk_document(text, config)
    assert len(pieces) > 1
    for piece in pieces:
        assert count_words(piece.text) <= 50


def test_chunks_linked_sequentially() -> None:
    text = "one two three. four five six. seven eight nine. ten eleven twelve."
    config = _config(max_words_per_chunk=3, chunk_overlap_words=0)
    pieces = chunk_document(text, config)
    assert len(pieces) >= 2
    for i, piece in enumerate(pieces):
        assert piece.parent_chunk_index == i
        assert piece.prev_chunk_index == (i - 1 if i > 0 else None)
        assert piece.next_chunk_index == (
            i + 1 if i < len(pieces) - 1 else None
        )


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
