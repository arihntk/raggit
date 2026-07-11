"""Format-aware, word-based text chunking with metadata and sequential links.

Chunking strategy:

1. Split the document into structural units (sections / pages / functions).
2. Keep a unit intact if it is <= max_words_per_chunk words.
3. If a unit is too large, recursively split it:
   section -> paragraphs -> sentences -> word windows.
4. Each produced piece knows its previous and next sibling index so retrieval
   can walk forward/backward through the document until relevance drops.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from raggit.api.models import ChunkingConfig, RAGConfig


@dataclass
class ChunkPiece:
    """A chunk of text with structural metadata and sibling links."""

    text: str
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    parent_chunk_index: int | None = None
    prev_chunk_index: int | None = None
    next_chunk_index: int | None = None
    content_hash: str = ""
    word_count: int = 0


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph structure."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Structural splitters
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[tuple[str, int]]:
    """Split text into paragraphs, returning (paragraph, start_offset)."""
    paragraphs: list[tuple[str, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        para = text[cursor:match.start()].strip()
        if para:
            paragraphs.append((para, cursor))
        cursor = match.end()
    trailing = text[cursor:].strip()
    if trailing:
        paragraphs.append((trailing, cursor))
    return paragraphs


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
)

# Common abbreviations that should not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "vs", "etc", "eg", "ie", "et al", "fig", "figs",
    "e.g", "i.e", "al", "vol", "vols", "pp", "pg",
    "u.s.a", "u.s", "u.k", "e.u", "a.m", "p.m",
}


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split text into sentences with abbreviation-aware heuristics."""
    text = _clean_text(text)
    if not text:
        return []

    parts: list[tuple[str, int]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        end = match.end()
        sentence = text[cursor:end].strip()
        if not sentence:
            cursor = end
            continue
        # Avoid splitting on abbreviations like "e.g. " or "U.S.A. "
        last_word = sentence.rstrip(".!? ").split()[-1].lower().rstrip(".")
        if last_word in _ABBREVIATIONS:
            continue
        parts.append((sentence, cursor))
        cursor = end

    trailing = text[cursor:].strip()
    if trailing:
        parts.append((trailing, cursor))
    return parts


def _split_word_windows(
    text: str,
    max_words: int,
    overlap_words: int,
    base_offset: int = 0,
) -> list[tuple[str, int]]:
    """Split a long sentence into overlapping word windows."""
    words = text.split()
    if len(words) <= max_words:
        return [(text, base_offset)]

    step = max(1, max_words - overlap_words)
    windows: list[tuple[str, int]] = []
    for i in range(0, len(words), step):
        window_words = words[i : i + max_words]
        window_text = " ".join(window_words)
        # Best-effort offset within the original text.
        start = base_offset + text.find(window_text)
        if start < base_offset:
            start = base_offset
        windows.append((window_text, start))
        if i + max_words >= len(words):
            break
    return windows


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


# Numbered heading: "3.2.2 Scaled Dot-Product Attention"
_NUMBERED_HEADING_RE = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z0-9\s\-,:;/()]{2,120}$"
)

# Standalone short title line in title case or all caps, followed by body.
_TITLE_LINE_RE = re.compile(
    r"(?m)^\s*([A-Z][A-Za-z0-9\s\-,:;/()]{2,80}|[A-Z\s\-,:;/()]{3,80})\s*$"
)


def _detect_headings(text: str) -> list[tuple[str | None, str, int]]:
    """Detect section headings heuristically.

    Returns segments as (section_title, body, start_offset).
    """
    # Prefer numbered headings; fall back to title-line headings.
    matches = list(_NUMBERED_HEADING_RE.finditer(text))
    if not matches:
        matches = list(_TITLE_LINE_RE.finditer(text))

    if not matches:
        return [(None, text, 0)]

    segments: list[tuple[str | None, str, int]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segments.append((None, preamble, 0))

    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        segments.append((title, body, match.start()))

    return segments


def _split_markdown_sections(text: str) -> list[tuple[str | None, str, int]]:
    """Split markdown into (section_title, body, start_offset) segments."""
    pattern = re.compile(r"(?m)^(#{1,6}\s+.+)$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text, 0)]

    sections: list[tuple[str | None, str, int]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()]
        if preamble.strip():
            sections.append((None, preamble, 0))

    for i, match in enumerate(matches):
        title = match.group(1).lstrip("#").strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append((title, body, match.start()))
    return sections


def _split_pdf_pages(text: str) -> list[tuple[int | None, str, int]]:
    """Split text that uses form-feed or page markers into pages."""
    if "\f" in text:
        pages = text.split("\f")
        offset = 0
        result: list[tuple[int | None, str, int]] = []
        for i, page in enumerate(pages, start=1):
            result.append((i, page, offset))
            offset += len(page) + 1
        return result

    marker = re.compile(r"(?m)^---\s*Page\s+(\d+)\s*---\s*$")
    matches = list(marker.finditer(text))
    if not matches:
        return [(None, text, 0)]

    page_result: list[tuple[int | None, str, int]] = []
    if matches[0].start() > 0:
        page_result.append((None, text[: matches[0].start()], 0))
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_result.append((page_num, text[start:end], match.start()))
    return page_result


def _split_code_by_functions(text: str) -> list[tuple[str | None, str, int]]:
    """Heuristic split of source code on top-level def/class/function."""
    pattern = re.compile(
        r"(?m)^(?:def |class |async def |function |export (?:async )?function |export class )"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text, 0)]

    segments: list[tuple[str | None, str, int]] = []
    if matches[0].start() > 0:
        head = text[: matches[0].start()]
        if head.strip():
            segments.append((None, head, 0))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        first_line = body.splitlines()[0] if body.splitlines() else ""
        title = first_line.strip()[:200] or None
        segments.append((title, body, start))
    return segments


# ---------------------------------------------------------------------------
# Format detection and recursive chunking
# ---------------------------------------------------------------------------


def _detect_format(path: str | None) -> str:
    if not path:
        return "text"
    ext = Path(path).suffix.lower()
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext == ".pdf":
        return "pdf"
    if ext in {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rb",
        ".php",
    }:
        return "code"
    return "text"


def _recursive_split_unit(
    text: str,
    unit: str,
    max_words: int,
    overlap_words: int,
    base_offset: int = 0,
) -> list[tuple[str, int]]:
    """Recursively split text into pieces no larger than max_words.

    Order of recursion:
        section -> paragraph -> sentence -> word_window
    """
    text = text.strip()
    if not text:
        return []

    if count_words(text) <= max_words:
        return [(text, base_offset)]

    if unit == "section":
        parts = _split_paragraphs(text)
        if len(parts) <= 1:
            # Cannot split by paragraph; go straight to sentences.
            return _recursive_split_unit(
                text, "paragraph", max_words, overlap_words, base_offset
            )
        result: list[tuple[str, int]] = []
        for part, offset in parts:
            result.extend(
                _recursive_split_unit(part, "paragraph", max_words, overlap_words, offset)
            )
        return result

    if unit == "paragraph":
        parts = _split_sentences(text)
        if len(parts) <= 1:
            # Cannot split by sentence; go to word windows.
            return _recursive_split_unit(
                text, "sentence", max_words, overlap_words, base_offset
            )
        result = []
        for part, offset in parts:
            result.extend(
                _recursive_split_unit(part, "sentence", max_words, overlap_words, offset)
            )
        return result

    # Final fallback: word windows.
    return _split_word_windows(text, max_words, overlap_words, base_offset)


def _pieces_from_segments(
    segments: list[tuple[str | None, str, int]],
    config: ChunkingConfig,
    page_number: int | None = None,
) -> list[ChunkPiece]:
    """Convert structural segments into ChunkPieces with recursive splitting."""
    max_words = config.max_words_per_chunk
    overlap_words = config.chunk_overlap_words
    pieces: list[ChunkPiece] = []

    for section_title, body, base_offset in segments:
        if not body.strip():
            continue

        # Try to keep the whole section if it fits.
        parts: list[tuple[str, int]]
        if count_words(body) <= max_words:
            parts = [(body, base_offset)]
        else:
            parts = _recursive_split_unit(
                body, "section", max_words, overlap_words, base_offset
            )

        for part_text, part_offset in parts:
            piece = ChunkPiece(
                text=part_text,
                section_title=section_title,
                page_number=page_number,
                start_offset=part_offset,
                end_offset=part_offset + len(part_text),
                content_hash=_content_hash(part_text),
                word_count=count_words(part_text),
            )
            pieces.append(piece)

    return pieces


def chunk_document(
    text: str,
    config: RAGConfig,
    path: str | None = None,
) -> list[ChunkPiece]:
    """Format-aware chunking that preserves section/page metadata."""
    chunking = config.chunking
    # Keep flat fields in sync for callers that only set chunk_size on RAGConfig.
    default_chunking = ChunkingConfig()
    if (
        config.chunk_size != default_chunking.max_words_per_chunk
        or config.chunk_overlap != default_chunking.chunk_overlap_words
    ):
        chunking = chunking.model_copy(
            update={
                "max_words_per_chunk": config.chunk_size,
                "chunk_overlap_words": config.chunk_overlap,
            }
        )

    fmt = _detect_format(path) if chunking.format_aware else "text"
    pieces: list[ChunkPiece] = []

    if fmt == "markdown":
        sections = _split_markdown_sections(text)
        pieces = _pieces_from_segments(sections, chunking)
    elif fmt == "pdf":
        for page_num, page_text, base_offset in _split_pdf_pages(text):
            # First try to detect headings within the page.
            sub_sections = _detect_headings(page_text)
            if len(sub_sections) == 1 and sub_sections[0][0] is None:
                # No headings detected; treat the page as one section.
                sub_sections = [(None, page_text, base_offset)]
            page_pieces = _pieces_from_segments(
                sub_sections, chunking, page_number=page_num
            )
            pieces.extend(page_pieces)
    elif fmt == "code":
        sections = _split_code_by_functions(text)
        pieces = _pieces_from_segments(sections, chunking)
    else:
        sections = _detect_headings(text)
        if len(sections) == 1 and sections[0][0] is None:
            sections = [(None, text, 0)]
        pieces = _pieces_from_segments(sections, chunking)

    # Assign sequential indices and sibling links.
    for i, piece in enumerate(pieces):
        piece.parent_chunk_index = i
        piece.prev_chunk_index = i - 1 if i > 0 else None
        piece.next_chunk_index = i + 1 if i < len(pieces) - 1 else None

    if chunking.dedup_enabled:
        pieces = dedup_chunks(pieces, similarity=chunking.dedup_similarity)
        # Re-link after dedup.
        for i, piece in enumerate(pieces):
            piece.parent_chunk_index = i
            piece.prev_chunk_index = i - 1 if i > 0 else None
            piece.next_chunk_index = i + 1 if i < len(pieces) - 1 else None

    return pieces


def chunk_text(text: str, config: RAGConfig, path: str | None = None) -> list[str]:
    """Back-compat: return plain text chunks."""
    return [p.text for p in chunk_document(text, config, path=path)]


def dedup_chunks(
    pieces: list[ChunkPiece],
    similarity: float = 0.92,
) -> list[ChunkPiece]:
    """Drop near-identical chunks using Jaccard on word sets + exact hash."""
    if not pieces:
        return pieces

    seen_hashes: set[str] = set()
    kept: list[ChunkPiece] = []
    kept_word_sets: list[set[str]] = []

    for piece in pieces:
        if piece.content_hash and piece.content_hash in seen_hashes:
            continue
        words = set(re.findall(r"[a-z0-9]+", piece.text.lower()))
        is_dup = False
        for prev in kept_word_sets:
            if not words or not prev:
                continue
            jaccard = len(words & prev) / len(words | prev)
            if jaccard >= similarity:
                is_dup = True
                break
        if is_dup:
            continue
        if piece.content_hash:
            seen_hashes.add(piece.content_hash)
        kept.append(piece)
        kept_word_sets.append(words)
    return kept
