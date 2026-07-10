"""Format-aware, token-based text chunking with metadata and dedup."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from raggit.api.models import ChunkingConfig, RAGConfig


@dataclass
class ChunkPiece:
    """A chunk of text with structural metadata."""

    text: str
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    parent_chunk_index: int | None = None
    content_hash: str = ""
    token_count: int = 0


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken when available, else approximate."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # ~4 chars per token heuristic for English prose
        return max(1, (len(text) + 3) // 4)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _length_fn(token_based: bool) -> Callable[[str], int]:
    return count_tokens if token_based else len


def _splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=_length_fn(config.token_based),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _split_markdown_sections(text: str) -> list[tuple[str | None, str, int]]:
    """Split markdown into (section_title, body, start_offset) segments."""
    pattern = re.compile(r"(?m)^(#{1,6}\s+.+)$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text, 0)]

    sections: list[tuple[str | None, str, int]] = []
    # preamble before first heading
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

    # Explicit markers from some PDF extractors
    marker = re.compile(r"(?m)^---\s*Page\s+(\d+)\s*---\s*$")
    matches = list(marker.finditer(text))
    if not matches:
        return [(None, text, 0)]

    result = []
    if matches[0].start() > 0:
        result.append((None, text[: matches[0].start()], 0))
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append((page_num, text[start:end], match.start()))
    return result


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


def _pieces_from_segments(
    segments: list[tuple[str | None, str, int]],
    config: ChunkingConfig,
    page_number: int | None = None,
) -> list[ChunkPiece]:
    splitter = _splitter(config)
    pieces: list[ChunkPiece] = []
    for section_title, body, base_offset in segments:
        if not body.strip():
            continue
        parts = splitter.split_text(body)
        cursor = 0
        for part in parts:
            # locate part within body for offsets (best-effort)
            idx = body.find(part, cursor)
            if idx < 0:
                idx = cursor
            start = base_offset + idx
            end = start + len(part)
            cursor = idx + max(1, len(part) // 4)
            pieces.append(
                ChunkPiece(
                    text=part,
                    section_title=section_title,
                    page_number=page_number,
                    start_offset=start,
                    end_offset=end,
                    content_hash=_content_hash(part),
                    token_count=count_tokens(part),
                )
            )
    return pieces


def chunk_document(
    text: str,
    config: RAGConfig,
    path: str | None = None,
) -> list[ChunkPiece]:
    """Format-aware chunking that preserves section/page metadata."""
    chunking = config.chunking
    # Keep flat fields in sync for callers that only set chunk_size on RAGConfig
    if config.chunk_size != ChunkingConfig().chunk_size or config.chunk_overlap != 128:
        chunking = chunking.model_copy(
            update={
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            }
        )

    fmt = _detect_format(path) if chunking.format_aware else "text"
    pieces: list[ChunkPiece] = []

    if fmt == "markdown":
        sections = _split_markdown_sections(text)
        pieces = _pieces_from_segments(sections, chunking)
    elif fmt == "pdf":
        for page_num, page_text, base_offset in _split_pdf_pages(text):
            sections = [(None, page_text, base_offset)]
            page_pieces = _pieces_from_segments(sections, chunking, page_number=page_num)
            pieces.extend(page_pieces)
    elif fmt == "code":
        sections = _split_code_by_functions(text)
        pieces = _pieces_from_segments(sections, chunking)
    else:
        pieces = _pieces_from_segments([(None, text, 0)], chunking)

    # Assign parent indices: each chunk is its own parent by default.
    # Hierarchical retrieval expands windows at query time using parent_window.
    for i, piece in enumerate(pieces):
        piece.parent_chunk_index = i

    if chunking.dedup_enabled:
        pieces = dedup_chunks(pieces, similarity=chunking.dedup_similarity)

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
