"""Recursive text chunking."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from raggit.api.models import RAGConfig


def create_chunker(config: RAGConfig) -> RecursiveCharacterTextSplitter:
    """Create a recursive character text splitter from config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, config: RAGConfig) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = create_chunker(config)
    return splitter.split_text(text)
