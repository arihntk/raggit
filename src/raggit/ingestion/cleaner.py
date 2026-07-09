"""Chunk cleaning utilities."""

from __future__ import annotations

import re
import unicodedata


# Common patterns to clean up
def clean_chunk(text: str) -> str:
    """Normalize and clean a text chunk.

    Steps:
    1. Unicode NFKC normalization
    2. Strip leading/trailing whitespace
    3. Collapse whitespace
    4. Remove control characters except newlines
    5. Fix common hyphenation line breaks
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove control chars except newline/tab/space
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")

    # Fix hyphenation at line ends: "word-\nword" -> "wordword"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse multiple whitespace/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()
