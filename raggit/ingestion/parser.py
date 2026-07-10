"""Document parsers for supported file formats."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from raggit.core.logging import get_logger

logger = get_logger("raggit.ingestion.parser")


class ParseError(Exception):
    """Raised when a document cannot be parsed."""


class Parser(ABC):
    """Abstract document parser."""

    supported_extensions: set[str]

    @abstractmethod
    def parse(self, content: bytes, filename: str = "") -> str:
        """Parse document bytes into plain text."""


class TextParser(Parser):
    """Parser for plain text and markdown files."""

    supported_extensions = {".txt", ".md", ".markdown"}

    def parse(self, content: bytes, filename: str = "") -> str:
        """Decode bytes as UTF-8 text."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("utf-8", errors="replace")


class PDFParser(Parser):
    """Parser for PDF documents using PyMuPDF."""

    supported_extensions = {".pdf"}

    def parse(self, content: bytes, filename: str = "") -> str:
        """Extract text from PDF bytes."""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text())
            doc.close()
            return "\n\n".join(parts)
        except Exception as exc:
            msg = f"Failed to parse PDF {filename}: {exc}"
            raise ParseError(msg) from exc


class DOCXParser(Parser):
    """Parser for DOCX documents."""

    supported_extensions = {".docx", ".doc"}

    def parse(self, content: bytes, filename: str = "") -> str:
        """Extract text from DOCX bytes."""
        from io import BytesIO

        try:
            document = DocxDocument(BytesIO(content))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as exc:
            msg = f"Failed to parse DOCX {filename}: {exc}"
            raise ParseError(msg) from exc


class HTMLParser(Parser):
    """Parser for HTML documents."""

    supported_extensions = {".html", ".htm"}

    def parse(self, content: bytes, filename: str = "") -> str:
        """Extract visible text from HTML bytes."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")

        soup = BeautifulSoup(text, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        visible = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        visible = re.sub(r"\n\s*\n+", "\n\n", visible)
        return visible.strip()


class ParserRegistry:
    """Registry of document parsers."""

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self.register(TextParser())
        self.register(PDFParser())
        self.register(DOCXParser())
        self.register(HTMLParser())

    def register(self, parser: Parser) -> None:
        """Register a parser for its supported extensions."""
        for ext in parser.supported_extensions:
            self._parsers[ext.lower()] = parser

    def get_parser(self, path: str) -> Parser:
        """Return the parser for a given file path."""
        ext = Path(path).suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            msg = f"No parser available for extension '{ext}'"
            raise ParseError(msg)
        return parser

    def parse(self, content: bytes, path: str) -> str:
        """Parse content using the appropriate parser."""
        parser = self.get_parser(path)
        return parser.parse(content, filename=path)


# Global registry instance
registry = ParserRegistry()


def parse_document(content: bytes, path: str) -> str:
    """Convenience function to parse document bytes to text."""
    return registry.parse(content, path)
