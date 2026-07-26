"""raggit chunks command."""

from __future__ import annotations

import asyncio
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging
from raggit.db.session import AsyncSessionLocal

console = Console()


def register_chunks(app: typer.Typer) -> None:
    """Register the chunks command with the CLI application."""

    @app.command()
    def chunks(
        document_id: str | None = typer.Argument(
            None,
            help="UUID of the document whose chunks to show.",
        ),
        filename: str | None = typer.Option(
            None, "--filename", help="Look up the document by filename instead of UUID."
        ),
        full_content: bool = typer.Option(
            False, "--full", help="Show the full chunk content instead of a preview."
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
    ) -> None:
        """List every chunk for a document."""
        if not document_id and not filename:
            console.print(
                "[red]Provide a document UUID or use --filename to select a document.[/red]"
            )
            raise typer.Exit(1)
        asyncio.run(_chunks(document_id, filename, full_content, log_level))


async def _chunks(
    document_id: str | None,
    filename: str | None,
    full_content: bool,
    log_level: str | None,
) -> None:
    from raggit.db.repository import ChunkRepository, DocumentRepository

    settings = get_settings()
    config = settings.rag_config
    if log_level is not None:
        config.log_level = log_level
    configure_logging(config.log_level)

    async with AsyncSessionLocal() as session:
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        doc = None
        if document_id:
            try:
                doc = await doc_repo.get_by_id(UUID(document_id))
            except ValueError as exc:
                console.print(f"[red]Invalid document UUID: {document_id}[/red]")
                raise typer.Exit(1) from exc

        if doc is None and filename:
            docs = await doc_repo.list_all()
            matches = [d for d in docs if d.filename == filename]
            if len(matches) == 1:
                doc = matches[0]
            elif len(matches) > 1:
                console.print(
                    f"[red]Multiple documents match filename '{filename}'. "
                    "Use the document UUID.[/red]"
                )
                raise typer.Exit(1)

        if doc is None:
            lookup = document_id or filename
            console.print(f"[red]Document not found: {lookup}[/red]")
            raise typer.Exit(1)

        chunks = await chunk_repo.get_by_document(UUID(doc.id))

    table = Table(title=f"Chunks for {doc.filename} ({len(chunks)} total)")
    table.add_column("Index", justify="right")
    table.add_column("Chunk ID")
    table.add_column("Section")
    table.add_column("Page", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Content")

    for chunk in chunks:
        content = chunk.cleaned_content
        if not full_content:
            content = content[:200] + ("..." if len(content) > 200 else "")
        table.add_row(
            str(chunk.chunk_index),
            str(chunk.id),
            chunk.section_title or "",
            str(chunk.page_number) if chunk.page_number is not None else "",
            str(chunk.word_count) if chunk.word_count is not None else "",
            content,
        )

    console.print(table)
