"""raggit status command."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging
from raggit.db.session import AsyncSessionLocal

console = Console()


def register_status(app: typer.Typer) -> None:
    """Register the status command with the CLI application."""

    @app.command()
    def status(
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
    ) -> None:
        """Show indexing status."""
        asyncio.run(_status(log_level))


async def _status(log_level: str | None) -> None:
    from raggit.db.repository import DocumentRepository, EmbeddingCollectionRepository

    settings = get_settings()
    config = settings.rag_config
    if log_level is not None:
        config.log_level = log_level
    configure_logging(config.log_level)

    async with AsyncSessionLocal() as session:
        repo = DocumentRepository(session)
        docs = await repo.list_all()
        collections = await EmbeddingCollectionRepository(session).list_all()

    table = Table(title="Document Index Status")
    table.add_column("ID")
    table.add_column("Filename")
    table.add_column("Status")
    table.add_column("Tenant")
    table.add_column("Tags")
    table.add_column("Updated")
    for doc in docs:
        table.add_row(
            doc.id,
            doc.filename,
            doc.status.value,
            doc.tenant_id or "",
            ", ".join(doc.tags) if doc.tags else "",
            str(doc.updated_at),
        )
    console.print(table)
    console.print(f"Total documents: {len(docs)}")

    if collections:
        ctable = Table(title="Embedding Collections")
        ctable.add_column("Name")
        ctable.add_column("Model")
        ctable.add_column("Dim")
        ctable.add_column("Active")
        for coll in collections:
            ctable.add_row(
                coll.name,
                coll.embedding_model,
                str(coll.vector_size),
                "yes" if coll.is_active else "no",
            )
        console.print(ctable)
