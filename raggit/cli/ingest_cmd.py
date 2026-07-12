"""raggit ingest command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging, get_logger
from raggit.db.session import AsyncSessionLocal

app_ingest = typer.Typer()
console = Console()
logger = get_logger("raggit.cli.ingest")


def _apply_overrides(
    config: Any,
    *,
    chunk_size: int | None,
    chunk_overlap: int | None,
    preserve_sections: bool | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    log_level: str | None,
    tenant_id: str | None,
    tags: list[str] | None,
) -> None:
    """Apply CLI overrides onto the loaded RAGConfig."""
    if log_level is not None:
        config.log_level = log_level
    if chunk_size is not None:
        config.chunk_size = chunk_size
        config.chunking.max_words_per_chunk = chunk_size
    if chunk_overlap is not None:
        config.chunk_overlap = chunk_overlap
        config.chunking.chunk_overlap_words = chunk_overlap
    if preserve_sections is not None:
        config.chunking.preserve_sections = preserve_sections
    if embedding_provider is not None:
        config.embedding.provider = embedding_provider
    if embedding_model is not None:
        config.embedding.model = embedding_model
    if tenant_id is not None:
        config.default_tenant_id = tenant_id
    if tags:
        config.default_tags = tags


def register_ingest(app: typer.Typer) -> None:
    """Register the ingest command with the CLI application."""

    @app.command()
    def ingest(
        path: Path | None = typer.Argument(
            None,
            help="Directory or file to ingest. Required for local storage; optional for cloud.",
            exists=False,
        ),
        chunk_size: int | None = typer.Option(None, "--chunk-size", help="Override chunk size"),
        chunk_overlap: int | None = typer.Option(
            None, "--chunk-overlap", help="Override chunk overlap"
        ),
        preserve_sections: bool | None = typer.Option(
            None, "--preserve-sections/--split-sections", help="Override section preservation"
        ),
        embedding_provider: str | None = typer.Option(
            None, "--embedding-provider", help="Override embedding provider"
        ),
        embedding_model: str | None = typer.Option(
            None, "--embedding-model", help="Override embedding model"
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
        tenant: str | None = typer.Option(None, "--tenant", help="Default tenant id"),
        tag: list[str] = typer.Option(None, "--tag", help="Default tag (repeatable)"),
    ) -> None:
        """Run one-time ingestion over a path."""
        asyncio.run(
            _ingest(
                path,
                chunk_size,
                chunk_overlap,
                preserve_sections,
                embedding_provider,
                embedding_model,
                log_level,
                tenant,
                tag,
            )
        )


async def _ingest(
    path: Path | None,
    chunk_size: int | None,
    chunk_overlap: int | None,
    preserve_sections: bool | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    log_level: str | None,
    tenant: str | None,
    tags: list[str] | None,
) -> None:
    from raggit.core.audit import log_event
    from raggit.ingestion.indexer import Indexer
    from raggit.storage.factory import create_storage

    settings = get_settings()
    config = settings.rag_config
    _apply_overrides(
        config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        preserve_sections=preserve_sections,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        log_level=log_level,
        tenant_id=tenant,
        tags=tags,
    )
    configure_logging(config.log_level)

    storage_config = config.storage
    if storage_config is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    if path is not None:
        resolved_path = path.resolve()
        if storage_config.source_type.value == "local" and not resolved_path.exists():
            console.print(f"[red]Path does not exist: {resolved_path}[/red]")
            raise typer.Exit(1)
        storage_config.uri = str(resolved_path)
    elif storage_config.source_type.value == "local":
        console.print("[red]A path is required for local storage.[/red]")
        raise typer.Exit(1)

    storage = create_storage(storage_config)
    indexer = Indexer(storage, config)

    started_at = asyncio.get_event_loop().time()
    try:
        async with AsyncSessionLocal() as session, session.begin():
            await log_event(
                session,
                level="INFO",
                component="raggit.cli.ingest",
                message="Ingestion started",
                extra={
                    "path": str(path.resolve()) if path else storage_config.uri,
                    "storage_type": storage_config.source_type.value,
                },
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Indexing files...", total=None)

                def _update(file_obj: object, current: int, total: int) -> None:
                    if total > 0:
                        progress.update(task, total=total, completed=current)
                    sf = getattr(file_obj, "relative_path", str(file_obj))
                    progress.update(task, description=f"Indexing {sf}")

                await indexer.sync_all(session, progress_callback=_update)
                progress.update(task, description="Indexing complete", completed=1, total=1)

            await log_event(
                session,
                level="INFO",
                component="raggit.cli.ingest",
                message="Ingestion completed",
                extra={
                    "path": str(path.resolve()) if path else storage_config.uri,
                    "duration_seconds": asyncio.get_event_loop().time() - started_at,
                },
            )

        elapsed = asyncio.get_event_loop().time() - started_at
        console.print(f"[green]Ingestion complete in {elapsed:.2f}s.[/green]")
    except Exception as exc:
        logger.exception("Ingestion failed", error=str(exc))
        console.print(f"[red]Ingestion failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        await indexer.close()
        await storage.close()
