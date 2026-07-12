"""raggit watch command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging, get_logger
from raggit.db.session import AsyncSessionLocal

console = Console()
logger = get_logger("raggit.cli.watch")


def _apply_overrides(
    config: Any,
    *,
    poll_interval: int | None,
    log_level: str | None,
    tenant_id: str | None,
    tags: list[str] | None,
) -> None:
    """Apply CLI overrides onto the loaded RAGConfig."""
    if log_level is not None:
        config.log_level = log_level
    if poll_interval is not None:
        config.storage.poll_interval_seconds = poll_interval
    if tenant_id is not None:
        config.default_tenant_id = tenant_id
    if tags:
        config.default_tags = tags


def register_watch(app: typer.Typer) -> None:
    """Register the watch command with the CLI application."""

    @app.command()
    def watch(
        path: Path | None = typer.Argument(
            None,
            help="Directory to watch. Overrides configured storage URI for local storage.",
            exists=False,
        ),
        poll_interval: int | None = typer.Option(
            None, "--poll-interval", help="Override watcher poll interval in seconds"
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
        tenant: str | None = typer.Option(None, "--tenant", help="Default tenant id"),
        tag: list[str] = typer.Option(
            None, "--tag", help="Default tag for new documents (repeatable)"
        ),
    ) -> None:
        """Watch a directory for file changes and index continuously."""
        asyncio.run(_watch(path, poll_interval, log_level, tenant, tag))


async def _watch(
    path: Path | None,
    poll_interval: int | None,
    log_level: str | None,
    tenant: str | None,
    tags: list[str] | None,
) -> None:
    from raggit.core.audit import log_event
    from raggit.ingestion.indexer import Indexer
    from raggit.storage.base import FileAddedEvent, FileDeletedEvent, FileEvent, FileModifiedEvent
    from raggit.storage.factory import create_storage

    settings = get_settings()
    config = settings.rag_config
    _apply_overrides(
        config,
        poll_interval=poll_interval,
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

    storage = create_storage(storage_config)
    indexer = Indexer(storage, config)
    poll_interval_value = int(storage_config.poll_interval_seconds)

    console.print(
        Panel(
            f"[bold]Watching[/bold] {storage_config.uri}\nPoll interval: {poll_interval_value}s",
            title="raggit watch",
            border_style="blue",
        )
    )

    async with AsyncSessionLocal() as session, session.begin():
        await log_event(
            session,
            level="INFO",
            component="raggit.cli.watch",
            message="Watcher started",
            extra={
                "path": storage_config.uri,
                "poll_interval_seconds": poll_interval_value,
            },
        )
        await indexer.sync_all(session)

    async def on_event(event: FileEvent) -> None:
        async with AsyncSessionLocal() as session, session.begin():
            if isinstance(event, (FileAddedEvent, FileModifiedEvent)):
                console.print(f"[cyan]+[/cyan] {event.file.relative_path}")
                await indexer.index_file(session, event.file)
            elif isinstance(event, FileDeletedEvent):
                console.print(f"[red]-[/red] {event.file.relative_path}")
                await indexer.remove_file(session, event.file)

    try:
        await storage.watch(on_event, poll_interval_seconds=poll_interval_value)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
    finally:
        await indexer.close()
        await storage.close()
