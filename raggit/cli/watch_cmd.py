"""raggit watch command."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging, get_logger
from raggit.core.watcher import WatcherService
from raggit.storage.base import FileDeletedEvent, FileEvent

console = Console()
logger = get_logger("raggit.cli.watch")


def _apply_overrides(
    config: Any,
    *,
    path: Path | None,
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
    if path is not None:
        resolved_path = path.resolve()
        if config.storage.source_type.value == "local" and not resolved_path.exists():
            console.print(f"[red]Path does not exist: {resolved_path}[/red]")
            raise typer.Exit(1)
        config.storage.uri = str(resolved_path)


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
    settings = get_settings()
    config = settings.rag_config
    _apply_overrides(
        config,
        path=path,
        poll_interval=poll_interval,
        log_level=log_level,
        tenant_id=tenant,
        tags=tags,
    )
    configure_logging(config.log_level)

    if config.storage is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    poll_interval_value = int(config.storage.poll_interval_seconds)
    console.print(
        Panel(
            f"[bold]Watching[/bold] {config.storage.uri}\n"
            f"Poll interval: {poll_interval_value}s",
            title="raggit watch",
            border_style="blue",
        )
    )

    def _on_event(event: FileEvent) -> None:
        if isinstance(event, FileDeletedEvent):
            console.print(f"[red]-[/red] {event.file.relative_path}")
        else:
            console.print(f"[cyan]+[/cyan] {event.file.relative_path}")

    watcher = WatcherService(config, on_event=_on_event)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        await watcher.start()
        await stop_event.wait()
    except Exception as exc:
        logger.exception("Watcher failed", error=str(exc))
        console.print(f"[red]Watcher failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        await watcher.stop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
