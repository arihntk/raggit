"""raggit serve command."""

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

console = Console()
logger = get_logger("raggit.cli.serve")


def _apply_overrides(
    config: Any,
    *,
    path: Path | None,
    log_level: str | None,
    tenant_id: str | None,
    tags: list[str] | None,
) -> None:
    """Apply CLI overrides onto the loaded RAGConfig."""
    if log_level is not None:
        config.log_level = log_level
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


def register_serve(app: typer.Typer) -> None:
    """Register the serve command with the CLI application."""

    @app.command()
    def serve(
        path: Path | None = typer.Argument(
            None,
            help="Directory to watch. Overrides configured storage URI for local storage.",
            exists=False,
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
        tenant: str | None = typer.Option(None, "--tenant", help="Default tenant id"),
        tag: list[str] = typer.Option(
            None, "--tag", help="Default tag for new documents (repeatable)"
        ),
    ) -> None:
        """Run the long-running raggit service (watcher + future API)."""
        asyncio.run(_serve(path, log_level, tenant, tag))


async def _serve(
    path: Path | None,
    log_level: str | None,
    tenant: str | None,
    tags: list[str] | None,
) -> None:
    settings = get_settings()
    config = settings.rag_config
    _apply_overrides(
        config,
        path=path,
        log_level=log_level,
        tenant_id=tenant,
        tags=tags,
    )
    configure_logging(config.log_level)

    if config.storage is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    watcher = WatcherService(config)

    console.print(
        Panel(
            f"[bold]Serving[/bold] {config.storage.uri}\n"
            f"Storage: {config.storage.source_type.value}",
            title="raggit serve",
            border_style="blue",
        )
    )

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
        logger.exception("Watcher service failed", error=str(exc))
        console.print(f"[red]Watcher service failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        console.print("\n[yellow]Stopping watcher service...[/yellow]")
        await watcher.stop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
