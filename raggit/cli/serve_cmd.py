"""raggit serve command."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from raggit.api.server import run_api_server
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
        host: str = typer.Option("0.0.0.0", "--host", help="API server host"),
        port: int = typer.Option(8000, "--port", help="API server port"),
        no_watcher: bool = typer.Option(
            False, "--no-watcher", help="Do not start the storage watcher"
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
        tenant: str | None = typer.Option(None, "--tenant", help="Default tenant id"),
        tag: list[str] = typer.Option(
            None, "--tag", help="Default tag for new documents (repeatable)"
        ),
    ) -> None:
        """Run the long-running raggit service (FastAPI API + optional watcher)."""
        asyncio.run(_serve(path, host, port, no_watcher, log_level, tenant, tag))


async def _serve(
    path: Path | None,
    host: str,
    port: int,
    no_watcher: bool,
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

    watcher: WatcherService | None = None
    if not no_watcher:
        watcher = WatcherService(config)

    console.print(
        Panel(
            f"[bold]API[/bold] http://{host}:{port}\n"
            f"[bold]Storage[/bold] {config.storage.uri}\n"
            f"[bold]Watcher[/bold] {'disabled' if no_watcher else 'enabled'}",
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

    api_task = asyncio.create_task(run_api_server(host=host, port=port))
    watcher_task: asyncio.Task[None] | None = None

    async def _start_watcher() -> None:
        if watcher is None:
            return
        try:
            await watcher.start()
        except Exception:
            logger.exception("Watcher service failed during startup")
            stop_event.set()

    if watcher is not None:
        watcher_task = asyncio.create_task(_start_watcher())

    try:
        await stop_event.wait()
    except Exception as exc:
        logger.exception("Service failed", error=str(exc))
        console.print(f"[red]Service failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        console.print("\n[yellow]Stopping raggit service...[/yellow]")
        api_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await api_task
        if watcher_task is not None:
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task
        if watcher is not None:
            await watcher.stop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
