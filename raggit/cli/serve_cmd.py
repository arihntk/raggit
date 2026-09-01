"""raggit serve – single canonical long-running service.

Rewritten from scratch: ``raggit serve`` is the **only** way to run the
watcher continuously. It starts the FastAPI HTTP API and the document
watcher together, supervising both so they run independently and
automatically without requiring manual ``raggit watch`` invocations.

Key properties
~~~~~~~~~~~~~~
- Automatic: watcher starts by default with the API; no manual second
  command needed. In Docker the container runs ``raggit serve`` and both
  services are up immediately.
- Independent: watcher runs as a background ``WatcherService`` task; the
  API and watcher supervise each other – if either crashes the process
  shuts down cleanly.
- Robust: handles SIGINT/SIGTERM on Unix and Windows, debounces rapid
  file events, waits for file stability before indexing, and cleans up
  observer threads on exit.
- Single implementation: all watcher logic lives in
  ``raggit.core.watcher.WatcherService`` – this module only wires CLI
  options, signals, and the API server. The legacy ``raggit watch``
  command has been removed in favor of this unified entrypoint.
"""

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
from raggit.storage.base import FileDeletedEvent, FileEvent

console = Console()
logger = get_logger("raggit.cli.serve")


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
        path_str = str(path)
        if config.storage.source_type.value == "local":
            resolved_path = path.resolve()
            if not resolved_path.exists():
                console.print(f"[red]Path does not exist: {resolved_path}[/red]")
                raise typer.Exit(1)
            config.storage.uri = str(resolved_path)
        else:
            # Cloud storage: handle cloud URIs without local Path resolution
            from raggit.storage.factory import _apply_cloud_uri_to_config, _is_cloud_uri

            if _is_cloud_uri(path_str):
                _apply_cloud_uri_to_config(path_str, config.storage)
            else:
                # Plain prefix override for cloud
                config.storage.prefix = path_str.strip("/")
                config.storage.uri = path_str


def register_serve(app: typer.Typer) -> None:
    """Register the ``serve`` command with the CLI application."""

    @app.command()
    def serve(
        path: Path | None = typer.Argument(
            None,
            help="Directory to watch. Overrides configured storage URI for local storage.",
            exists=False,
        ),
        host: str = typer.Option("0.0.0.0", "--host", help="API server host"),
        port: int = typer.Option(8000, "--port", help="API server port"),
        poll_interval: int | None = typer.Option(
            None, "--poll-interval", help="Override watcher poll interval in seconds"
        ),
        no_watcher: bool = typer.Option(
            False, "--no-watcher", help="Do not start the storage watcher (API only)"
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
        tenant: str | None = typer.Option(None, "--tenant", help="Default tenant id"),
        tag: list[str] = typer.Option(
            None, "--tag", help="Default tag for new documents (repeatable)"
        ),
    ) -> None:
        """Run the long-running raggit service (FastAPI API + watcher).

        The watcher starts automatically – you do not need to run a separate
        ``raggit watch`` command. Use ``--no-watcher`` only if you want an
        API-only deployment.
        """
        asyncio.run(_serve(path, host, port, poll_interval, no_watcher, log_level, tenant, tag))


async def _serve(
    path: Path | None,
    host: str,
    port: int,
    poll_interval: int | None,
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
        poll_interval=poll_interval,
        log_level=log_level,
        tenant_id=tenant,
        tags=tags,
    )
    configure_logging(config.log_level)

    if config.storage is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    # ------------------------------------------------------------------
    # Watcher – single canonical instance. It will also be exposed via
    # ``raggit.api.server._watcher`` so /watcher/status reflects the real
    # state even though the lifecycle is owned here.
    # The watcher is automatic – no manual ``raggit watch`` needed.
    # ------------------------------------------------------------------
    import raggit.api.server as api_server  # noqa: WPS433

    if no_watcher:
        api_server._watcher_auto_start_disabled = True
    else:
        api_server._watcher_auto_start_disabled = False

    watcher: WatcherService | None = None
    if not no_watcher:

        def _on_event(event: FileEvent) -> None:
            # Live indicator for terminal users (mirrors old `raggit watch` UX).
            try:
                if isinstance(event, FileDeletedEvent):
                    console.print(f"[red]-[/red] {event.file.relative_path}")
                else:
                    console.print(f"[cyan]+[/cyan] {event.file.relative_path}")
            except Exception:
                pass

        watcher = WatcherService(config, on_event=_on_event)
        # Publish early so the API lifespan sees it and skips auto-start.
        api_server._watcher = watcher

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
    installed_sigs: list[int] = []

    def _signal_handler() -> None:
        if not stop_event.is_set():
            stop_event.set()

    # Install signal handlers – works on Unix; on Windows we fall back to
    # signal.signal which is sufficient for Ctrl+C.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
            installed_sigs.append(sig)
        except (NotImplementedError, ValueError, RuntimeError):
            try:
                signal.signal(sig, lambda *_: _signal_handler())  # type: ignore[arg-type]
                installed_sigs.append(sig)
            except Exception:
                pass

    api_task: asyncio.Task[None] | None = None

    try:
        # Start watcher first so the initial sync completes before we claim readiness.
        if watcher is not None:
            try:
                await watcher.start()
            except Exception:
                logger.exception("Watcher failed to start")
                console.print("[red]Watcher failed to start – see logs for details[/red]")
                # Clear the premature publish so lifespan can retry or report correctly.
                try:
                    import raggit.api.server as _srv

                    if getattr(_srv, "_watcher", None) is watcher:
                        _srv._watcher = None
                except Exception:
                    pass
                raise typer.Exit(1) from None

            # If the background watch loop crashes, shut the whole service down.
            if watcher._watch_task is not None:

                def _watcher_done_cb(task: asyncio.Task[None]) -> None:
                    if task.cancelled():
                        return
                    exc = task.exception()
                    if exc is not None:
                        logger.error("Watcher background task crashed", error=str(exc))
                        if not stop_event.is_set():
                            stop_event.set()

                watcher._watch_task.add_done_callback(_watcher_done_cb)

        # Start FastAPI server as a background task. ``run_api_server`` blocks
        # until the server exits, so we run it in a task and wait on stop_event.
        api_task = asyncio.create_task(run_api_server(host=host, port=port), name="raggit-api")

        def _api_done_cb(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error("API server crashed", error=str(exc))
            else:
                logger.info("API server stopped")
            if not stop_event.is_set():
                stop_event.set()

        api_task.add_done_callback(_api_done_cb)

        # Block until a signal or a supervised task exits. This also covers
        # the case where uvicorn handles SIGINT via signal.signal on Windows
        # and our loop handler was overwritten – api_task will complete and
        # its done callback will set stop_event.
        stop_wait_task = asyncio.create_task(stop_event.wait(), name="raggit-stop-wait")
        done, pending = await asyncio.wait(
            [stop_wait_task, api_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel the waiter that didn't win.
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # If api_task completed first (e.g. via uvicorn signal handling),
        # ensure stop_event is set so the finally block runs correctly.
        if api_task in done and not stop_event.is_set():
            stop_event.set()

    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Service failed", error=str(exc))
        console.print(f"[red]Service failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    finally:
        console.print("\n[yellow]Stopping raggit service...[/yellow]")

        if api_task is not None:
            api_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await api_task

        if watcher is not None:
            with contextlib.suppress(Exception):
                await watcher.stop()
            try:
                import raggit.api.server as api_server

                if getattr(api_server, "_watcher", None) is watcher:
                    api_server._watcher = None
            except Exception:
                pass

        for sig in installed_sigs:
            with contextlib.suppress(Exception):
                try:
                    loop.remove_signal_handler(sig)  # type: ignore[arg-type]
                except Exception:
                    with contextlib.suppress(Exception):
                        signal.signal(sig, signal.SIG_DFL)

        console.print("[green]raggit service stopped[/green]")
