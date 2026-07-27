"""raggit mcp command.

Starts the optional MCP (Model Context Protocol) server for raggit. Requires the
``mcp`` extra to be installed:

    uv pip install 'raggit[mcp]'
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging

console = Console()


def _check_mcp() -> None:
    """Raise a clear error if the mcp package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        msg = (
            "The 'mcp' extra is required for MCP support. "
            "Install it with: uv pip install 'raggit[mcp]'"
        )
        console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1) from exc


def register_mcp(app: typer.Typer) -> None:
    """Register the mcp command with the CLI application."""

    @app.command()
    def mcp(
        transport: str = typer.Option(
            "stdio",
            "--transport",
            help="Transport protocol: stdio or sse",
        ),
        host: str = typer.Option("0.0.0.0", "--host", help="SSE server host"),
        port: int = typer.Option(8001, "--port", help="SSE server port"),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
    ) -> None:
        """Start the optional MCP server for raggit."""
        _check_mcp()
        settings = get_settings()
        config = settings.rag_config
        if log_level is not None:
            config.log_level = log_level
        configure_logging(config.log_level)

        if transport == "stdio":
            asyncio.run(_run_stdio())
        elif transport == "sse":
            asyncio.run(_run_sse(host, port))
        else:
            console.print(f"[red]Unknown transport: {transport}. Use stdio or sse.[/red]")
            raise typer.Exit(1)


async def _run_stdio() -> None:
    from raggit.mcp_server import run_stdio_server

    await run_stdio_server()


async def _run_sse(host: str, port: int) -> None:
    from raggit.mcp_server import run_sse_server

    await run_sse_server(host=host, port=port)
