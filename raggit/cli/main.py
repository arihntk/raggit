"""raggit CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from raggit.cli.chunks_cmd import register_chunks
from raggit.cli.eval_cmd import register_eval
from raggit.cli.ingest_cmd import register_ingest
from raggit.cli.mcp_cmd import register_mcp
from raggit.cli.query_cmd import register_query
from raggit.cli.serve_cmd import register_serve
from raggit.cli.setup_cmd import register_setup
from raggit.cli.status_cmd import register_status

app = typer.Typer(
    name="raggit",
    help="Plug-and-play production-grade RAG system",
    no_args_is_help=True,
)

register_setup(app)
register_ingest(app)
register_serve(app)
register_query(app)
register_status(app)
register_chunks(app)
register_eval(app)
register_mcp(app)

# -- Deprecated ``raggit watch`` shim ------------------------------------
# The watcher is now fully automatic via ``raggit serve`` (which runs the
# API + watcher together). We keep a hidden alias so existing scripts get a
# helpful migration message instead of a cryptic "No such command" error.
_console = Console()


@app.command("watch", hidden=True)
def _deprecated_watch() -> None:
    """Deprecated: use ``raggit serve`` instead."""
    _console.print(
        "[yellow]``raggit watch`` has been removed.[/yellow]\n"
        "The watcher now runs automatically inside ``raggit serve`` – you no "
        "longer need to run a separate watch command.\n"
        "\n"
        "  [bold]Instead run:[/bold]  raggit serve\n"
        "  [dim]API-only:[/dim]       raggit serve --no-watcher\n"
        "  [dim]Watcher-only via API:[/dim] curl -X POST http://localhost:8000/watcher/start\n"
    )
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
