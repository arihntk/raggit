"""raggit CLI entry point."""

from __future__ import annotations

import typer

from raggit.cli.chunks_cmd import register_chunks
from raggit.cli.eval_cmd import register_eval
from raggit.cli.ingest_cmd import register_ingest
from raggit.cli.query_cmd import register_query
from raggit.cli.serve_cmd import register_serve
from raggit.cli.setup_cmd import register_setup
from raggit.cli.status_cmd import register_status
from raggit.cli.watch_cmd import register_watch

app = typer.Typer(
    name="raggit",
    help="Plug-and-play production-grade RAG system",
    no_args_is_help=True,
)

register_setup(app)
register_ingest(app)
register_serve(app)
register_watch(app)
register_query(app)
register_status(app)
register_chunks(app)
register_eval(app)

if __name__ == "__main__":
    app()
