"""raggit CLI entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from raggit.api.models import RAGConfig
from raggit.core.config import get_settings
from raggit.core.logging import configure_logging, get_logger
from raggit.db.session import AsyncSessionLocal
from raggit.ingestion.indexer import Indexer
from raggit.llm.augmenter import augment_and_answer
from raggit.llm.factory import create_llm
from raggit.retrieval.engine import RetrievalEngine
from raggit.storage.base import FileAddedEvent, FileDeletedEvent, FileModifiedEvent
from raggit.storage.factory import create_storage
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="raggit",
    help="Plug-and-play production-grade RAG system",
    no_args_is_help=True,
)
console = Console()
logger = get_logger("raggit.cli")


def _get_config() -> RAGConfig:
    """Load configuration."""
    return get_settings().rag_config


@app.command()
def setup(
    database_url: str = typer.Option(
        "postgresql+asyncpg://raggit:raggit@localhost:5433/raggit",
        help="PostgreSQL connection URL",
    ),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant URL"),
    storage_uri: str = typer.Option("./data/documents", help="Local document directory"),
    llm_provider: str = typer.Option("openai", help="LLM provider: openai or ollama"),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    llm_api_key: str | None = typer.Option(None, help="LLM API key"),
) -> None:
    """Interactive setup: write configuration to ~/.config/raggit/raggit.env."""
    from raggit.core.config import config_file_path

    config_path = config_file_path()
    env_lines = [
        f"DATABASE_URL={database_url}",
        f"QDRANT_URL={qdrant_url}",
        "QDRANT_COLLECTION=raggit_chunks",
        "STORAGE_SOURCE_TYPE=local",
        f"STORAGE_URI={storage_uri}",
        f"LLM_PROVIDER={llm_provider}",
        f"LLM_MODEL={llm_model}",
    ]
    if llm_api_key:
        env_lines.append(f"LLM_API_KEY={llm_api_key}")

    config_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    console.print(f"[green]Configuration written to {config_path}[/green]")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Directory or file to ingest", exists=True),  # noqa: B008
) -> None:
    """Run one-time ingestion over a path."""
    asyncio.run(_ingest(path))


async def _ingest(path: Path) -> None:
    config = _get_config()
    configure_logging(config.log_level)

    storage_config = config.storage
    if storage_config is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    storage_config.uri = str(path.resolve())
    storage = create_storage(storage_config)
    indexer = Indexer(storage, config)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await indexer.sync_all(session)
        await session.commit()

    await indexer.close()
    console.print("[green]Ingestion complete.[/green]")


@app.command()
def watch(
    path: Path | None = typer.Option(None, help="Directory to watch"),  # noqa: B008
) -> None:
    """Watch a directory for file changes and index continuously."""
    asyncio.run(_watch(path))


async def _watch(path: Path | None) -> None:
    config = _get_config()
    configure_logging(config.log_level)

    storage_config = config.storage
    if storage_config is None:
        console.print("[red]No storage configured. Run `raggit setup` first.[/red]")
        raise typer.Exit(1)

    if path:
        storage_config.uri = str(path.resolve())

    storage = create_storage(storage_config)
    indexer = Indexer(storage, config)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await indexer.sync_all(session)
        await session.commit()

    async def on_event(event):
        async with AsyncSessionLocal() as session:
            async with session.begin():
                if isinstance(event, (FileAddedEvent, FileModifiedEvent)):
                    await indexer.index_file(session, event.file)
                elif isinstance(event, FileDeletedEvent):
                    await indexer.remove_file(session, event.file)
            await session.commit()

    try:
        await storage.watch(on_event)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
    finally:
        await indexer.close()


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int | None = typer.Option(None, help="Override top-k"),
) -> None:
    """Ask a question against the indexed documents."""
    asyncio.run(_query(question, top_k))


async def _query(question: str, top_k: int | None) -> None:
    config = _get_config()
    configure_logging(config.log_level)

    from raggit.db.repository import ChunkRepository
    from raggit.db.vector import VectorStore
    from raggit.ingestion.embedder import create_embedder

    embedder = create_embedder(config.embedding)
    vector_store = VectorStore(config)

    async with AsyncSessionLocal() as session:
        chunk_repo = ChunkRepository(session)
        engine = RetrievalEngine(
            embedder=embedder,
            vector_store=vector_store,
            chunk_repo=chunk_repo,
            min_top_k=top_k if top_k is not None else config.min_top_k,
            max_top_k=top_k if top_k is not None else config.max_top_k,
            top_k_ratio=0.0 if top_k is not None else config.top_k_ratio,
            rrf_k=config.rrf_k,
        )

        result = await engine.retrieve(question)

        table = Table(title="Retrieved Chunks")
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Chunk")
        for rank, retrieved in enumerate(result.chunks, start=1):
            table.add_row(
                str(rank),
                f"{retrieved.score:.4f}",
                retrieved.chunk.cleaned_content[:300],
            )
        console.print(table)

        if config.llm.provider and config.llm.api_key:
            llm = create_llm(config.llm)
            answer = await augment_and_answer(llm, result)
            console.print("\n[bold cyan]Answer:[/bold cyan]")
            console.print(answer)
        else:
            console.print("\n[yellow]No LLM configured; showing retrieved chunks only.[/yellow]")

    await engine.close()


@app.command()
def status() -> None:
    """Show indexing status."""
    asyncio.run(_status())


async def _status() -> None:
    from raggit.db.repository import DocumentRepository

    async with AsyncSessionLocal() as session:
        repo = DocumentRepository(session)
        docs = await repo.list_all()

    table = Table(title="Document Index Status")
    table.add_column("Filename")
    table.add_column("Status")
    table.add_column("Updated")
    for doc in docs:
        table.add_row(doc.filename, doc.status.value, str(doc.updated_at))
    console.print(table)
    console.print(f"Total documents: {len(docs)}")


@app.command()
def tui() -> None:
    """Launch the Textual TUI dashboard."""
    from raggit.cli.tui import RAGgitApp

    app = RAGgitApp()
    app.run()


if __name__ == "__main__":
    app()
