"""Textual TUI dashboard for raggit."""

from __future__ import annotations

from raggit.core.config import get_settings
from raggit.db.session import AsyncSessionLocal
from raggit.ingestion.indexer import Indexer
from raggit.storage.factory import create_storage
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Header, Input, Label, Static


class RAGgitApp(App[None]):
    """Simple TUI for raggit."""

    CSS = """
    Screen { align: center middle; }
    #main { width: 80%; height: 80%; border: solid green; padding: 1 2; }
    #log { height: 1fr; border: solid yellow; padding: 1 2; }
    Input { margin: 1 0; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="main"):
            yield Label("raggit TUI", id="title")
            yield Button("Sync Documents", id="sync", variant="primary")
            yield Input(placeholder="Ask a question...", id="query")
            yield Static("Logs will appear here...", id="log")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sync":
            await self._sync()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query":
            await self._query(event.value)

    async def _sync(self) -> None:
        log = self.query_one("#log", Static)
        log.update("Syncing documents...")

        config = get_settings().rag_config
        if config.storage is None:
            log.update("No storage configured. Run `raggit setup` first.")
            return

        storage = create_storage(config.storage)
        indexer = Indexer(storage, config)

        async with AsyncSessionLocal() as session:
            async with session.begin():
                await indexer.sync_all(session)
            await session.commit()

        await indexer.close()
        log.update("Sync complete.")

    async def _query(self, question: str) -> None:
        log = self.query_one("#log", Static)
        log.update(f"Querying: {question}...")

        from raggit.db.repository import ChunkRepository
        from raggit.db.vector import VectorStore
        from raggit.ingestion.embedder import create_embedder
        from raggit.retrieval.engine import RetrievalEngine

        config = get_settings().rag_config
        embedder = create_embedder(config.embedding)
        vector_store = VectorStore(config)

        async with AsyncSessionLocal() as session:
            chunk_repo = ChunkRepository(session)
            engine = RetrievalEngine(
                embedder=embedder,
                vector_store=vector_store,
                chunk_repo=chunk_repo,
                min_top_k=config.min_top_k,
                max_top_k=config.max_top_k,
                top_k_ratio=config.top_k_ratio,
                rrf_k=config.rrf_k,
            )
            result = await engine.retrieve(question)
            lines = [f"Q: {result.query}", f"Keywords: {', '.join(result.sanitized_keywords)}"]
            for rank, retrieved in enumerate(result.chunks, start=1):
                lines.append(
                    f"{rank}. [score={retrieved.score:.4f}] {retrieved.chunk.cleaned_content[:200]}"
                )
            log.update("\n".join(lines))
            await engine.close()
