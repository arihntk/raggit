"""Optional MCP (Model Context Protocol) server for raggit.

The MCP integration is installed as an optional extra:

    uv pip install 'raggit[mcp]'

This module can be imported safely when ``mcp`` is not installed; runtime
errors are raised only when MCP features are actually invoked.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging, get_logger

logger = get_logger("raggit.mcp_server")

# Lazily-loaded MCP types so the module imports without ``mcp`` installed.
_FastMCP: Any | None = None


def _load_mcp() -> Any:
    """Import and return the FastMCP class, raising a clear error if missing."""
    global _FastMCP
    if _FastMCP is None:
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:
            msg = (
                "The 'mcp' extra is required for MCP support. "
                "Install it with: uv pip install 'raggit[mcp]'"
            )
            raise RuntimeError(msg) from exc
        _FastMCP = FastMCP
    return _FastMCP


def _get_mcp_server() -> Any:
    """Return the singleton FastMCP server instance."""
    if not hasattr(_get_mcp_server, "_instance") or _get_mcp_server._instance is None:
        fast_mcp_cls = _load_mcp()
        _get_mcp_server._instance = fast_mcp_cls(
            "raggit",
            instructions=(
                "You are connected to raggit, a production-grade retrieval-augmented "
                "generation system. Use the available tools to query indexed documents, "
                "inspect status, manage configuration, trigger ingestion, and run evaluations."
            ),
        )
        _register_tools(_get_mcp_server._instance)
    return _get_mcp_server._instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_config() -> Any:
    """Return the current runtime configuration."""
    return get_settings().rag_config


def _json_dump(obj: Any) -> str:
    """Serialize an object to a compact JSON string."""
    return json.dumps(obj, default=str, indent=2)


def _parse_uuid(value: str) -> UUID:
    """Parse a string UUID or raise a ValueError."""
    return UUID(str(value))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _register_tools(mcp: Any) -> None:
    """Register all raggit tools on the MCP server."""

    @mcp.tool()
    async def query(
        question: str,
        top_k: int | None = None,
        tenant_id: str | None = None,
        tags: str | None = None,
        generate_answer: bool = True,
    ) -> str:
        """Ask a question against the indexed documents."""
        from raggit.api.models import QueryFilters
        from raggit.db.repository import (
            ChunkRepository,
            DocumentRepository,
            EmbeddingCollectionRepository,
        )
        from raggit.db.session import get_session
        from raggit.db.vector import VectorStore
        from raggit.ingestion.embedder import create_embedder
        from raggit.llm.augmenter import augment_and_answer
        from raggit.llm.factory import create_llm
        from raggit.retrieval.engine import RetrievalEngine

        config = _get_config()
        configure_logging(config.log_level)

        filters = QueryFilters()
        if tenant_id:
            filters.tenant_id = tenant_id
        if tags:
            filters.tags = [t.strip() for t in tags.split(",") if t.strip()]

        if top_k is not None:
            config.retrieval.min_top_k = top_k
            config.retrieval.max_top_k = top_k
            config.retrieval.top_k_ratio = 0.0

        embedder = create_embedder(config.embedding)
        vector_store = VectorStore(config)
        llm = None
        llm_ready = config.llm.provider == "ollama" or bool(config.llm.api_key)
        if generate_answer and config.llm.provider and llm_ready:
            llm = create_llm(config.llm)

        async with get_session() as session, session.begin():
            active = await EmbeddingCollectionRepository(session).get_active()
            if active is not None:
                vector_store.set_collection(active.name)

            engine = RetrievalEngine(
                embedder=embedder,
                vector_store=vector_store,
                chunk_repo=ChunkRepository(session),
                document_repo=DocumentRepository(session),
                config=config,
                llm=llm,
            )
            try:
                result = await engine.retrieve(question, filters=filters)
                if llm is not None:
                    result = await augment_and_answer(llm, result, safety=config.safety)
            finally:
                await engine.close()

            return _json_dump(
                {
                    "query": result.query,
                    "answer": result.answer,
                    "refused": result.refused,
                    "refusal_reason": result.refusal_reason,
                    "grounded": result.grounded,
                    "citations": [cite.model_dump(mode="json") for cite in result.citations],
                    "chunks": [
                        {
                            "chunk_id": str(r.chunk.id),
                            "document_id": str(r.chunk.document_id),
                            "filename": r.chunk.filename,
                            "chunk_index": r.chunk.chunk_index,
                            "score": r.score,
                            "excerpt": r.chunk.cleaned_content[:240],
                        }
                        for r in result.chunks
                    ],
                }
            )

    @mcp.tool()
    async def get_status() -> str:
        """Return overall system status: document counts and active collections."""
        from raggit.db.repository import DocumentRepository, EmbeddingCollectionRepository
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            docs = await DocumentRepository(session).list_all()
            collections = await EmbeddingCollectionRepository(session).list_all()

        return _json_dump(
            {
                "total_documents": len(docs),
                "documents": [
                    {
                        "id": str(d.id),
                        "filename": d.filename,
                        "status": d.status.value,
                        "tenant_id": d.tenant_id,
                        "tags": list(d.tags or []),
                        "updated_at": d.updated_at.isoformat(),
                    }
                    for d in docs
                ],
                "collections": [
                    {
                        "name": c.name,
                        "model": c.embedding_model,
                        "provider": c.embedding_provider,
                        "vector_size": c.vector_size,
                        "active": c.is_active,
                    }
                    for c in collections
                ],
            }
        )

    @mcp.tool()
    async def list_documents(
        status: str | None = None,
        tenant_id: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> str:
        """List indexed documents with optional filters."""
        from raggit.db.repository import DocumentRepository
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            docs = await DocumentRepository(session).list_all()

        if status:
            docs = [d for d in docs if d.status.value == status]
        if tenant_id:
            docs = [d for d in docs if d.tenant_id == tenant_id]
        if tag:
            tags = [t.strip() for t in tag.split(",") if t.strip()]
            docs = [d for d in docs if any(t in (d.tags or []) for t in tags)]

        return _json_dump(
            [
                {
                    "id": str(d.id),
                    "filename": d.filename,
                    "source_uri": d.source_uri,
                    "status": d.status.value,
                    "tenant_id": d.tenant_id,
                    "tags": list(d.tags or []),
                }
                for d in docs[:limit]
            ]
        )

    @mcp.tool()
    async def get_document(document_id: str) -> str:
        """Return a single document by ID."""
        from raggit.db.repository import DocumentRepository
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            doc = await DocumentRepository(session).get_by_id(_parse_uuid(document_id))
            if doc is None:
                return _json_dump({"error": "Document not found"})

            return _json_dump(
                {
                    "id": str(doc.id),
                    "filename": doc.filename,
                    "source_uri": doc.source_uri,
                    "source_type": doc.source_type.value,
                    "status": doc.status.value,
                    "content_hash": doc.content_hash,
                    "tenant_id": doc.tenant_id,
                    "tags": list(doc.tags or []),
                    "error_message": doc.error_message,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            )

    @mcp.tool()
    async def list_chunks(document_id: str, limit: int = 100) -> str:
        """List chunks for a document."""
        from raggit.db.repository import ChunkRepository, DocumentRepository
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            doc = await DocumentRepository(session).get_by_id(_parse_uuid(document_id))
            if doc is None:
                return _json_dump({"error": "Document not found"})

            chunks = await ChunkRepository(session).get_by_document(_parse_uuid(document_id))

        return _json_dump(
            [
                {
                    "id": str(c.id),
                    "chunk_index": c.chunk_index,
                    "word_count": c.word_count,
                    "section_title": c.section_title,
                    "page_number": c.page_number,
                    "content": c.cleaned_content[:500],
                }
                for c in chunks[:limit]
            ]
        )

    @mcp.tool()
    async def get_chunk(chunk_id: str) -> str:
        """Return a single chunk by ID."""
        from raggit.db.repository import ChunkRepository
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            chunk = await ChunkRepository(session).get_by_id(_parse_uuid(chunk_id))
            if chunk is None:
                return _json_dump({"error": "Chunk not found"})

            return _json_dump(
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count,
                    "section_title": chunk.section_title,
                    "page_number": chunk.page_number,
                    "content": chunk.cleaned_content,
                }
            )

    @mcp.tool()
    async def list_logs(
        level: str | None = None,
        component: str | None = None,
        limit: int = 100,
    ) -> str:
        """Return structured audit log entries."""
        from sqlalchemy import desc, select

        from raggit.db.models import LogModel
        from raggit.db.session import get_session

        config = _get_config()
        configure_logging(config.log_level)

        async with get_session() as session, session.begin():
            stmt = select(LogModel).order_by(desc(LogModel.created_at)).limit(limit)
            if level:
                stmt = stmt.where(LogModel.level == level.upper())
            if component:
                stmt = stmt.where(LogModel.component == component)
            result = await session.execute(stmt)
            logs = result.scalars().all()

        return _json_dump(
            [
                {
                    "id": str(log.id),
                    "level": log.level,
                    "component": log.component,
                    "message": log.message,
                    "extra": log.extra,
                    "created_at": log.created_at.isoformat(),
                }
                for log in logs
            ]
        )

    @mcp.tool()
    async def get_config() -> str:
        """Return the currently active runtime configuration."""
        configure_logging(_get_config().log_level)
        return _json_dump(_get_config().model_dump(mode="json"))

    @mcp.tool()
    async def ingest(
        path: str | None = None,
        tenant_id: str | None = None,
        tags: str | None = None,
    ) -> str:
        """Trigger a one-time ingestion run."""
        import asyncio
        from pathlib import Path

        from raggit.db.session import get_session
        from raggit.ingestion.indexer import Indexer
        from raggit.storage.factory import create_storage

        config = _get_config()
        configure_logging(config.log_level)

        if tenant_id:
            config.default_tenant_id = tenant_id
        if tags:
            config.default_tags = [t.strip() for t in tags.split(",") if t.strip()]

        storage_config = config.storage
        if storage_config is None:
            return _json_dump({"error": "No storage configured"})

        if path is not None:
            resolved = Path(path).expanduser().resolve()
            if storage_config.source_type.value == "local" and not resolved.exists():
                return _json_dump({"error": f"Path does not exist: {resolved}"})
            storage_config.uri = str(resolved)

        storage = create_storage(storage_config)
        indexer = Indexer(storage, config)

        started_at = asyncio.get_event_loop().time()
        try:
            async with get_session() as session, session.begin():
                await indexer.sync_all(session)
            elapsed = asyncio.get_event_loop().time() - started_at
            return _json_dump(
                {
                    "status": "completed",
                    "duration_seconds": elapsed,
                    "path": storage_config.uri,
                }
            )
        except Exception as exc:
            logger.exception("Ingestion failed", error=str(exc))
            return _json_dump({"error": f"Ingestion failed: {exc}"})
        finally:
            await indexer.close()
            await storage.close()

    @mcp.tool()
    async def run_eval(
        dataset_json: str | None = None,
        dataset_path: str | None = None,
    ) -> str:
        """Run an evaluation dataset and return the report summary."""
        from raggit.eval import EvalDataset, EvalRunner
        from raggit.eval.loader import load_dataset

        if dataset_path:
            dataset = load_dataset(dataset_path)
        elif dataset_json:
            dataset = EvalDataset(**json.loads(dataset_json))
        else:
            return _json_dump({"error": "Provide dataset_json or dataset_path"})

        config = _get_config()
        configure_logging(config.log_level)
        runner = EvalRunner(config)
        try:
            report = await runner.run(dataset)
        finally:
            await runner.close()

        return _json_dump(
            {
                "dataset_name": report.summary.dataset_name,
                "total_tests": report.summary.total_tests,
                "passed_tests": report.summary.passed_tests,
                "failed_tests": report.summary.failed_tests,
                "total_duration_ms": report.summary.total_duration_ms,
                "aggregates": [
                    {
                        "metric": agg.metric,
                        "mean": agg.mean,
                        "min": agg.min,
                        "max": agg.max,
                        "median": agg.median,
                    }
                    for agg in report.summary.aggregates
                ],
            }
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _check_mcp() -> None:
    """Raise a clear error if the mcp package is not installed."""
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        msg = (
            "The 'mcp' extra is required for MCP support. "
            "Install it with: uv pip install 'raggit[mcp]'"
        )
        raise RuntimeError(msg) from exc


async def run_stdio_server() -> None:
    """Run the MCP server over stdio (default for MCP clients)."""
    _check_mcp()
    configure_logging(_get_config().log_level)
    server = _get_mcp_server()
    await server.run_stdio_async()


async def run_sse_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run a standalone MCP SSE server."""
    import uvicorn

    _check_mcp()
    configure_logging(_get_config().log_level)
    app = _get_mcp_server().sse_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def get_sse_app() -> Any:
    """Return the SSE Starlette app for mounting in another ASGI server."""
    _check_mcp()
    configure_logging(_get_config().log_level)
    return _get_mcp_server().sse_app()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_stdio_server())
