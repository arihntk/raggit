"""FastAPI HTTP API for raggit.

Exposes endpoints for querying, managing documents and chunks, inspecting logs,
updating configuration, triggering ingestion, and controlling the watcher.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, select

from raggit.api.deps import SessionDep
from raggit.api.models import (
    Chunk,
    ChunkingConfig,
    Document,
    EmbeddingConfig,
    LLMConfig,
    QueryFilters,
    QueryResult,
    QueryRewriteMode,
    RAGConfig,
    RetrievalConfig,
    SafetyConfig,
    SourceType,
    StorageConfig,
)
from raggit.core.audit import log_event
from raggit.core.config import config_file_path, get_settings
from raggit.core.logging import configure_logging, get_logger
from raggit.core.watcher import WatcherService
from raggit.db.models import ChunkModel, DocumentModel, LogModel
from raggit.db.repository import ChunkRepository, DocumentRepository, EmbeddingCollectionRepository
from raggit.db.session import AsyncSessionLocal, reset_engine
from raggit.db.vector import VectorStore
from raggit.ingestion.embedder import create_embedder
from raggit.ingestion.indexer import Indexer
from raggit.llm.augmenter import augment_and_answer
from raggit.llm.factory import create_llm
from raggit.retrieval.engine import RetrievalEngine
from raggit.storage.factory import create_storage

logger = get_logger("raggit.api.server")

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class _PartialRAGConfig(BaseModel):
    """Subset of RAGConfig fields that can be updated at runtime.

    Nested models are optional so callers can update only the knobs they care
    about without resending the entire configuration.
    """

    model_config = ConfigDict(extra="forbid")

    database_url: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    qdrant_api_key: str | None = None
    log_level: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    min_top_k: int | None = None
    max_top_k: int | None = None
    top_k_ratio: float | None = None
    rrf_k: int | None = None
    chunking: ChunkingConfig | None = None
    retrieval: RetrievalConfig | None = None
    safety: SafetyConfig | None = None
    embedding: EmbeddingConfig | None = None
    llm: LLMConfig | None = None
    storage: StorageConfig | None = None
    default_tenant_id: str | None = None
    default_tags: list[str] | None = None


class ConfigUpdateRequest(BaseModel):
    """Request body for updating the active configuration."""

    config: _PartialRAGConfig


class ConfigResponse(BaseModel):
    """Current runtime configuration."""

    config: RAGConfig
    config_file: str


class QueryRequest(BaseModel):
    """Request body for a retrieval query."""

    query: str
    filters: QueryFilters = Field(default_factory=QueryFilters)
    top_k: int | None = None
    min_top_k: int | None = None
    max_top_k: int | None = None
    top_k_ratio: float | None = None
    rrf_k: int | None = None
    min_score: float | None = None
    parent_window: int | None = None
    rewrite: QueryRewriteMode = QueryRewriteMode.NONE
    multi_query_count: int | None = None
    reranker_enabled: bool | None = None
    reranker_model: str | None = None
    reranker_top_n: int | None = None
    generate_answer: bool = True
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Request body for a one-time ingestion run."""

    path: str | None = None
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    preserve_sections: bool | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None


class WatcherStartRequest(BaseModel):
    """Request body for starting the watcher."""

    path: str | None = None
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    poll_interval_seconds: int | None = None


class WatcherStatusResponse(BaseModel):
    """Watcher runtime state."""

    running: bool
    storage_type: str | None = None
    uri: str | None = None


class IngestResponse(BaseModel):
    """Result of a one-time ingestion run."""

    status: str
    duration_seconds: float
    detail: str | None = None


class LogResponse(BaseModel):
    """Public representation of a structured audit log entry."""

    id: str
    level: str
    component: str
    message: str
    extra: dict[str, Any] | None = None
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    """Summary of a document's lifecycle status."""

    id: str
    filename: str
    status: str
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    updated_at: datetime


class StatusResponse(BaseModel):
    """Overall system status."""

    total_documents: int
    documents: list[DocumentStatusResponse]
    collections: list[dict[str, Any]]


# Module-level watcher handle (single-process).
_watcher: WatcherService | None = None
_watcher_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup and shutdown lifespan events."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("FastAPI server starting")
    try:
        yield
    finally:
        logger.info("FastAPI server shutting down")
        async with _watcher_lock:
            if _watcher is not None:
                await _watcher.stop()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="raggit",
    description="Production-grade RAG HTTP API",
    version="0.1.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _settings_env_mapping(config: RAGConfig) -> dict[str, Any]:
    """Map a RAGConfig back to environment variable names used by Settings."""
    values: dict[str, Any] = {}
    if config.database_url is not None:
        values["DATABASE_URL"] = config.database_url
    if config.qdrant_url is not None:
        values["QDRANT_URL"] = config.qdrant_url
    if config.qdrant_collection is not None:
        values["QDRANT_COLLECTION"] = config.qdrant_collection
    if config.qdrant_api_key is not None:
        values["QDRANT_API_KEY"] = config.qdrant_api_key
    if config.log_level is not None:
        values["LOG_LEVEL"] = config.log_level
    if config.chunk_size is not None:
        values["CHUNK_SIZE"] = config.chunk_size
    if config.chunk_overlap is not None:
        values["CHUNK_OVERLAP"] = config.chunk_overlap
    if config.chunking is not None:
        values["CHUNKING_DEDUP_ENABLED"] = config.chunking.dedup_enabled
        values["CHUNKING_DEDUP_SIMILARITY"] = config.chunking.dedup_similarity
        values["CHUNKING_FORMAT_AWARE"] = config.chunking.format_aware
        values["CHUNKING_PRESERVE_SECTIONS"] = config.chunking.preserve_sections
    if config.min_top_k is not None:
        values["MIN_TOP_K"] = config.min_top_k
    if config.max_top_k is not None:
        values["MAX_TOP_K"] = config.max_top_k
    if config.top_k_ratio is not None:
        values["TOP_K_RATIO"] = config.top_k_ratio
    if config.rrf_k is not None:
        values["RRF_K"] = config.rrf_k
    if config.retrieval is not None:
        r = config.retrieval
        values["RETRIEVAL_PARENT_WINDOW"] = r.parent_window
        values["RETRIEVAL_MIN_SCORE"] = r.min_score
        values["RETRIEVAL_QUERY_REWRITE"] = r.query_rewrite.value
        values["RETRIEVAL_MULTI_QUERY_COUNT"] = r.multi_query_count
        values["RETRIEVAL_TRAVERSAL_ENABLED"] = r.traversal_enabled
        values["RETRIEVAL_TRAVERSAL_MAX_STEPS"] = r.traversal_max_steps
        values["RETRIEVAL_TRAVERSAL_MIN_SCORE"] = r.traversal_min_score
        values["RETRIEVAL_TRAVERSAL_DROP_RATIO"] = r.traversal_drop_ratio
        if r.reranker is not None:
            values["RERANKER_ENABLED"] = r.reranker.enabled
            values["RERANKER_MODEL"] = r.reranker.model
            values["RERANKER_TOP_N"] = r.reranker.top_n
    if config.safety is not None:
        s = config.safety
        values["SAFETY_REFUSE_ON_EMPTY"] = s.refuse_on_empty
        values["SAFETY_REFUSE_ON_LOW_SCORE"] = s.refuse_on_low_score
        values["SAFETY_MIN_ANSWER_SCORE"] = s.min_answer_score
        values["SAFETY_GROUNDEDNESS_CHECK"] = s.groundedness_check
        values["SAFETY_PII_REDACTION"] = s.pii_redaction
        values["SAFETY_PROMPT_INJECTION_HARDENING"] = s.prompt_injection_hardening
    if config.embedding is not None:
        e = config.embedding
        values["EMBEDDING_PROVIDER"] = e.provider
        values["EMBEDDING_MODEL"] = e.model
        values["EMBEDDING_API_KEY"] = e.api_key
        values["EMBEDDING_BASE_URL"] = e.base_url
        values["EMBEDDING_BATCH_SIZE"] = e.batch_size
    if config.llm is not None:
        llm = config.llm
        values["LLM_PROVIDER"] = llm.provider
        values["LLM_MODEL"] = llm.model
        values["LLM_BASE_URL"] = llm.base_url
        values["LLM_API_KEY"] = llm.api_key
        values["LLM_TEMPERATURE"] = llm.temperature
        values["LLM_MAX_TOKENS"] = llm.max_tokens
    if config.storage is not None:
        st = config.storage
        values["STORAGE_SOURCE_TYPE"] = st.source_type.value
        values["STORAGE_URI"] = st.uri
        values["STORAGE_BUCKET"] = st.bucket
        values["STORAGE_CONTAINER"] = st.container
        values["STORAGE_PREFIX"] = st.prefix
        values["STORAGE_REGION"] = st.region
        values["STORAGE_AWS_ACCESS_KEY_ID"] = st.aws_access_key_id
        values["STORAGE_AWS_SECRET_ACCESS_KEY"] = st.aws_secret_access_key
        values["STORAGE_GCS_SERVICE_ACCOUNT_PATH"] = st.gcs_service_account_path
        values["STORAGE_AZURE_CONNECTION_STRING"] = st.azure_connection_string
        values["STORAGE_POLL_INTERVAL_SECONDS"] = st.poll_interval_seconds
    if config.default_tenant_id is not None:
        values["DEFAULT_TENANT_ID"] = config.default_tenant_id
    if config.default_tags is not None:
        values["DEFAULT_TAGS"] = ",".join(config.default_tags)
    return values


def _apply_partial_config(config: RAGConfig, partial: _PartialRAGConfig) -> RAGConfig:
    """Merge a partial config into a full RAGConfig."""
    data = config.model_dump()
    partial_data = partial.model_dump(exclude_none=True)
    for key, value in partial_data.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return RAGConfig(**data)


def _write_config(config: RAGConfig) -> str:
    """Persist config to the env file and reload settings."""
    from raggit.cli._env import write_env_file

    env_values = _settings_env_mapping(config)
    config_path = write_env_file(env_values)
    get_settings.cache_clear()
    configure_logging(get_settings().log_level)
    return str(config_path)


# ---------------------------------------------------------------------------
# Health and status
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return the currently active configuration."""
    return ConfigResponse(
        config=get_settings().rag_config,
        config_file=str(config_file_path()),
    )


@app.post("/config", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """Update the active configuration and persist it to the env file."""
    current = get_settings().rag_config
    new_config = _apply_partial_config(current, request.config)
    config_path = _write_config(new_config)

    # If the database URL changed, recreate the engine so new sessions use it.
    if new_config.database_url != current.database_url:
        await reset_engine()

    return ConfigResponse(config=new_config, config_file=config_path)


@app.get("/status", response_model=StatusResponse)
async def get_status(session: SessionDep) -> StatusResponse:
    """Return indexed documents and active embedding collections."""
    docs = await DocumentRepository(session).list_all()
    collections = await EmbeddingCollectionRepository(session).list_all()
    return StatusResponse(
        total_documents=len(docs),
        documents=[
            DocumentStatusResponse(
                id=str(d.id),
                filename=d.filename,
                status=d.status.value,
                tenant_id=d.tenant_id,
                tags=list(d.tags or []),
                updated_at=d.updated_at,
            )
            for d in docs
        ],
        collections=[
            {
                "name": c.name,
                "model": c.embedding_model,
                "provider": c.embedding_provider,
                "vector_size": c.vector_size,
                "active": c.is_active,
            }
            for c in collections
        ],
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@app.get("/documents", response_model=list[Document])
async def list_documents(
    session: SessionDep,
    status: str | None = None,
    tenant_id: str | None = None,
    tag: list[str] | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[DocumentModel]:
    """List indexed documents with optional filters."""
    repo = DocumentRepository(session)
    docs = await repo.list_all()
    if status:
        docs = [d for d in docs if d.status.value == status]
    if tenant_id:
        docs = [d for d in docs if d.tenant_id == tenant_id]
    if tag:
        docs = [d for d in docs if any(t in (d.tags or []) for t in tag)]
    return docs[offset : offset + limit]


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(session: SessionDep, document_id: UUID) -> DocumentModel:
    """Return a single document by ID."""
    doc = await DocumentRepository(session).get_by_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{document_id}")
async def delete_document(session: SessionDep, document_id: UUID) -> dict[str, str]:
    """Hard-delete a document and its chunks/vectors."""
    repo = DocumentRepository(session)
    doc = await repo.get_by_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    config = get_settings().rag_config
    vector_store = VectorStore(config)
    active = await EmbeddingCollectionRepository(session).get_active()
    if active is not None:
        vector_store.set_collection(active.name)
    await vector_store.delete_by_document(document_id)
    await vector_store.close()

    await ChunkRepository(session).delete_by_document(document_id)
    await repo.hard_delete(document_id)
    await session.commit()

    await log_event(
        session,
        level="INFO",
        component="raggit.api.server",
        message="Document deleted via API",
        extra={"document_id": str(document_id), "source_uri": doc.source_uri},
    )
    return {"status": "deleted", "document_id": str(document_id)}


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


@app.get("/documents/{document_id}/chunks", response_model=list[Chunk])
async def list_document_chunks(
    session: SessionDep,
    document_id: UUID,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[ChunkModel]:
    """Return chunks for a document."""
    doc = await DocumentRepository(session).get_by_id(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = await ChunkRepository(session).get_by_document(document_id)
    return chunks[offset : offset + limit]


@app.get("/chunks/{chunk_id}", response_model=Chunk)
async def get_chunk(session: SessionDep, chunk_id: UUID) -> ChunkModel:
    """Return a single chunk by ID."""
    chunk = await ChunkRepository(session).get_by_id(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@app.get("/logs", response_model=list[LogResponse])
async def list_logs(
    session: SessionDep,
    level: str | None = None,
    component: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[LogResponse]:
    """Return structured audit log entries."""
    stmt = select(LogModel).order_by(desc(LogModel.created_at))
    if level:
        stmt = stmt.where(LogModel.level == level.upper())
    if component:
        stmt = stmt.where(LogModel.component == component)
    stmt = stmt.offset(offset).limit(limit)
    result = await session.execute(stmt)
    rows = result.scalars().all()
    logs: list[LogResponse] = []
    for row in rows:
        extra: dict[str, Any] | None = None
        if row.extra:
            try:
                extra = json.loads(row.extra)
            except json.JSONDecodeError:
                extra = {"raw": row.extra}
        logs.append(
            LogResponse(
                id=str(row.id),
                level=row.level,
                component=row.component,
                message=row.message,
                extra=extra,
                created_at=row.created_at,
            )
        )
    return logs


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResult)
async def run_query(session: SessionDep, request: QueryRequest) -> QueryResult:
    """Run a hybrid retrieval query and optionally generate an answer."""
    from uuid import UUID as _UUID

    config = get_settings().rag_config
    configure_logging(config.log_level)

    # Apply per-request overrides.
    if request.top_k is not None:
        config.retrieval.min_top_k = request.top_k
        config.retrieval.max_top_k = request.top_k
        config.retrieval.top_k_ratio = 0.0
        config.min_top_k = request.top_k
        config.max_top_k = request.top_k
        config.top_k_ratio = 0.0
    else:
        if request.min_top_k is not None:
            config.retrieval.min_top_k = request.min_top_k
            config.min_top_k = request.min_top_k
        if request.max_top_k is not None:
            config.retrieval.max_top_k = request.max_top_k
            config.max_top_k = request.max_top_k
        if request.top_k_ratio is not None:
            config.retrieval.top_k_ratio = request.top_k_ratio
            config.top_k_ratio = request.top_k_ratio
    if request.rrf_k is not None:
        config.retrieval.rrf_k = request.rrf_k
        config.rrf_k = request.rrf_k
    if request.parent_window is not None:
        config.retrieval.parent_window = request.parent_window
    if request.min_score is not None:
        config.retrieval.min_score = request.min_score
    if request.rewrite is not None:
        config.retrieval.query_rewrite = request.rewrite
    if request.multi_query_count is not None:
        config.retrieval.multi_query_count = request.multi_query_count
    if request.reranker_enabled is not None:
        config.retrieval.reranker.enabled = request.reranker_enabled
    if request.reranker_model is not None:
        config.retrieval.reranker.model = request.reranker_model
    if request.reranker_top_n is not None:
        config.retrieval.reranker.top_n = request.reranker_top_n

    filters = request.filters
    if request.tenant_id:
        filters.tenant_id = request.tenant_id
    if request.tags:
        filters.tags = request.tags

    doc_ids = [_UUID(str(d)) for d in filters.document_ids]
    filters.document_ids = doc_ids

    embedder = create_embedder(config.embedding)
    vector_store = VectorStore(config)
    active = await EmbeddingCollectionRepository(session).get_active()
    if active is not None:
        vector_store.set_collection(active.name)

    llm = None
    llm_ready = config.llm.provider == "ollama" or bool(config.llm.api_key)
    if request.generate_answer and config.llm.provider and llm_ready:
        llm = create_llm(config.llm)

    engine = RetrievalEngine(
        embedder=embedder,
        vector_store=vector_store,
        chunk_repo=ChunkRepository(session),
        document_repo=DocumentRepository(session),
        config=config,
        llm=llm,
    )

    try:
        result = await engine.retrieve(request.query, filters=filters)
        if llm is not None:
            result = await augment_and_answer(llm, result, safety=config.safety)
    finally:
        await engine.close()

    await log_event(
        session,
        level="INFO",
        component="raggit.api.server",
        message="Query answered via API",
        extra={
            "query": request.query,
            "answer": result.answer,
            "refused": result.refused,
            "grounded": result.grounded,
            "citation_count": len(result.citations),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse)
async def run_ingest(request: IngestRequest) -> IngestResponse:
    """Run a one-time ingestion over configured storage or a given path."""
    config = get_settings().rag_config
    configure_logging(config.log_level)

    if request.tenant_id:
        config.default_tenant_id = request.tenant_id
    if request.tags:
        config.default_tags = request.tags
    if request.chunk_size is not None:
        config.chunk_size = request.chunk_size
        config.chunking.max_words_per_chunk = request.chunk_size
    if request.chunk_overlap is not None:
        config.chunk_overlap = request.chunk_overlap
        config.chunking.chunk_overlap_words = request.chunk_overlap
    if request.preserve_sections is not None:
        config.chunking.preserve_sections = request.preserve_sections
    if request.embedding_provider is not None:
        config.embedding.provider = request.embedding_provider
    if request.embedding_model is not None:
        config.embedding.model = request.embedding_model

    storage_config = config.storage
    if storage_config is None:
        raise HTTPException(status_code=400, detail="No storage configured")

    if request.path is not None:
        from pathlib import Path

        resolved = Path(request.path).expanduser().resolve()
        if storage_config.source_type == SourceType.LOCAL and not resolved.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {resolved}")
        storage_config.uri = str(resolved)

    storage = create_storage(storage_config)
    indexer = Indexer(storage, config)

    started_at = asyncio.get_event_loop().time()
    try:
        async with AsyncSessionLocal() as session, session.begin():
            await log_event(
                session,
                level="INFO",
                component="raggit.api.server",
                message="API ingestion started",
                extra={
                    "path": request.path or storage_config.uri,
                    "storage_type": storage_config.source_type.value,
                },
            )
            await indexer.sync_all(session)
            await log_event(
                session,
                level="INFO",
                component="raggit.api.server",
                message="API ingestion completed",
                extra={
                    "path": request.path or storage_config.uri,
                    "duration_seconds": asyncio.get_event_loop().time() - started_at,
                },
            )
        elapsed = asyncio.get_event_loop().time() - started_at
        return IngestResponse(
            status="completed",
            duration_seconds=elapsed,
        )
    except Exception as exc:
        logger.exception("API ingestion failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        await indexer.close()
        await storage.close()


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


@app.get("/watcher/status", response_model=WatcherStatusResponse)
async def watcher_status() -> WatcherStatusResponse:
    """Return whether the watcher is running."""
    watcher = _watcher
    running = watcher is not None
    storage_type = None
    uri = None
    if watcher is not None and watcher.config.storage is not None:
        storage_type = watcher.config.storage.source_type.value
        uri = watcher.config.storage.uri
    return WatcherStatusResponse(
        running=running,
        storage_type=storage_type,
        uri=uri,
    )


@app.post("/watcher/start", response_model=WatcherStatusResponse)
async def watcher_start(request: WatcherStartRequest) -> WatcherStatusResponse:
    """Start the storage watcher."""
    global _watcher
    async with _watcher_lock:
        if _watcher is not None:
            raise HTTPException(status_code=409, detail="Watcher is already running")

        config = get_settings().rag_config
        configure_logging(config.log_level)

        if request.tenant_id:
            config.default_tenant_id = request.tenant_id
        if request.tags:
            config.default_tags = request.tags
        if request.poll_interval_seconds is not None and config.storage is not None:
            config.storage.poll_interval_seconds = request.poll_interval_seconds

        if request.path is not None and config.storage is not None:
            from pathlib import Path

            resolved = Path(request.path).expanduser().resolve()
            if config.storage.source_type == SourceType.LOCAL and not resolved.exists():
                raise HTTPException(status_code=400, detail=f"Path does not exist: {resolved}")
            config.storage.uri = str(resolved)

        if config.storage is None:
            raise HTTPException(status_code=400, detail="No storage configured")

        _watcher = WatcherService(config)
        await _watcher.start()

    return WatcherStatusResponse(
        running=True,
        storage_type=config.storage.source_type.value,
        uri=config.storage.uri,
    )


@app.post("/watcher/stop", response_model=WatcherStatusResponse)
async def watcher_stop() -> WatcherStatusResponse:
    """Stop the storage watcher."""
    global _watcher
    async with _watcher_lock:
        if _watcher is None:
            raise HTTPException(status_code=409, detail="Watcher is not running")
        await _watcher.stop()
        storage = _watcher.config.storage
        storage_type = storage.source_type.value if storage else None
        uri = storage.uri if storage else None
        _watcher = None

    return WatcherStatusResponse(
        running=False,
        storage_type=storage_type,
        uri=uri,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Factory for FastAPI application instances (for testing)."""
    return app


async def run_api_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the uvicorn server programmatically."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
