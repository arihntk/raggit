"""Document indexing orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from raggit.api.models import DocumentStatus, RAGConfig, SourceType
from raggit.core.audit import log_event
from raggit.core.logging import get_logger
from raggit.db.repository import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingCollectionRepository,
)
from raggit.db.vector import VectorStore
from raggit.ingestion.chunker import ChunkPiece, chunk_document, count_tokens
from raggit.ingestion.cleaner import clean_chunk
from raggit.ingestion.embedder import collection_name_for_model, create_embedder
from raggit.ingestion.injection import harden_against_injection
from raggit.ingestion.parser import parse_document
from raggit.ingestion.pii import redact_pii
from raggit.storage.base import Storage, StorageFile

logger = get_logger("raggit.ingestion.indexer")

# Minimum cleaned length to keep a chunk indexable.
_MIN_CHUNK_LENGTH = 20


class Indexer:
    """Orchestrates the document ingestion pipeline."""

    def __init__(self, storage: Storage, config: RAGConfig) -> None:
        self.storage = storage
        self.config = config
        self.embedder = create_embedder(config.embedding)
        self.vector_store = VectorStore(config)

    async def _ensure_embedding_collection(self, session: AsyncSession, vector_size: int) -> str:
        """Pin collection name to embedding model; migrate if model/size changed."""
        base = self.config.qdrant_collection
        model = self.embedder.model_name
        version = self.embedder.model_version
        name = collection_name_for_model(base, model, version, vector_size)

        # Always use a model-scoped name for migration safety.
        target = name

        self.vector_store.set_collection(target)
        await self.vector_store.ensure_collection(vector_size)

        repo = EmbeddingCollectionRepository(session)
        await repo.activate(
            name=target,
            embedding_provider=self.embedder.provider_name,
            embedding_model=model,
            vector_size=vector_size,
            model_version=version,
            meta={"base_collection": base},
        )
        return target

    async def index_file(self, session: AsyncSession, file: StorageFile) -> None:
        """Run the full ingestion pipeline for a single file."""
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        logger.info("Indexing file", path=file.path)

        existing = await doc_repo.get_by_uri(file.path)
        previous_hash = existing.content_hash if existing else None
        previous_status = existing.status if existing else None

        try:
            source_type = SourceType(self.storage.source_type)
        except ValueError:
            source_type = SourceType.LOCAL

        doc = await doc_repo.upsert(
            source_type=source_type,
            source_uri=file.path,
            filename=file.relative_path,
            status=DocumentStatus.INDEXING,
            tenant_id=self.config.default_tenant_id,
        )
        document_id = UUID(str(doc.id))

        try:
            content_hash = await self.storage.compute_hash(file.path)
            await log_event(
                session,
                level="INFO",
                component="raggit.ingestion.indexer",
                message="Started indexing file",
                extra={
                    "document_id": str(document_id),
                    "source_uri": file.path,
                    "filename": file.relative_path,
                    "tenant_id": doc.tenant_id,
                    "content_hash": content_hash,
                },
            )
            if (
                previous_hash is not None
                and previous_hash == content_hash
                and previous_status == DocumentStatus.COMPLETED
            ):
                await doc_repo.update_status(document_id, DocumentStatus.COMPLETED)
                logger.info("Skipping unchanged file", path=file.path)
                return
            doc.content_hash = content_hash
            await session.flush()

            raw_bytes = await self.storage.read_file(file.path)
            raw_text = parse_document(raw_bytes, file.path)

            await doc_repo.update_status(document_id, DocumentStatus.PARSED)

            pieces = chunk_document(raw_text, self.config, path=file.path)
            prepared: list[tuple[ChunkPiece, str]] = []
            for piece in pieces:
                cleaned = clean_chunk(piece.text)
                if self.config.safety.pii_redaction:
                    cleaned = redact_pii(cleaned)
                if self.config.safety.prompt_injection_hardening:
                    cleaned = harden_against_injection(cleaned)
                if len(cleaned.strip()) > _MIN_CHUNK_LENGTH:
                    prepared.append((piece, cleaned))

            await doc_repo.update_status(document_id, DocumentStatus.CHUNKED)

            # Use active collection if known
            coll_repo = EmbeddingCollectionRepository(session)
            active = await coll_repo.get_active()
            if active is not None:
                self.vector_store.set_collection(active.name)

            await chunk_repo.delete_by_document(document_id)
            await self.vector_store.delete_by_document(document_id)

            if not prepared:
                await doc_repo.update_status(document_id, DocumentStatus.COMPLETED)
                logger.info("No indexable content", path=file.path)
                return

            cleaned_chunks = [cleaned for _, cleaned in prepared]

            def _progress(done: int, total: int) -> None:
                if done == total or done % max(1, total // 10) == 0:
                    logger.info("Embedding progress", done=done, total=total, path=file.path)

            embeddings = await self.embedder.embed(
                cleaned_chunks, progress_callback=_progress
            )
            if len(embeddings) != len(prepared):
                msg = (
                    f"Embedding count mismatch: got {len(embeddings)} vectors "
                    f"for {len(prepared)} chunks"
                )
                raise RuntimeError(msg)

            vector_size = len(embeddings[0]) if embeddings else self.embedder.vector_size
            await self._ensure_embedding_collection(session, vector_size)
            await doc_repo.update_status(document_id, DocumentStatus.EMBEDDED)

            created_at_ts = datetime.now(UTC).timestamp()
            tags = list(doc.tags or [])

            for idx, ((piece, cleaned), vector) in enumerate(
                zip(prepared, embeddings, strict=True)
            ):
                chunk_model = await chunk_repo.create(
                    document_id=document_id,
                    chunk_index=idx,
                    raw_content=piece.text,
                    cleaned_content=cleaned,
                    token_count=piece.token_count or count_tokens(cleaned),
                    embedding_model=self.embedder.model_name,
                    parent_chunk_index=piece.parent_chunk_index,
                    section_title=piece.section_title,
                    page_number=piece.page_number,
                    start_offset=piece.start_offset,
                    end_offset=piece.end_offset,
                    content_hash=piece.content_hash,
                )

                await session.execute(
                    text(
                        "UPDATE chunks SET fts_vector = to_tsvector('english', :content) "
                        "WHERE id = :id"
                    ),
                    {"content": cleaned, "id": chunk_model.id},
                )

                vector_id = await self.vector_store.upsert(
                    document_id=document_id,
                    chunk_id=UUID(str(chunk_model.id)),
                    vector=vector,
                    source_uri=file.path,
                    filename=file.relative_path,
                    tenant_id=doc.tenant_id,
                    tags=tags,
                    page_number=piece.page_number,
                    section_title=piece.section_title,
                    created_at_ts=created_at_ts,
                )
                chunk_model.vector_id = str(vector_id)

            await doc_repo.update_status(document_id, DocumentStatus.COMPLETED)
            logger.info(
                "Indexed file successfully",
                path=file.path,
                chunks=len(prepared),
                collection=self.vector_store.collection,
            )
            await log_event(
                session,
                level="INFO",
                component="raggit.ingestion.indexer",
                message="Indexed file successfully",
                extra={
                    "document_id": str(document_id),
                    "source_uri": file.path,
                    "filename": file.relative_path,
                    "chunks": len(prepared),
                    "collection": self.vector_store.collection,
                    "tenant_id": doc.tenant_id,
                },
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc!s}"
            if self.config.safety.pii_redaction:
                error_msg = redact_pii(error_msg)
            logger.exception("Failed to index file", path=file.path, error=error_msg)
            await log_event(
                session,
                level="ERROR",
                component="raggit.ingestion.indexer",
                message="Failed to index file",
                extra={
                    "document_id": str(document_id),
                    "source_uri": file.path,
                    "filename": file.relative_path,
                    "tenant_id": doc.tenant_id,
                    "error": error_msg,
                },
            )
            await doc_repo.update_status(
                document_id, DocumentStatus.FAILED, error_message=error_msg
            )

    async def remove_file(self, session: AsyncSession, file: StorageFile) -> None:
        """Remove a document and all associated data."""
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        doc = await doc_repo.get_by_uri(file.path)
        if not doc:
            logger.warning("Document not found for deletion", path=file.path)
            return

        coll_repo = EmbeddingCollectionRepository(session)
        active = await coll_repo.get_active()
        if active is not None:
            self.vector_store.set_collection(active.name)

        document_id = UUID(str(doc.id))
        await chunk_repo.delete_by_document(document_id)
        await self.vector_store.delete_by_document(document_id)
        await doc_repo.hard_delete(document_id)

        logger.info("Removed document and associated data", path=file.path)
        await log_event(
            session,
            level="INFO",
            component="raggit.ingestion.indexer",
            message="Removed document and associated data",
            extra={
                "document_id": str(document_id),
                "source_uri": file.path,
                "filename": file.relative_path,
            },
        )

    async def sync_all(
        self,
        session: AsyncSession,
        *,
        progress_callback: Callable[[StorageFile, int, int], Any] | None = None,
    ) -> None:
        """Full sync: index new/modified files, remove missing ones."""
        files = await self.storage.list_files()
        current_uris = {f.path for f in files}

        doc_repo = DocumentRepository(session)
        docs = await doc_repo.list_all()
        existing_uris = {d.source_uri for d in docs if d.status != DocumentStatus.DELETED}

        for uri in existing_uris - current_uris:
            file = StorageFile(
                path=uri,
                relative_path=uri,
                size=0,
                modified_at=datetime.now(UTC),
            )
            await self.remove_file(session, file)

        total = len(files)
        for i, file in enumerate(files, start=1):
            await self.index_file(session, file)
            if progress_callback is not None:
                progress_callback(file, i, total)

    async def close(self) -> None:
        """Release resources."""
        await self.vector_store.close()
