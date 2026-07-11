"""Data access layer for raggit entities."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from raggit.api.models import DocumentStatus, QueryFilters, SourceType
from raggit.db.models import (
    ChunkModel,
    DocumentModel,
    EmbeddingCollectionModel,
    LogModel,
)


class DocumentRepository:
    """Repository for document CRUD and lifecycle operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_uri(self, source_uri: str) -> DocumentModel | None:
        """Fetch a document by its source URI."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.source_uri == source_uri)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id: UUID) -> DocumentModel | None:
        """Fetch a document by ID."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.id == str(document_id))
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[DocumentModel]:
        """List all documents."""
        result = await self.session.execute(select(DocumentModel))
        return list(result.scalars().all())

    async def list_by_status(self, status: DocumentStatus) -> list[DocumentModel]:
        """List documents by status."""
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.status == status)
        )
        return list(result.scalars().all())

    async def upsert(
        self,
        source_type: SourceType | str,
        source_uri: str,
        filename: str,
        content_hash: str | None = None,
        status: DocumentStatus = DocumentStatus.PENDING,
        tenant_id: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentModel:
        """Insert or re-activate a document."""
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        existing = await self.get_by_uri(source_uri)
        if existing:
            existing.status = status
            existing.error_message = None
            existing.deleted_at = None
            if content_hash is not None:
                existing.content_hash = content_hash
            existing.filename = filename
            if tenant_id is not None:
                existing.tenant_id = tenant_id
            if tags is not None:
                existing.tags = tags
            await self.session.flush()
            return existing

        doc = DocumentModel(
            source_type=source_type,
            source_uri=source_uri,
            filename=filename,
            content_hash=content_hash,
            status=status,
            tenant_id=tenant_id,
            tags=tags or [],
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def update_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> None:
        """Update document status."""
        await self.session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == str(document_id))
            .values(status=status, error_message=error_message)
        )

    async def mark_deleted(self, document_id: UUID) -> None:
        """Soft-delete a document."""
        await self.session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == str(document_id))
            .values(status=DocumentStatus.DELETED, deleted_at=func.now())
        )

    async def hard_delete(self, document_id: UUID) -> None:
        """Permanently delete a document and cascaded chunks."""
        await self.session.execute(
            delete(DocumentModel).where(DocumentModel.id == str(document_id))
        )


class ChunkRepository:
    """Repository for chunk CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        document_id: UUID,
        chunk_index: int,
        raw_content: str,
        cleaned_content: str,
        word_count: int | None = None,
        embedding_model: str | None = None,
        vector_id: UUID | None = None,
        parent_chunk_index: int | None = None,
        prev_chunk_id: UUID | None = None,
        next_chunk_id: UUID | None = None,
        section_title: str | None = None,
        page_number: int | None = None,
        start_offset: int | None = None,
        end_offset: int | None = None,
        content_hash: str | None = None,
    ) -> ChunkModel:
        """Create a new chunk."""
        chunk = ChunkModel(
            document_id=str(document_id),
            chunk_index=chunk_index,
            raw_content=raw_content,
            cleaned_content=cleaned_content,
            word_count=word_count,
            embedding_model=embedding_model,
            vector_id=str(vector_id) if vector_id else None,
            parent_chunk_index=parent_chunk_index,
            prev_chunk_id=str(prev_chunk_id) if prev_chunk_id else None,
            next_chunk_id=str(next_chunk_id) if next_chunk_id else None,
            section_title=section_title,
            page_number=page_number,
            start_offset=start_offset,
            end_offset=end_offset,
            content_hash=content_hash,
        )
        self.session.add(chunk)
        await self.session.flush()
        return chunk

    async def delete_by_document(self, document_id: UUID) -> None:
        """Delete all chunks for a document."""
        await self.session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == str(document_id))
        )

    async def get_by_document(self, document_id: UUID) -> list[ChunkModel]:
        """Get all chunks for a document."""
        result = await self.session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == str(document_id))
            .order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())

    async def get_by_id(self, chunk_id: UUID) -> ChunkModel | None:
        """Get a chunk by ID."""
        result = await self.session.execute(
            select(ChunkModel).where(ChunkModel.id == str(chunk_id))
        )
        return result.scalar_one_or_none()

    async def get_by_chunk_index(
        self, document_id: UUID, chunk_index: int
    ) -> ChunkModel | None:
        """Get a chunk by its document and sequential index."""
        result = await self.session.execute(
            select(ChunkModel)
            .where(
                ChunkModel.document_id == str(document_id),
                ChunkModel.chunk_index == chunk_index,
            )
        )
        return result.scalar_one_or_none()

    async def update_links(
        self,
        chunk_id: UUID,
        *,
        prev_chunk_id: UUID | None = None,
        next_chunk_id: UUID | None = None,
    ) -> None:
        """Update sequential sibling links for a chunk."""
        values: dict[str, Any] = {}
        if prev_chunk_id is not None:
            values["prev_chunk_id"] = str(prev_chunk_id)
        if next_chunk_id is not None:
            values["next_chunk_id"] = str(next_chunk_id)
        if values:
            await self.session.execute(
                update(ChunkModel)
                .where(ChunkModel.id == str(chunk_id))
                .values(**values)
            )

    async def get_siblings(
        self,
        document_id: UUID,
        chunk_index: int,
        window: int,
    ) -> list[ChunkModel]:
        """Return chunks in [chunk_index - window, chunk_index + window] for a document."""
        if window <= 0:
            return []
        result = await self.session.execute(
            select(ChunkModel)
            .where(
                ChunkModel.document_id == str(document_id),
                ChunkModel.chunk_index >= chunk_index - window,
                ChunkModel.chunk_index <= chunk_index + window,
            )
            .order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())

    async def count_all(self) -> int:
        """Count total chunks in the index."""
        result = await self.session.execute(select(func.count()).select_from(ChunkModel))
        return int(result.scalar_one())

    def _apply_filters(self, stmt, filters: QueryFilters | None):  # type: ignore[no-untyped-def]
        """Join documents and apply metadata filters."""
        if filters is None:
            return stmt

        stmt = stmt.join(DocumentModel, ChunkModel.document_id == DocumentModel.id)

        if filters.source_uri_prefix:
            stmt = stmt.where(DocumentModel.source_uri.startswith(filters.source_uri_prefix))
        if filters.filename_prefix:
            stmt = stmt.where(DocumentModel.filename.startswith(filters.filename_prefix))
        if filters.tenant_id:
            stmt = stmt.where(DocumentModel.tenant_id == filters.tenant_id)
        if filters.document_ids:
            ids = [str(d) for d in filters.document_ids]
            stmt = stmt.where(ChunkModel.document_id.in_(ids))
        if filters.created_after:
            stmt = stmt.where(DocumentModel.created_at >= filters.created_after)
        if filters.created_before:
            stmt = stmt.where(DocumentModel.created_at <= filters.created_before)
        if filters.tags:
            # PostgreSQL array overlap: document tags contain any requested tag
            stmt = stmt.where(DocumentModel.tags.overlap(filters.tags))
        return stmt

    async def bm25_search(
        self,
        query: str,
        limit: int = 50,
        filters: QueryFilters | None = None,
    ) -> list[tuple[ChunkModel, float]]:
        """Execute a BM25-ish full-text search against chunk content."""
        ts_query = func.plainto_tsquery("english", query)
        stmt = select(
            ChunkModel,
            func.ts_rank_cd(ChunkModel.fts_vector, ts_query).label("rank"),
        ).where(ChunkModel.fts_vector.op("@@")(ts_query))
        stmt = self._apply_filters(stmt, filters)
        stmt = stmt.order_by(func.ts_rank_cd(ChunkModel.fts_vector, ts_query).desc()).limit(
            limit
        )
        result = await self.session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]


class EmbeddingCollectionRepository:
    """Tracks embedding model -> Qdrant collection mappings."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active(self) -> EmbeddingCollectionModel | None:
        result = await self.session.execute(
            select(EmbeddingCollectionModel).where(EmbeddingCollectionModel.is_active.is_(True))
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> EmbeddingCollectionModel | None:
        result = await self.session.execute(
            select(EmbeddingCollectionModel).where(EmbeddingCollectionModel.name == name)
        )
        return result.scalar_one_or_none()

    async def activate(
        self,
        name: str,
        embedding_provider: str,
        embedding_model: str,
        vector_size: int,
        model_version: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> EmbeddingCollectionModel:
        """Mark a collection as the active one; deactivate others."""
        await self.session.execute(
            update(EmbeddingCollectionModel).values(is_active=False)
        )
        existing = await self.get_by_name(name)
        if existing:
            existing.is_active = True
            existing.embedding_provider = embedding_provider
            existing.embedding_model = embedding_model
            existing.model_version = model_version
            existing.vector_size = vector_size
            if meta is not None:
                existing.meta = meta
            await self.session.flush()
            return existing

        row = EmbeddingCollectionModel(
            name=name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            model_version=model_version,
            vector_size=vector_size,
            is_active=True,
            meta=meta,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_all(self) -> list[EmbeddingCollectionModel]:
        result = await self.session.execute(select(EmbeddingCollectionModel))
        return list(result.scalars().all())


class LogRepository:
    """Repository for structured logs."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        level: str,
        component: str,
        message: str,
        extra: str | None = None,
    ) -> LogModel:
        """Persist a structured log entry."""
        log = LogModel(
            level=level,
            component=component,
            message=message,
            extra=extra,
        )
        self.session.add(log)
        await self.session.flush()
        return log
