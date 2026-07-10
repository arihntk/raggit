"""Data access layer for raggit entities."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from raggit.api.models import DocumentStatus, SourceType
from raggit.db.models import ChunkModel, DocumentModel, LogModel


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
            await self.session.flush()
            return existing

        doc = DocumentModel(
            source_type=source_type,
            source_uri=source_uri,
            filename=filename,
            content_hash=content_hash,
            status=status,
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
        token_count: int | None = None,
        embedding_model: str | None = None,
        vector_id: UUID | None = None,
    ) -> ChunkModel:
        """Create a new chunk."""
        chunk = ChunkModel(
            document_id=str(document_id),
            chunk_index=chunk_index,
            raw_content=raw_content,
            cleaned_content=cleaned_content,
            token_count=token_count,
            embedding_model=embedding_model,
            vector_id=str(vector_id) if vector_id else None,
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

    async def count_all(self) -> int:
        """Count total chunks in the index."""
        result = await self.session.execute(select(func.count()).select_from(ChunkModel))
        return int(result.scalar_one())

    async def bm25_search(self, query: str, limit: int = 50) -> list[tuple[ChunkModel, float]]:
        """Execute a BM25-ish full-text search against chunk content.

        Uses PostgreSQL ts_rank_cd for ranking.
        """
        from sqlalchemy import func

        ts_query = func.plainto_tsquery("english", query)
        stmt = (
            select(
                ChunkModel,
                func.ts_rank_cd(ChunkModel.fts_vector, ts_query).label("rank"),
            )
            .where(ChunkModel.fts_vector.op("@@")(ts_query))
            .order_by(func.ts_rank_cd(ChunkModel.fts_vector, ts_query).desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]


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
