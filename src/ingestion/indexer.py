"""Document indexing orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from raggit.api.models import DocumentStatus, RAGConfig
from raggit.core.logging import get_logger
from raggit.db.repository import ChunkRepository, DocumentRepository
from raggit.db.vector import VectorStore
from raggit.ingestion.chunker import chunk_text
from raggit.ingestion.cleaner import clean_chunk
from raggit.ingestion.embedder import create_embedder
from raggit.ingestion.parser import parse_document
from raggit.storage.base import Storage, StorageFile
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger("raggit.ingestion.indexer")


class Indexer:
    """Orchestrates the document ingestion pipeline."""

    def __init__(self, storage: Storage, config: RAGConfig) -> None:
        self.storage = storage
        self.config = config
        self.embedder = create_embedder(config.embedding)
        self.vector_store = VectorStore(config)

    async def index_file(self, session: AsyncSession, file: StorageFile) -> None:
        """Run the full ingestion pipeline for a single file."""
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        logger.info("Indexing file", path=file.path)

        # 1. Upsert document record
        doc = await doc_repo.upsert(
            source_type=self.storage.source_type,
            source_uri=file.path,
            filename=file.relative_path,
            status=DocumentStatus.INDEXING,
        )

        try:
            # 2. Compute hash and skip if unchanged
            if isinstance(self.storage, object) and hasattr(self.storage, "compute_hash"):
                content_hash = await self.storage.compute_hash(file.path)
                if doc.content_hash == content_hash and doc.status == DocumentStatus.COMPLETED:
                    logger.info("Skipping unchanged file", path=file.path)
                    return
                doc.content_hash = content_hash
                await session.flush()

            # 3. Read and parse
            raw_bytes = await self.storage.read_file(file.path)
            raw_text = parse_document(raw_bytes, file.path)

            await doc_repo.update_status(doc.id, DocumentStatus.PARSED)

            # 4. Chunk
            chunks = chunk_text(raw_text, self.config)

            # 5. Ensure vector collection exists and delete old chunks/vectors
            await self.vector_store.ensure_collection(self.embedder.vector_size)
            await chunk_repo.delete_by_document(UUID(doc.id))
            await self.vector_store.delete_by_document(UUID(doc.id))

            cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]
            cleaned_chunks = [c for c in cleaned_chunks if len(c.strip()) > 20]

            await doc_repo.update_status(doc.id, DocumentStatus.CHUNKED)

            if not cleaned_chunks:
                await doc_repo.update_status(doc.id, DocumentStatus.COMPLETED)
                logger.info("No indexable content", path=file.path)
                return

            # 6. Embed
            embeddings = await self.embedder.embed(cleaned_chunks)

            await doc_repo.update_status(doc.id, DocumentStatus.EMBEDDED)

            # 7. Persist chunks and vectors
            for idx, (raw, cleaned, vector) in enumerate(
                zip(chunks, cleaned_chunks, embeddings, strict=False)
            ):
                chunk_model = await chunk_repo.create(
                    document_id=UUID(doc.id),
                    chunk_index=idx,
                    raw_content=raw,
                    cleaned_content=cleaned,
                    token_count=len(cleaned.split()),
                    embedding_model=self.embedder.model_name,
                )
                # Update fts_vector
                chunk_model.fts_vector = text("to_tsvector('english', :content)").bindparams(
                    content=cleaned
                )
                await session.flush()

                vector_id = await self.vector_store.upsert(
                    document_id=UUID(doc.id),
                    chunk_id=UUID(chunk_model.id),
                    vector=vector,
                )
                chunk_model.vector_id = str(vector_id)

            await doc_repo.update_status(doc.id, DocumentStatus.COMPLETED)
            logger.info(
                "Indexed file successfully",
                path=file.path,
                chunks=len(cleaned_chunks),
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc!s}"
            logger.exception("Failed to index file", path=file.path, error=error_msg)
            await doc_repo.update_status(
                UUID(doc.id), DocumentStatus.FAILED, error_message=error_msg
            )

    async def remove_file(self, session: AsyncSession, file: StorageFile) -> None:
        """Remove a document and all associated data."""
        doc_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        doc = await doc_repo.get_by_uri(file.path)
        if not doc:
            logger.warning("Document not found for deletion", path=file.path)
            return

        document_id = UUID(doc.id)
        await chunk_repo.delete_by_document(document_id)
        await self.vector_store.delete_by_document(document_id)
        await doc_repo.hard_delete(document_id)

        logger.info("Removed document and associated data", path=file.path)

    async def sync_all(self, session: AsyncSession) -> None:
        """Full sync: index new/modified files, remove missing ones."""
        files = await self.storage.list_files()
        current_uris = {f.path for f in files}

        doc_repo = DocumentRepository(session)
        docs = await doc_repo.list_all()
        existing_uris = {d.source_uri for d in docs if d.status != DocumentStatus.DELETED}

        # Delete files no longer present
        for uri in existing_uris - current_uris:
            file = StorageFile(
                path=uri,
                relative_path=uri,
                size=0,
                modified_at=datetime.now(UTC),
            )
            await self.remove_file(session, file)

        # Index files currently present
        for file in files:
            await self.index_file(session, file)

    async def close(self) -> None:
        """Release resources."""
        await self.vector_store.close()
