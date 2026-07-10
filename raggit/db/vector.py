"""Qdrant vector store client with metadata filters and collection migration."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from raggit.api.models import QueryFilters, RAGConfig
from raggit.core.logging import get_logger

logger = get_logger("raggit.db.vector")


def _build_qdrant_filter(filters: QueryFilters | None) -> Filter | None:
    """Translate QueryFilters into a Qdrant payload filter.

    Prefix filters (source_uri_prefix, filename_prefix) are applied client-side
    because Qdrant does not support native prefix matching.
    """
    if filters is None:
        return None

    must: list[Any] = []
    if filters.tenant_id:
        must.append(
            FieldCondition(key="tenant_id", match=MatchValue(value=filters.tenant_id))
        )
    if filters.document_ids:
        must.append(
            FieldCondition(
                key="document_id",
                match=MatchAny(any=[str(d) for d in filters.document_ids]),
            )
        )
    if filters.tags:
        must.append(FieldCondition(key="tags", match=MatchAny(any=filters.tags)))
    if filters.created_after:
        must.append(
            FieldCondition(
                key="created_at_ts",
                range=Range(gte=filters.created_after.timestamp()),
            )
        )
    if filters.created_before:
        must.append(
            FieldCondition(
                key="created_at_ts",
                range=Range(lte=filters.created_before.timestamp()),
            )
        )

    if not must:
        return None
    return Filter(must=must)


class VectorStore:
    """Async Qdrant vector store."""

    def __init__(self, config: RAGConfig, collection: str | None = None) -> None:
        self.config = config
        self.client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
        )
        self.collection = collection or config.qdrant_collection

    def set_collection(self, name: str) -> None:
        """Switch the active collection (e.g. after model migration)."""
        self.collection = name

    async def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not exist."""
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            logger.info(
                "Creating Qdrant collection",
                collection=self.collection,
                vector_size=vector_size,
            )
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    async def collection_vector_size(self) -> int | None:
        """Return existing collection vector size, or None if missing."""
        if not await self.client.collection_exists(self.collection):
            return None
        info = await self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if vectors is None:
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
        return None

    async def upsert(
        self,
        document_id: UUID,
        chunk_id: UUID,
        vector: list[float],
        vector_id: UUID | None = None,
        *,
        source_uri: str | None = None,
        filename: str | None = None,
        tenant_id: str | None = None,
        tags: list[str] | None = None,
        page_number: int | None = None,
        section_title: str | None = None,
        created_at_ts: float | None = None,
    ) -> UUID:
        """Upsert a vector point into Qdrant with filterable payload."""
        point_id = vector_id or uuid4()
        payload: dict[str, Any] = {
            "document_id": str(document_id),
            "chunk_id": str(chunk_id),
        }
        if source_uri is not None:
            payload["source_uri"] = source_uri
        if filename is not None:
            payload["filename"] = filename
        if tenant_id is not None:
            payload["tenant_id"] = tenant_id
        if tags:
            payload["tags"] = tags
        if page_number is not None:
            payload["page_number"] = page_number
        if section_title is not None:
            payload["section_title"] = section_title
        if created_at_ts is not None:
            payload["created_at_ts"] = created_at_ts

        await self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    async def delete_by_document(self, document_id: UUID) -> None:
        """Delete all vectors for a document.

        No-op when the collection has not been created yet (first ingest).
        """
        if not await self.client.collection_exists(self.collection):
            return
        await self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id)),
                    )
                ]
            ),
        )

    async def search(
        self,
        vector: list[float],
        limit: int,
        filters: QueryFilters | None = None,
    ) -> list[tuple[UUID, UUID, float]]:
        """Search Qdrant by vector similarity.

        Returns list of (document_id, chunk_id, score).
        """
        if not await self.client.collection_exists(self.collection):
            return []

        query_filter = _build_qdrant_filter(filters)
        results = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        output: list[tuple[UUID, UUID, float]] = []
        for point in results.points:
            payload = point.payload or {}
            # Client-side prefix filters (Qdrant lacks native prefix matching)
            if filters and filters.source_uri_prefix:
                uri = str(payload.get("source_uri") or "")
                if not uri.startswith(filters.source_uri_prefix):
                    continue
            if filters and filters.filename_prefix:
                name = str(payload.get("filename") or "")
                if not name.startswith(filters.filename_prefix):
                    continue
            doc_id = UUID(payload["document_id"])
            chunk_id = UUID(payload["chunk_id"])
            output.append((doc_id, chunk_id, point.score))
        return output

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()
