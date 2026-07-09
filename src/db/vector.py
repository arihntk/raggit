"""Qdrant vector store client."""

from __future__ import annotations

from uuid import UUID, uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from raggit.api.models import RAGConfig
from raggit.core.logging import get_logger

logger = get_logger("raggit.db.vector")


class VectorStore:
    """Async Qdrant vector store."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
        )
        self.collection = config.qdrant_collection

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

    async def upsert(
        self,
        document_id: UUID,
        chunk_id: UUID,
        vector: list[float],
        vector_id: UUID | None = None,
    ) -> UUID:
        """Upsert a vector point into Qdrant."""
        point_id = vector_id or uuid4()
        await self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(point_id),
                    vector=vector,
                    payload={
                        "document_id": str(document_id),
                        "chunk_id": str(chunk_id),
                    },
                )
            ],
        )
        return point_id

    async def delete_by_document(self, document_id: UUID) -> None:
        """Delete all vectors for a document."""
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
    ) -> list[tuple[UUID, UUID, float]]:
        """Search Qdrant by vector similarity.

        Returns list of (document_id, chunk_id, score).
        """
        results = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        output: list[tuple[UUID, UUID, float]] = []
        for point in results.points:
            payload = point.payload or {}
            doc_id = UUID(payload["document_id"])
            chunk_id = UUID(payload["chunk_id"])
            output.append((doc_id, chunk_id, point.score))
        return output

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()
