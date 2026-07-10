"""Hybrid retrieval engine."""

from __future__ import annotations

import asyncio
import math
from uuid import UUID

from raggit.api.models import Chunk, QueryResult, RetrievedChunk
from raggit.core.logging import get_logger
from raggit.db.models import ChunkModel
from raggit.db.repository import ChunkRepository
from raggit.db.vector import VectorStore
from raggit.ingestion.embedder import Embedder
from raggit.retrieval.rrf import reciprocal_rank_fusion
from raggit.retrieval.sanitizer import sanitize_query

logger = get_logger("raggit.retrieval.engine")


def _clamp_top_k(total_chunks: int, min_k: int, max_k: int, ratio: float) -> int:
    """Compute dynamic top-k based on corpus size."""
    if total_chunks <= 0:
        return min_k
    return min(max_k, max(min_k, math.floor(total_chunks * ratio)))


def _as_uuid(value: UUID | str) -> UUID:
    """Normalize UUID-like values to UUID."""
    return value if isinstance(value, UUID) else UUID(str(value))


class RetrievalEngine:
    """Hybrid BM25 + semantic retrieval with RRF reranking."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_repo: ChunkRepository,
        min_top_k: int = 5,
        max_top_k: int = 50,
        top_k_ratio: float = 0.01,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_repo = chunk_repo
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.top_k_ratio = top_k_ratio
        self.rrf_k = rrf_k

    async def retrieve(self, query: str) -> QueryResult:
        """Run the full retrieval pipeline for a query."""
        cleaned_query, keywords = sanitize_query(query)
        keyword_query = " ".join(keywords)

        total_chunks = await self.chunk_repo.count_all()
        top_k = _clamp_top_k(
            total_chunks,
            self.min_top_k,
            self.max_top_k,
            self.top_k_ratio,
        )

        logger.info(
            "Retrieving",
            query=cleaned_query,
            keywords=keywords,
            top_k=top_k,
            total_chunks=total_chunks,
        )

        # Run BM25 and semantic retrieval in parallel
        async def _bm25() -> list[tuple[ChunkModel, float]]:
            if not keyword_query:
                return []
            return await self.chunk_repo.bm25_search(keyword_query, limit=top_k * 2)

        async def _semantic() -> list[tuple[UUID, UUID, float]]:
            query_vector = (await self.embedder.embed([cleaned_query]))[0]
            return await self.vector_store.search(query_vector, limit=top_k * 2)

        bm25_results, semantic_results = await asyncio.gather(_bm25(), _semantic())

        # Build ranked lists of chunk IDs
        bm25_ranked = [_as_uuid(chunk.id) for chunk, _ in bm25_results]
        semantic_ranked = [chunk_id for _, chunk_id, _ in semantic_results]

        # RRF fusion
        fused = reciprocal_rank_fusion([bm25_ranked, semantic_ranked], k=self.rrf_k)
        fused = fused[:top_k]

        # Fetch chunk models and build response
        retrieved: list[RetrievedChunk] = []
        rank_by_id: dict[UUID, dict[str, int]] = {}
        for rank, chunk_id in enumerate(bm25_ranked, start=1):
            rank_by_id.setdefault(chunk_id, {})["bm25"] = rank
        for rank, (_, chunk_id, _) in enumerate(semantic_results, start=1):
            rank_by_id.setdefault(chunk_id, {})["semantic"] = rank

        for chunk_id, score in fused:
            chunk_model = await self.chunk_repo.get_by_id(chunk_id)
            if chunk_model is None:
                continue

            chunk = Chunk.model_validate(chunk_model)
            ranks = rank_by_id.get(chunk_id, {})
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank_bm25=ranks.get("bm25"),
                    rank_semantic=ranks.get("semantic"),
                )
            )

        return QueryResult(
            query=query,
            sanitized_keywords=keywords,
            chunks=retrieved,
            total_chunks_considered=total_chunks,
        )

    async def close(self) -> None:
        """Release resources."""
        await self.vector_store.close()
