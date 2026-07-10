"""Hybrid retrieval engine with RRF weights, rerank, filters, parents, rewrite."""

from __future__ import annotations

import asyncio
import math
from uuid import UUID

from raggit.api.models import (
    Chunk,
    Citation,
    QueryFilters,
    QueryResult,
    QueryRewriteMode,
    RAGConfig,
    RetrievedChunk,
)
from raggit.core.logging import get_logger
from raggit.db.models import ChunkModel
from raggit.db.repository import ChunkRepository, DocumentRepository
from raggit.db.vector import VectorStore
from raggit.ingestion.embedder import Embedder
from raggit.ingestion.injection import harden_against_injection
from raggit.llm.base import LLMProvider
from raggit.retrieval.reranker import CrossEncoderReranker, create_reranker
from raggit.retrieval.rewrite import rewrite_queries
from raggit.retrieval.rrf import reciprocal_rank_fusion
from raggit.retrieval.safety import apply_score_threshold, should_refuse
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


def _chunk_to_api(
    chunk_model: ChunkModel,
    *,
    source_uri: str | None = None,
    filename: str | None = None,
    tenant_id: str | None = None,
    tags: list[str] | None = None,
    harden: bool = False,
) -> Chunk:
    """Convert ORM chunk (+ optional doc fields) to API model."""
    content = chunk_model.cleaned_content
    if harden:
        content = harden_against_injection(content)
    return Chunk(
        id=_as_uuid(chunk_model.id),
        document_id=_as_uuid(chunk_model.document_id),
        chunk_index=chunk_model.chunk_index,
        raw_content=chunk_model.raw_content,
        cleaned_content=content,
        token_count=chunk_model.token_count,
        embedding_model=chunk_model.embedding_model,
        vector_id=_as_uuid(chunk_model.vector_id) if chunk_model.vector_id else None,
        parent_chunk_index=chunk_model.parent_chunk_index,
        section_title=chunk_model.section_title,
        page_number=chunk_model.page_number,
        start_offset=chunk_model.start_offset,
        end_offset=chunk_model.end_offset,
        content_hash=chunk_model.content_hash,
        source_uri=source_uri,
        filename=filename,
        tenant_id=tenant_id,
        tags=tags or [],
        created_at=chunk_model.created_at,
    )


def _make_citation(chunk: Chunk, score: float) -> Citation:
    return Citation(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        source_uri=chunk.source_uri,
        filename=chunk.filename,
        chunk_index=chunk.chunk_index,
        page_number=chunk.page_number,
        section_title=chunk.section_title,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        score=score,
        excerpt=chunk.cleaned_content[:240],
    )


class RetrievalEngine:
    """Hybrid BM25 + semantic retrieval with weighted RRF and optional rerank."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_repo: ChunkRepository,
        config: RAGConfig | None = None,
        *,
        min_top_k: int | None = None,
        max_top_k: int | None = None,
        top_k_ratio: float | None = None,
        rrf_k: int | None = None,
        llm: LLMProvider | None = None,
        document_repo: DocumentRepository | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_repo = chunk_repo
        self.document_repo = document_repo
        self.config = config
        retrieval = config.retrieval if config else None
        self.min_top_k = min_top_k if min_top_k is not None else (
            retrieval.min_top_k if retrieval else 5
        )
        self.max_top_k = max_top_k if max_top_k is not None else (
            retrieval.max_top_k if retrieval else 50
        )
        self.top_k_ratio = top_k_ratio if top_k_ratio is not None else (
            retrieval.top_k_ratio if retrieval else 0.01
        )
        self.rrf_k = rrf_k if rrf_k is not None else (retrieval.rrf_k if retrieval else 60)
        self.rrf_weight_bm25 = retrieval.rrf_weight_bm25 if retrieval else 1.0
        self.rrf_weight_semantic = retrieval.rrf_weight_semantic if retrieval else 1.0
        self.min_score = retrieval.min_score if retrieval else None
        self.parent_window = retrieval.parent_window if retrieval else 0
        self.query_rewrite = (
            retrieval.query_rewrite if retrieval else QueryRewriteMode.NONE
        )
        self.multi_query_count = retrieval.multi_query_count if retrieval else 3
        self.llm = llm
        self.safety = config.safety if config else None
        self._reranker: CrossEncoderReranker | None = (
            create_reranker(retrieval.reranker) if retrieval else None
        )
        self._harden = bool(
            config and config.safety.prompt_injection_hardening
        )

    async def retrieve(
        self,
        query: str,
        filters: QueryFilters | None = None,
    ) -> QueryResult:
        """Run the full retrieval pipeline for a query."""
        cleaned_query, keywords = sanitize_query(query)

        total_chunks = await self.chunk_repo.count_all()
        top_k = _clamp_top_k(
            total_chunks,
            self.min_top_k,
            self.max_top_k,
            self.top_k_ratio,
        )

        rewritten, hyde_passage = await rewrite_queries(
            self.query_rewrite,
            cleaned_query,
            self.llm,
            multi_query_count=self.multi_query_count,
        )

        logger.info(
            "Retrieving",
            query=cleaned_query,
            keywords=keywords,
            top_k=top_k,
            total_chunks=total_chunks,
            rewrite_mode=self.query_rewrite.value,
            rewritten=rewritten,
        )

        candidate_limit = top_k * 2
        if self._reranker is not None and self.config:
            candidate_limit = max(
                candidate_limit, self.config.retrieval.reranker.top_n
            )

        # Multi-query BM25: merge ranked lists
        async def _bm25_all() -> list[UUID]:
            ranked: list[UUID] = []
            seen: set[UUID] = set()
            for q in rewritten:
                _, kws = sanitize_query(q)
                keyword_query = " ".join(kws) if kws else q
                if not keyword_query.strip():
                    continue
                hits = await self.chunk_repo.bm25_search(
                    keyword_query, limit=candidate_limit, filters=filters
                )
                for chunk, _ in hits:
                    cid = _as_uuid(chunk.id)
                    if cid not in seen:
                        seen.add(cid)
                        ranked.append(cid)
            return ranked

        embed_text = hyde_passage if hyde_passage else cleaned_query

        async def _semantic() -> list[tuple[UUID, UUID, float]]:
            query_vector = (await self.embedder.embed([embed_text]))[0]
            return await self.vector_store.search(
                query_vector, limit=candidate_limit, filters=filters
            )

        bm25_ranked, semantic_results = await asyncio.gather(_bm25_all(), _semantic())
        semantic_ranked = [chunk_id for _, chunk_id, _ in semantic_results]

        fused = reciprocal_rank_fusion(
            [bm25_ranked, semantic_ranked],
            k=self.rrf_k,
            weights=[self.rrf_weight_bm25, self.rrf_weight_semantic],
        )

        rank_by_id: dict[UUID, dict[str, int]] = {}
        for rank, chunk_id in enumerate(bm25_ranked, start=1):
            rank_by_id.setdefault(chunk_id, {})["bm25"] = rank
        for rank, (_, chunk_id, _) in enumerate(semantic_results, start=1):
            rank_by_id.setdefault(chunk_id, {})["semantic"] = rank

        # Optional cross-encoder rerank on top candidates
        if self._reranker is not None and fused:
            top_n = (
                self.config.retrieval.reranker.top_n
                if self.config
                else 20
            )
            candidates: list[tuple[UUID, str]] = []
            for chunk_id, _ in fused[:top_n]:
                model = await self.chunk_repo.get_by_id(chunk_id)
                if model is not None:
                    candidates.append((chunk_id, model.cleaned_content))
            reranked = await self._reranker.rerank(cleaned_query, candidates)
            score_map = dict(reranked)
            fused = [(cid, score_map.get(cid, 0.0)) for cid, _ in fused if cid in score_map]
            fused.sort(key=lambda item: item[1], reverse=True)
            for rank, (chunk_id, _) in enumerate(fused, start=1):
                rank_by_id.setdefault(chunk_id, {})["rerank"] = rank

        fused = fused[:top_k]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, score in fused:
            chunk_model = await self.chunk_repo.get_by_id(chunk_id)
            if chunk_model is None:
                continue

            source_uri = filename = tenant_id = None
            tags: list[str] = []
            if self.document_repo is not None:
                doc = await self.document_repo.get_by_id(_as_uuid(chunk_model.document_id))
                if doc is not None:
                    source_uri = doc.source_uri
                    filename = doc.filename
                    tenant_id = doc.tenant_id
                    tags = list(doc.tags or [])

            # Parent-document expansion: merge sibling windows into content
            display_model = chunk_model
            if self.parent_window > 0:
                siblings = await self.chunk_repo.get_siblings(
                    _as_uuid(chunk_model.document_id),
                    chunk_model.chunk_index,
                    self.parent_window,
                )
                if siblings:
                    merged = "\n\n".join(s.cleaned_content for s in siblings)
                    chunk = _chunk_to_api(
                        display_model,
                        source_uri=source_uri,
                        filename=filename,
                        tenant_id=tenant_id,
                        tags=tags,
                        harden=self._harden,
                    )
                    chunk = chunk.model_copy(update={"cleaned_content": (
                        harden_against_injection(merged) if self._harden else merged
                    )})
                else:
                    chunk = _chunk_to_api(
                        display_model,
                        source_uri=source_uri,
                        filename=filename,
                        tenant_id=tenant_id,
                        tags=tags,
                        harden=self._harden,
                    )
            else:
                chunk = _chunk_to_api(
                    display_model,
                    source_uri=source_uri,
                    filename=filename,
                    tenant_id=tenant_id,
                    tags=tags,
                    harden=self._harden,
                )

            ranks = rank_by_id.get(chunk_id, {})
            citation = _make_citation(chunk, score)
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank_bm25=ranks.get("bm25"),
                    rank_semantic=ranks.get("semantic"),
                    rank_rerank=ranks.get("rerank"),
                    citation=citation,
                )
            )

        retrieved = apply_score_threshold(retrieved, self.min_score, min_keep=0)

        refused = False
        refusal_reason = None
        if self.safety is not None:
            refused, refusal_reason = should_refuse(retrieved, self.safety)

        citations = [r.citation for r in retrieved if r.citation is not None]

        return QueryResult(
            query=query,
            sanitized_keywords=keywords,
            chunks=retrieved,
            citations=citations,
            refused=refused,
            refusal_reason=refusal_reason,
            rewritten_queries=rewritten if rewritten != [cleaned_query] else [],
            total_chunks_considered=total_chunks,
        )

    async def close(self) -> None:
        """Release resources."""
        await self.vector_store.close()
