"""Pipeline evaluation – ingestion and retrieval chains.

Pipeline tier tests the *connection* between components:

- **Ingestion pipeline**: parse → chunk → clean → dedup → PII/injection → embed → store
- **Retrieval pipeline**: sanitize → rewrite → BM25/semantic → RRF → rerank → threshold → parent-window → traversal

Unlike component eval (synthetic, no DB), pipeline eval uses the real DB/vector store
but focuses on the pipeline as a whole, not the final LLM answer.
"""

from __future__ import annotations

import time
from typing import Any

from raggit.api.models import QueryFilters, RAGConfig
from raggit.core.logging import get_logger
from raggit.db.repository import ChunkRepository, DocumentRepository, EmbeddingCollectionRepository
from raggit.db.session import AsyncSessionLocal
from raggit.db.vector import VectorStore
from raggit.eval.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from raggit.eval.models import (
    EvalDataset,
    EvalKind,
    EvalReport,
    EvalSummary,
    MetricAggregate,
    PipelineScores,
    PipelineType,
    RetrievalScores,
    TestResult,
)
from raggit.ingestion.embedder import create_embedder
from raggit.retrieval.engine import RetrievalEngine

logger = get_logger("raggit.eval.pipeline")


def _aggregate(values: list[float]) -> MetricAggregate:
    if not values:
        return MetricAggregate(metric="", values=[])
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
    return MetricAggregate(metric="", mean=sum(values) / n, min=min(values), max=max(values), median=median, values=values)


class PipelineRunner:
    """Run pipeline-tier datasets."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embedder = create_embedder(config.embedding)
        self.vector_store = VectorStore(config)

    async def _run_ingestion(self, tc: Any) -> PipelineScores:
        """Evaluate ingestion pipeline for a synthetic document."""
        from raggit.ingestion.chunker import chunk_document
        from raggit.ingestion.cleaner import clean_chunk
        from raggit.ingestion.parser import registry

        # tc.documents is list of {path, content_bytes/text, expected_text}
        scores = PipelineScores()
        start = time.perf_counter()
        try:
            total = len(tc.documents) if tc.documents else 1
            success = 0
            for doc in tc.documents or [{"path": "doc.txt", "text": tc.query or "hello world"}]:
                path = doc.get("path", "doc.txt")
                text = doc.get("text") or doc.get("content", "")
                content_bytes = doc.get("content_bytes")
                if content_bytes and isinstance(content_bytes, str):
                    content_bytes = content_bytes.encode()
                if content_bytes:
                    parsed = registry.parse(content_bytes, path)
                else:
                    parsed = text
                # chunk → clean
                pieces = chunk_document(parsed, self.config, path=path)
                cleaned = [clean_chunk(p.text) for p in pieces]
                # basic success: at least one non-empty cleaned chunk
                if any(c.strip() for c in cleaned):
                    success += 1
            scores.ingestion_success = success / total if total else 1.0
            scores.latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            logger.exception("Ingestion pipeline test failed", test_id=tc.id, error=str(exc))
            scores.ingestion_success = 0.0
            scores.latency_ms = (time.perf_counter() - start) * 1000
        return scores

    async def _run_retrieval(self, tc: Any, dataset: EvalDataset) -> tuple[RetrievalScores, PipelineScores, list[Any], float]:
        """Evaluate retrieval pipeline via real engine."""
        retrieval_scores = RetrievalScores()
        pipe_scores = PipelineScores()
        start = time.perf_counter()
        retrieved_ids: list[Any] = []
        try:
            async with AsyncSessionLocal() as session, session.begin():
                active = await EmbeddingCollectionRepository(session).get_active()
                if active is not None:
                    self.vector_store.set_collection(active.name)
                filters: QueryFilters = tc.filters or QueryFilters()
                engine = RetrievalEngine(
                    embedder=self.embedder,
                    vector_store=self.vector_store,
                    chunk_repo=ChunkRepository(session),
                    document_repo=DocumentRepository(session),
                    config=self.config,
                )
                result = await engine.retrieve(tc.query or "", filters=filters)
                retrieved_ids = [r.chunk.id for r in result.chunks]
                # Score retrieval
                if tc.expected_chunk_ids:
                    relevant = tc.expected_chunk_ids
                    retrieved = retrieved_ids
                    for k in dataset.k_values:
                        key = f"@{k}"
                        retrieval_scores.recall_at_k[key] = recall_at_k(retrieved, relevant, k=k)
                        retrieval_scores.precision_at_k[key] = precision_at_k(retrieved, relevant, k=k)
                        retrieval_scores.ndcg_at_k[key] = ndcg_at_k(retrieved, relevant, k=k)
                        retrieval_scores.hit_rate_at_k[key] = hit_rate_at_k(retrieved, relevant, k=k)
                    retrieval_scores.mrr = mean_reciprocal_rank(retrieved, relevant)
                pipe_scores.retrieval_success = 1.0 if not result.refused else 0.0
                pipe_scores.retrieval = retrieval_scores
                pipe_scores.latency_ms = (time.perf_counter() - start) * 1000
                return retrieval_scores, pipe_scores, retrieved_ids, pipe_scores.latency_ms or 0
        except Exception as exc:
            logger.exception("Retrieval pipeline test failed", test_id=tc.id, error=str(exc))
            pipe_scores.retrieval_success = 0.0
            pipe_scores.latency_ms = (time.perf_counter() - start) * 1000
            return retrieval_scores, pipe_scores, [], pipe_scores.latency_ms or 0

    async def run(self, dataset: EvalDataset) -> EvalReport:
        start = time.perf_counter()
        results: list[TestResult] = []
        for tc in dataset.pipeline_tests:
            t0 = time.perf_counter()
            pipeline_scores: PipelineScores | None = None
            retrieval_scores = RetrievalScores()
            retrieved_ids: list[Any] = []
            errors: list[str] = []
            metric_values: dict[str, float] = {}
            try:
                if tc.pipeline == PipelineType.INGESTION:
                    pipeline_scores = await self._run_ingestion(tc)
                    metric_values["pipeline_ingestion_success_rate"] = pipeline_scores.ingestion_success or 0.0
                    metric_values["pipeline_ingestion_latency"] = pipeline_scores.latency_ms or 0.0
                    retrieved_ids = []
                elif tc.pipeline == PipelineType.RETRIEVAL or tc.pipeline == PipelineType.E2E:
                    retrieval_scores, pipeline_scores, retrieved_ids, latency = await self._run_retrieval(tc, dataset)
                    metric_values["pipeline_retrieval_success_rate"] = pipeline_scores.retrieval_success or 0.0
                    metric_values["pipeline_retrieval_latency"] = latency
                    # also aggregate retrieval metrics
                    for k, v in retrieval_scores.recall_at_k.items():
                        metric_values[f"retrieval_recall{k}"] = v
                    if retrieval_scores.mrr is not None:
                        metric_values["retrieval_mrr"] = retrieval_scores.mrr
                    for k, v in retrieval_scores.ndcg_at_k.items():
                        metric_values[f"retrieval_ndcg{k}"] = v
                    for k, v in retrieval_scores.hit_rate_at_k.items():
                        metric_values[f"retrieval_hit_rate{k}"] = v
                else:
                    errors.append(f"Unknown pipeline {tc.pipeline}")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            latency = (time.perf_counter() - t0) * 1000
            results.append(
                TestResult(
                    test_id=tc.id,
                    query=tc.query or "",
                    pipeline=tc.pipeline.value if hasattr(tc.pipeline, "value") else str(tc.pipeline),
                    tags=list(tc.tags),
                    latency_ms=latency,
                    retrieved_chunk_ids=retrieved_ids,  # type: ignore[arg-type]
                    retrieval_scores=retrieval_scores,
                    pipeline_scores=pipeline_scores,
                    metric_values=metric_values,
                    errors=errors,
                    metadata=dict(tc.metadata),
                )
            )

        duration = (time.perf_counter() - start) * 1000
        aggregates = self._build_aggregates(results, dataset)
        summary = EvalSummary(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            kind=EvalKind.PIPELINE,
            total_tests=len(dataset.pipeline_tests),
            passed_tests=sum(1 for r in results if not r.errors),
            failed_tests=sum(1 for r in results if r.errors),
            total_duration_ms=duration,
            aggregates=aggregates,
            per_test=results,
        )
        return EvalReport(summary=summary, dataset=dataset, config_snapshot=self.config.model_dump(mode="json"))

    def _build_aggregates(self, results: list[TestResult], dataset: EvalDataset) -> list[MetricAggregate]:
        all_keys: set[str] = set()
        for r in results:
            all_keys.update(r.metric_values.keys())
        for m in dataset.metrics:
            all_keys.add(m)
        aggs: list[MetricAggregate] = []
        for key in sorted(all_keys):
            vals = [r.metric_values.get(key, 0.0) for r in results if key in r.metric_values]
            if not vals and key in dataset.metrics:
                vals = [0.0] * len(results)
            if not vals:
                continue
            agg = _aggregate(vals)
            agg.metric = key
            aggs.append(agg)
        return aggs

    async def close(self) -> None:
        await self.vector_store.close()
