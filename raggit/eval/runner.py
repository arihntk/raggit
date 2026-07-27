"""Evaluation runner for raggit.

The runner executes a dataset of test cases against the live retrieval and
augmentation pipelines and computes requested metrics.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from raggit.api.models import QueryResult, RAGConfig
from raggit.core.audit import log_event
from raggit.core.logging import get_logger
from raggit.db.repository import ChunkRepository, DocumentRepository, EmbeddingCollectionRepository
from raggit.db.session import AsyncSessionLocal
from raggit.db.vector import VectorStore
from raggit.eval.judge import judge_answer, judge_groundedness
from raggit.eval.metrics import (
    contains_answer,
    cosine_similarity,
    exact_match,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    refusal_accuracy,
)
from raggit.eval.models import (
    AnswerScores,
    EvalDataset,
    EvalReport,
    EvalSummary,
    MetricAggregate,
    MetricName,
    RetrievalScores,
    TestResult,
)
from raggit.ingestion.embedder import create_embedder
from raggit.llm.augmenter import augment_and_answer
from raggit.llm.factory import create_llm
from raggit.retrieval.engine import RetrievalEngine

logger = get_logger("raggit.eval.runner")


class EvaluationError(Exception):
    """Raised when an evaluation run cannot complete."""


def _aggregate(values: list[float]) -> MetricAggregate:
    """Compute aggregate statistics for a list of numeric scores."""
    if not values:
        return MetricAggregate(metric="", values=[])
    sorted_values = sorted(values)
    n = len(sorted_values)
    median = (
        sorted_values[n // 2]
        if n % 2 == 1
        else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )
    return MetricAggregate(
        metric="",
        mean=sum(values) / n,
        min=min(values),
        max=max(values),
        median=median,
        values=values,
    )


def _format_context(result: QueryResult) -> str:
    """Join retrieved chunks into a single context string for groundedness checks."""
    return "\n\n".join(
        r.chunk.cleaned_content for r in result.chunks
    )


class EvalRunner:
    """Run an evaluation dataset against the configured raggit system."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embedder = create_embedder(config.embedding)
        self.vector_store = VectorStore(config)
        self.llm: Any | None = None
        llm_ready = config.llm.provider == "ollama" or bool(config.llm.api_key)
        if config.llm.provider and llm_ready:
            self.llm = create_llm(config.llm)

    async def _run_single(
        self,
        test: Any,
        dataset: EvalDataset,
    ) -> TestResult:
        """Run a single test case and compute all requested metrics."""
        from raggit.api.models import QueryFilters

        result = TestResult(
            test_id=test.id,
            query=test.query,
            tags=list(test.tags or []),
        )

        start = time.perf_counter()
        try:
            async with AsyncSessionLocal() as session, session.begin():
                active = await EmbeddingCollectionRepository(session).get_active()
                if active is not None:
                    self.vector_store.set_collection(active.name)

                filters: QueryFilters = test.filters or QueryFilters()
                engine = RetrievalEngine(
                    embedder=self.embedder,
                    vector_store=self.vector_store,
                    chunk_repo=ChunkRepository(session),
                    document_repo=DocumentRepository(session),
                    config=self.config,
                    llm=self.llm,
                )

                query_result = await engine.retrieve(test.query, filters=filters)

                if self.llm is not None and not test.expected_refusal:
                    query_result = await augment_and_answer(
                        self.llm, query_result, safety=self.config.safety
                    )

                result.latency_ms = (time.perf_counter() - start) * 1000
                result.retrieved_chunk_ids = [
                    UUID(str(r.chunk.id)) for r in query_result.chunks
                ]
                result.answer = query_result.answer
                result.refused = query_result.refused
                result.refusal_reason = query_result.refusal_reason

                result.retrieval_scores = self._score_retrieval(
                    query_result, test, dataset
                )
                result.answer_scores = await self._score_answer(
                    query_result, test, dataset, result
                )

                await log_event(
                    session,
                    level="INFO",
                    component="raggit.eval.runner",
                    message="Evaluation test completed",
                    extra={
                        "test_id": test.id,
                        "query": test.query,
                        "refused": result.refused,
                        "latency_ms": result.latency_ms,
                    },
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            result.latency_ms = elapsed
            result.errors.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Evaluation test failed", test_id=test.id, error=str(exc))

        return result

    def _score_retrieval(
        self,
        query_result: QueryResult,
        test: Any,
        dataset: EvalDataset,
    ) -> RetrievalScores:
        """Compute retrieval metrics for a test case."""
        scores = RetrievalScores()
        if not test.expected_chunk_ids:
            return scores

        retrieved = [UUID(str(r.chunk.id)) for r in query_result.chunks]
        relevant = [UUID(str(cid)) for cid in test.expected_chunk_ids]
        requested = {MetricName(m) for m in dataset.metrics}

        for k in dataset.k_values:
            key = f"@{k}"
            if MetricName.RETRIEVAL_RECALL_AT_K in requested:
                scores.recall_at_k[key] = recall_at_k(retrieved, relevant, k=k)
            if MetricName.RETRIEVAL_PRECISION_AT_K in requested:
                scores.precision_at_k[key] = precision_at_k(retrieved, relevant, k=k)
            if MetricName.RETRIEVAL_NDCG_AT_K in requested:
                scores.ndcg_at_k[key] = ndcg_at_k(retrieved, relevant, k=k)
            if MetricName.RETRIEVAL_HIT_RATE_AT_K in requested:
                scores.hit_rate_at_k[key] = hit_rate_at_k(retrieved, relevant, k=k)

        if MetricName.RETRIEVAL_MRR in requested:
            scores.mrr = mean_reciprocal_rank(retrieved, relevant)

        return scores

    async def _score_answer(
        self,
        query_result: QueryResult,
        test: Any,
        dataset: EvalDataset,
        result: TestResult,
    ) -> AnswerScores:
        """Compute answer metrics for a test case."""
        scores = AnswerScores()
        requested = {MetricName(m) for m in dataset.metrics}
        answer = query_result.answer

        if MetricName.ANSWER_EXACT_MATCH in requested and test.expected_answer:
            scores.exact_match = exact_match(answer, test.expected_answer)

        if MetricName.ANSWER_CONTAINS in requested and test.expected_answer:
            scores.contains = contains_answer(answer, test.expected_answer)

        if (
            MetricName.ANSWER_SEMANTIC_SIMILARITY in requested
            and test.expected_answer
            and answer
        ):
            try:
                embeddings = await self.embedder.embed([answer, test.expected_answer])
                scores.semantic_similarity = cosine_similarity(
                    embeddings[0], embeddings[1]
                )
            except Exception as exc:
                logger.warning(
                    "Semantic similarity failed", test_id=test.id, error=str(exc)
                )

        if (
            MetricName.ANSWER_LLM_JUDGE in requested
            and test.expected_answer
            and self.llm is not None
        ):
            try:
                judge_score, reasoning = await judge_answer(
                    self.llm,
                    question=test.query,
                    expected_answer=test.expected_answer,
                    actual_answer=answer or "",
                )
                scores.llm_judge_score = judge_score
                scores.llm_judge_reasoning = reasoning
            except Exception as exc:
                logger.warning("LLM judge failed", test_id=test.id, error=str(exc))

        if MetricName.GROUNDEDNESS in requested and answer and query_result.chunks:
            if self.llm is not None:
                try:
                    scores.groundedness = await judge_groundedness(
                        self.llm,
                        question=test.query,
                        answer=answer,
                        context=_format_context(query_result),
                    )
                except Exception as exc:
                    logger.warning(
                        "Groundedness judge failed", test_id=test.id, error=str(exc)
                    )
            else:
                scores.groundedness = query_result.grounded

        if MetricName.REFUSAL_ACCURACY in requested:
            result.refusal_accuracy = refusal_accuracy(
                query_result.refused, test.expected_refusal
            )

        return scores

    async def run(self, dataset: EvalDataset) -> EvalReport:
        """Run all test cases in the dataset and produce a report."""
        results: list[TestResult] = []
        start = time.perf_counter()

        for test in dataset.tests:
            result = await self._run_single(test, dataset)
            results.append(result)

        duration_ms = (time.perf_counter() - start) * 1000
        aggregates = self._build_aggregates(results, dataset)
        summary = EvalSummary(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            total_tests=len(dataset.tests),
            passed_tests=sum(1 for r in results if not r.errors),
            failed_tests=sum(1 for r in results if r.errors),
            total_duration_ms=duration_ms,
            aggregates=aggregates,
            per_test=results,
        )

        return EvalReport(
            summary=summary,
            dataset=dataset,
            config_snapshot=self.config.model_dump(mode="json"),
        )

    def _build_aggregates(
        self, results: list[TestResult], dataset: EvalDataset
    ) -> list[MetricAggregate]:
        """Compute aggregate scores across all test cases."""
        requested = {MetricName(m) for m in dataset.metrics}
        aggregates: list[MetricAggregate] = []

        def _add(name: str, values: list[float]) -> None:
            if values:
                agg = _aggregate(values)
                agg.metric = name
                aggregates.append(agg)

        for k in dataset.k_values:
            key = f"@{k}"
            if MetricName.RETRIEVAL_RECALL_AT_K in requested:
                _add(
                    f"retrieval_recall{key}",
                    [r.retrieval_scores.recall_at_k.get(key, 0.0) for r in results],
                )
            if MetricName.RETRIEVAL_PRECISION_AT_K in requested:
                _add(
                    f"retrieval_precision{key}",
                    [r.retrieval_scores.precision_at_k.get(key, 0.0) for r in results],
                )
            if MetricName.RETRIEVAL_NDCG_AT_K in requested:
                _add(
                    f"retrieval_ndcg{key}",
                    [r.retrieval_scores.ndcg_at_k.get(key, 0.0) for r in results],
                )
            if MetricName.RETRIEVAL_HIT_RATE_AT_K in requested:
                _add(
                    f"retrieval_hit_rate{key}",
                    [r.retrieval_scores.hit_rate_at_k.get(key, 0.0) for r in results],
                )

        if MetricName.RETRIEVAL_MRR in requested:
            _add(
                "retrieval_mrr",
                [r.retrieval_scores.mrr or 0.0 for r in results],
            )

        if MetricName.ANSWER_EXACT_MATCH in requested:
            _add(
                "answer_exact_match",
                [1.0 if r.answer_scores.exact_match else 0.0 for r in results],
            )

        if MetricName.ANSWER_CONTAINS in requested:
            _add(
                "answer_contains",
                [1.0 if r.answer_scores.contains else 0.0 for r in results],
            )

        if MetricName.ANSWER_SEMANTIC_SIMILARITY in requested:
            _add(
                "answer_semantic_similarity",
                [
                    r.answer_scores.semantic_similarity or 0.0
                    for r in results
                    if r.answer_scores.semantic_similarity is not None
                ],
            )

        if MetricName.ANSWER_LLM_JUDGE in requested:
            _add(
                "answer_llm_judge",
                [
                    r.answer_scores.llm_judge_score or 0.0
                    for r in results
                    if r.answer_scores.llm_judge_score is not None
                ],
            )

        if MetricName.GROUNDEDNESS in requested:
            _add(
                "groundedness",
                [1.0 if r.answer_scores.groundedness else 0.0 for r in results],
            )

        if MetricName.REFUSAL_ACCURACY in requested:
            _add(
                "refusal_accuracy",
                [r.refusal_accuracy or 0.0 for r in results],
            )

        if MetricName.LATENCY_MS in requested:
            _add("latency_ms", [r.latency_ms or 0.0 for r in results])

        return aggregates

    async def close(self) -> None:
        """Release resources held by the runner."""
        await self.vector_store.close()
