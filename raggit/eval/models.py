"""Pydantic models for the raggit evaluation framework."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from raggit.api.models import QueryFilters


class MetricName(StrEnum):
    """Built-in evaluation metric names."""

    RETRIEVAL_RECALL_AT_K = "retrieval_recall@k"
    RETRIEVAL_PRECISION_AT_K = "retrieval_precision@k"
    RETRIEVAL_MRR = "retrieval_mrr"
    RETRIEVAL_NDCG_AT_K = "retrieval_ndcg@k"
    RETRIEVAL_HIT_RATE_AT_K = "retrieval_hit_rate@k"
    ANSWER_EXACT_MATCH = "answer_exact_match"
    ANSWER_CONTAINS = "answer_contains"
    ANSWER_SEMANTIC_SIMILARITY = "answer_semantic_similarity"
    ANSWER_LLM_JUDGE = "answer_llm_judge"
    GROUNDEDNESS = "groundedness"
    LATENCY_MS = "latency_ms"
    REFUSAL_ACCURACY = "refusal_accuracy"


DEFAULT_METRICS: list[str] = [
    MetricName.RETRIEVAL_RECALL_AT_K,
    MetricName.RETRIEVAL_MRR,
    MetricName.RETRIEVAL_HIT_RATE_AT_K,
    MetricName.ANSWER_SEMANTIC_SIMILARITY,
    MetricName.GROUNDEDNESS,
    MetricName.LATENCY_MS,
]


class TestCase(BaseModel):
    """A single evaluation test case."""

    __test__ = False
    model_config = ConfigDict(extra="allow")

    id: str
    query: str
    filters: QueryFilters = Field(default_factory=QueryFilters)
    expected_chunk_ids: list[UUID] = Field(default_factory=list)
    expected_answer: str | None = None
    expected_refusal: bool = False
    judgement_criteria: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A collection of test cases for evaluating raggit."""

    name: str = "raggit-eval"
    description: str | None = None
    version: str = "1.0.0"
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    k_values: list[int] = Field(default_factory=lambda: [5, 10])
    tests: list[TestCase]


class RetrievalScores(BaseModel):
    """Per-test retrieval evaluation scores."""

    recall_at_k: dict[str, float] = Field(default_factory=dict)
    precision_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float | None = None
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    hit_rate_at_k: dict[str, float] = Field(default_factory=dict)


class AnswerScores(BaseModel):
    """Per-test answer evaluation scores."""

    exact_match: bool | None = None
    contains: bool | None = None
    semantic_similarity: float | None = None
    llm_judge_score: float | None = None
    llm_judge_reasoning: str | None = None
    groundedness: bool | None = None


class TestResult(BaseModel):
    """Result of running a single test case."""

    __test__ = False
    test_id: str
    query: str
    tags: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    retrieved_chunk_ids: list[UUID] = Field(default_factory=list)
    retrieval_scores: RetrievalScores = Field(default_factory=RetrievalScores)
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    answer_scores: AnswerScores = Field(default_factory=AnswerScores)
    refusal_accuracy: float | None = None
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricAggregate(BaseModel):
    """Aggregate score for a metric across the dataset."""

    metric: str
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    values: list[float] = Field(default_factory=list)


class EvalSummary(BaseModel):
    """Summary of an evaluation run."""

    dataset_name: str
    dataset_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration_ms: float
    run_at: datetime = Field(default_factory=datetime.now)
    aggregates: list[MetricAggregate] = Field(default_factory=list)
    per_test: list[TestResult] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Full evaluation report including summary and raw results."""

    summary: EvalSummary
    dataset: EvalDataset
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
