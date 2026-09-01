"""Pydantic models for the raggit evaluation framework.

Three-tier design
-----------------
- **Component**: isolated unit tests for each core primitive (parser,
  chunker, cleaner, PII, injection, sanitizer, embedder, RRF, reranker,
  safety, storage, watcher). Synthetic data, no DB required for many.
- **Pipeline**: ingestion pipeline (parse→chunk→clean→embed) and retrieval
  pipeline (sanitize→rewrite→BM25/semantic→RRF→rerank→threshold→parent→
  traversal) evaluated as a cohesive chain with real DB/vector store.
- **System**: end-to-end (ingestion + retrieval + LLM augmentation) as
  before, via :class:`EvalRunner`.

Every feature of raggit maps to at least one metric, so coverage is exhaustive.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from raggit.api.models import QueryFilters


class MetricName(StrEnum):
    """Built-in evaluation metric names – exhaustive per-feature."""

    # -- Retrieval core (existing, kept for backwards compat) --
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

    # -- Component: parser --
    PARSER_PARSE_SUCCESS = "parser_parse_success"
    PARSER_TEXT_FIDELITY = "parser_text_fidelity"
    PARSER_PAGE_PRESERVATION = "parser_page_preservation"
    PARSER_HTML_STRIPPING = "parser_html_stripping"

    # -- Component: chunker --
    CHUNKER_SECTION_PRESERVATION = "chunker_section_preservation"
    CHUNKER_PAGE_PRESERVATION = "chunker_page_preservation"
    CHUNKER_FUNCTION_BOUNDARY = "chunker_function_boundary"
    CHUNKER_DEDUP_EFFECTIVENESS = "chunker_dedup_effectiveness"
    CHUNKER_WORD_COUNT_ACCURACY = "chunker_word_count_accuracy"
    CHUNKER_OVERLAP_CORRECTNESS = "chunker_overlap_correctness"
    CHUNKER_FORMAT_AWARE = "chunker_format_aware"

    # -- Component: cleaner --
    CLEANER_UNICODE_NORMALIZATION = "cleaner_unicode_normalization"
    CLEANER_HYPHENATION_FIX = "cleaner_hyphenation_fix"
    CLEANER_WHITESPACE_COLLAPSE = "cleaner_whitespace_collapse"
    CLEANER_EFFECTIVENESS = "cleaner_effectiveness"

    # -- Component: PII / injection --
    PII_REDACTION_RECALL = "pii_redaction_recall"
    PII_REDACTION_PRECISION = "pii_redaction_precision"
    PII_REDACTION_F1 = "pii_redaction_f1"
    INJECTION_HARDENING_RECALL = "injection_hardening_recall"
    INJECTION_HARDENING_PRECISION = "injection_hardening_precision"

    # -- Component: sanitizer / rewrite --
    SANITIZER_KEYWORD_RECALL = "sanitizer_keyword_recall"
    SANITIZER_STOPWORD_REMOVAL_RATE = "sanitizer_stopword_removal_rate"
    REWRITE_MULTI_QUERY_GAIN = "rewrite_multi_query_gain"
    REWRITE_HYDE_GAIN = "rewrite_hyde_gain"

    # -- Component: retrieval primitives --
    BM25_RECALL_AT_K = "bm25_recall@k"
    SEMANTIC_RECALL_AT_K = "semantic_recall@k"
    HYBRID_RECALL_AT_K = "hybrid_recall@k"
    RRF_FUSION_QUALITY = "rrf_fusion_quality"
    RERANKER_GAIN = "reranker_gain"
    THRESHOLD_RETENTION_RATE = "threshold_retention_rate"
    PARENT_WINDOW_GAIN = "parent_window_gain"
    TRAVERSAL_PRECISION = "traversal_precision"

    # -- Component: safety / augmenter --
    SAFETY_GROUNDEDNESS_ACCURACY = "safety_groundedness_accuracy"
    SAFETY_REFUSAL_F1 = "safety_refusal_f1"
    AUGMENTER_CITATION_COVERAGE = "augmenter_citation_coverage"
    AUGMENTER_CONTEXT_ISOLATION = "augmenter_context_isolation"

    # -- Component: storage / watcher --
    STORAGE_PATH_TRAVERSAL_BLOCK_RATE = "storage_path_traversal_block_rate"
    STORAGE_WATCH_LATENCY = "storage_watch_latency"
    WATCHER_DEBOUNCE_ACCURACY = "watcher_debounce_accuracy"
    WATCHER_FILE_STABILITY = "watcher_file_stability"

    # -- Component: embedder --
    EMBEDDER_COSINE_ACCURACY = "embedder_cosine_accuracy"
    EMBEDDER_LATENCY = "embedder_latency"

    # -- Pipeline --
    PIPELINE_INGESTION_SUCCESS_RATE = "pipeline_ingestion_success_rate"
    PIPELINE_INGESTION_LATENCY = "pipeline_ingestion_latency"
    PIPELINE_RETRIEVAL_SUCCESS_RATE = "pipeline_retrieval_success_rate"
    PIPELINE_RETRIEVAL_LATENCY = "pipeline_retrieval_latency"

    # -- System --
    SYSTEM_E2E_SUCCESS = "system_e2e_success"
    SYSTEM_CITATION_PRECISION = "system_citation_precision"
    SYSTEM_CITATION_RECALL = "system_citation_recall"
    SYSTEM_HALLUCINATION_RATE = "system_hallucination_rate"
    SYSTEM_LATENCY_P50 = "system_latency_p50"
    SYSTEM_LATENCY_P95 = "system_latency_p95"
    SYSTEM_AUDIT_COVERAGE = "system_audit_coverage"
    FILTER_TENANT_ACCURACY = "filter_tenant_accuracy"
    FILTER_TAG_ACCURACY = "filter_tag_accuracy"
    FILTER_PREFIX_ACCURACY = "filter_prefix_accuracy"
    MCP_TOOL_SUCCESS = "mcp_tool_success"


DEFAULT_METRICS: list[str] = [
    MetricName.RETRIEVAL_RECALL_AT_K,
    MetricName.RETRIEVAL_MRR,
    MetricName.RETRIEVAL_HIT_RATE_AT_K,
    MetricName.ANSWER_SEMANTIC_SIMILARITY,
    MetricName.GROUNDEDNESS,
    MetricName.LATENCY_MS,
]

# Every feature should be measurable – exhaustive default for `raggit eval --kind all`
ALL_METRICS: list[str] = [m.value for m in MetricName]

# Tier presets
COMPONENT_METRICS: list[str] = [
    MetricName.PARSER_PARSE_SUCCESS,
    MetricName.PARSER_TEXT_FIDELITY,
    MetricName.CHUNKER_SECTION_PRESERVATION,
    MetricName.CHUNKER_DEDUP_EFFECTIVENESS,
    MetricName.CLEANER_EFFECTIVENESS,
    MetricName.PII_REDACTION_F1,
    MetricName.INJECTION_HARDENING_RECALL,
    MetricName.SANITIZER_KEYWORD_RECALL,
    MetricName.RRF_FUSION_QUALITY,
    MetricName.RERANKER_GAIN,
    MetricName.SAFETY_GROUNDEDNESS_ACCURACY,
    MetricName.EMBEDDER_COSINE_ACCURACY,
]

PIPELINE_METRICS: list[str] = [
    MetricName.PIPELINE_INGESTION_SUCCESS_RATE,
    MetricName.PIPELINE_RETRIEVAL_SUCCESS_RATE,
    MetricName.RETRIEVAL_RECALL_AT_K,
    MetricName.RETRIEVAL_NDCG_AT_K,
    MetricName.PARENT_WINDOW_GAIN,
    MetricName.TRAVERSAL_PRECISION,
]

SYSTEM_METRICS: list[str] = [
    MetricName.SYSTEM_E2E_SUCCESS,
    MetricName.SYSTEM_CITATION_PRECISION,
    MetricName.SYSTEM_HALLUCINATION_RATE,
    MetricName.ANSWER_CONTAINS,
    MetricName.GROUNDEDNESS,
    MetricName.REFUSAL_ACCURACY,
    MetricName.FILTER_TENANT_ACCURACY,
    MetricName.LATENCY_MS,
]


class EvalKind(StrEnum):
    """Evaluation tier."""

    COMPONENT = "component"
    PIPELINE = "pipeline"
    SYSTEM = "system"
    ALL = "all"


class ComponentType(StrEnum):
    """Isolated component under test."""

    PARSER = "parser"
    CHUNKER = "chunker"
    CLEANER = "cleaner"
    PII = "pii"
    INJECTION = "injection"
    SANITIZER = "sanitizer"
    EMBEDDER = "embedder"
    RRF = "rrf"
    RERANKER = "reranker"
    SAFETY = "safety"
    AUGMENTER = "augmenter"
    STORAGE = "storage"
    WATCHER = "watcher"
    RETRIEVER = "retriever"  # hybrid BM25+semantic isolated


class PipelineType(StrEnum):
    """Pipeline under test."""

    INGESTION = "ingestion"
    RETRIEVAL = "retrieval"
    E2E = "e2e"  # ingestion + retrieval without LLM


class TestCase(BaseModel):
    """A single evaluation test case (system tier, backwards compatible)."""

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


class ComponentTestCase(BaseModel):
    """Test case for isolated component evaluation."""

    __test__ = False
    model_config = ConfigDict(extra="allow")

    id: str
    component: ComponentType
    # Generic input/expected – validated per-component in runner
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineTestCase(BaseModel):
    """Test case for pipeline evaluation."""

    __test__ = False
    model_config = ConfigDict(extra="allow")

    id: str
    pipeline: PipelineType
    # For ingestion pipeline: raw document bytes/text + path
    # For retrieval pipeline: query + ingestion state
    query: str | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    expected_chunk_ids: list[UUID] = Field(default_factory=list)
    expected_answer: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A collection of test cases for evaluating raggit.

    Supports three tiers via ``kind``:
    - ``system`` (default): end-to-end via :class:`EvalRunner`
    - ``component``: isolated primitives via :class:`ComponentRunner`
    - ``pipeline``: ingestion/retrieval chains via :class:`PipelineRunner`
    - ``all``: runs all tiers sequentially
    """

    name: str = "raggit-eval"
    description: str | None = None
    version: str = "1.0.0"
    kind: EvalKind = EvalKind.SYSTEM
    # Optional sub-type for component/pipeline datasets
    component: ComponentType | None = None
    pipeline: PipelineType | None = None
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    k_values: list[int] = Field(default_factory=lambda: [5, 10])
    tests: list[TestCase] = Field(default_factory=list)
    # Tier-specific test lists (preferred over generic ``tests`` for those kinds)
    component_tests: list[ComponentTestCase] = Field(default_factory=list)
    pipeline_tests: list[PipelineTestCase] = Field(default_factory=list)


class RetrievalScores(BaseModel):
    """Per-test retrieval evaluation scores."""

    recall_at_k: dict[str, float] = Field(default_factory=dict)
    precision_at_k: dict[str, float] = Field(default_factory=dict)
    mrr: float | None = None
    ndcg_at_k: dict[str, float] = Field(default_factory=dict)
    hit_rate_at_k: dict[str, float] = Field(default_factory=dict)
    # Extended per-feature retrieval scores
    bm25_recall_at_k: dict[str, float] = Field(default_factory=dict)
    semantic_recall_at_k: dict[str, float] = Field(default_factory=dict)
    hybrid_recall_at_k: dict[str, float] = Field(default_factory=dict)
    rrf_quality: float | None = None
    reranker_gain: float | None = None
    threshold_retention: float | None = None
    parent_gain: float | None = None
    traversal_precision: float | None = None


class AnswerScores(BaseModel):
    """Per-test answer evaluation scores."""

    exact_match: bool | None = None
    contains: bool | None = None
    semantic_similarity: float | None = None
    llm_judge_score: float | None = None
    llm_judge_reasoning: str | None = None
    groundedness: bool | None = None
    # System citation quality
    citation_precision: float | None = None
    citation_recall: float | None = None
    hallucination: bool | None = None


class ComponentScores(BaseModel):
    """Per-test component evaluation scores (one entry per metric)."""

    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineScores(BaseModel):
    """Per-test pipeline evaluation scores."""

    ingestion_success: float | None = None
    retrieval_success: float | None = None
    latency_ms: float | None = None
    retrieval: RetrievalScores = Field(default_factory=RetrievalScores)


class TestResult(BaseModel):
    """Result of running a single test case (any tier)."""

    __test__ = False
    test_id: str
    query: str = ""
    component: str | None = None
    pipeline: str | None = None
    tags: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    retrieved_chunk_ids: list[UUID] = Field(default_factory=list)
    retrieval_scores: RetrievalScores = Field(default_factory=RetrievalScores)
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    answer_scores: AnswerScores = Field(default_factory=AnswerScores)
    component_scores: ComponentScores | None = None
    pipeline_scores: PipelineScores | None = None
    refusal_accuracy: float | None = None
    # Generic per-metric numeric map for aggregates
    metric_values: dict[str, float] = Field(default_factory=dict)
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
    kind: EvalKind = EvalKind.SYSTEM
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
