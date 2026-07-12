"""Pydantic models for raggit API surface."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    """Lifecycle status of a document in the index."""

    PENDING = "pending"
    INDEXING = "indexing"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class SourceType(StrEnum):
    """Supported document source types."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"


class QueryRewriteMode(StrEnum):
    """Query rewriting strategy for sparse or ambiguous queries."""

    NONE = "none"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"


class Document(BaseModel):
    """Public representation of an indexed document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_type: SourceType
    source_uri: str
    filename: str
    content_hash: str | None = None
    status: DocumentStatus
    error_message: str | None = None
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class Chunk(BaseModel):
    """Public representation of a document chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    raw_content: str
    cleaned_content: str
    word_count: int | None = None
    embedding_model: str | None = None
    vector_id: UUID | None = None
    # Hierarchical / location metadata
    parent_chunk_index: int | None = None
    section_title: str | None = None
    page_number: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    content_hash: str | None = None
    # Sequential sibling links for relevance-chain retrieval
    prev_chunk_id: UUID | None = None
    next_chunk_id: UUID | None = None
    # Denormalized document fields for citations / filters
    source_uri: str | None = None
    filename: str | None = None
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class Citation(BaseModel):
    """Source citation for an answer span or retrieved chunk."""

    chunk_id: UUID
    document_id: UUID
    source_uri: str | None = None
    filename: str | None = None
    chunk_index: int
    page_number: int | None = None
    section_title: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    score: float | None = None
    excerpt: str | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval pipeline."""

    chunk: Chunk
    score: float
    rank_bm25: int | None = None
    rank_semantic: int | None = None
    rank_rerank: int | None = None
    citation: Citation | None = None


class QueryFilters(BaseModel):
    """Optional metadata filters applied during retrieval."""

    source_uri_prefix: str | None = None
    filename_prefix: str | None = None
    tenant_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    document_ids: list[UUID] = Field(default_factory=list)
    created_after: datetime | None = None
    created_before: datetime | None = None


class QueryResult(BaseModel):
    """Result of a user query."""

    query: str
    sanitized_keywords: list[str]
    chunks: list[RetrievedChunk]
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    grounded: bool | None = None
    rewritten_queries: list[str] = Field(default_factory=list)
    total_chunks_considered: int = Field(..., description="Total chunks in the index")


class StorageConfig(BaseModel):
    """Configuration for a storage backend."""

    source_type: SourceType
    uri: str
    bucket: str | None = None
    container: str | None = None
    prefix: str | None = None
    region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    gcs_service_account_path: str | None = None
    azure_connection_string: str | None = None
    poll_interval_seconds: int = 30


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str = "openai"  # openai, ollama
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding provider."""

    provider: str = "sentence-transformers"  # sentence-transformers, openai
    model: str = "BAAI/bge-small-en-v1.5"
    model_version: str | None = None  # optional pinned revision / version tag
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 32
    max_concurrency: int = 4
    circuit_breaker_failures: int = 5
    circuit_breaker_reset_seconds: float = 60.0


class RerankerConfig(BaseModel):
    """Cross-encoder reranker configuration."""

    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 20  # candidates to rerank after RRF


class RetrievalConfig(BaseModel):
    """Hybrid retrieval knobs."""

    min_top_k: int = 5
    max_top_k: int = 50
    top_k_ratio: float = 0.01
    rrf_k: int = 60
    rrf_weight_bm25: float = 1.0
    rrf_weight_semantic: float = 1.0
    min_score: float | None = None  # drop chunks below this after fusion/rerank
    parent_window: int = 0  # expand +/-N sibling chunks around hits
    query_rewrite: QueryRewriteMode = QueryRewriteMode.NONE
    multi_query_count: int = 3
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    # Relevance-chain traversal: walk forward from a hit while relevance stays high.
    traversal_enabled: bool = True
    traversal_max_steps: int = 10
    traversal_min_score: float = 0.01
    traversal_drop_ratio: float = 0.5


class SafetyConfig(BaseModel):
    """Answer quality and safety settings."""

    refuse_on_empty: bool = True
    refuse_on_low_score: bool = True
    min_answer_score: float | None = 0.01
    groundedness_check: bool = True
    pii_redaction: bool = False
    prompt_injection_hardening: bool = True


class ChunkingConfig(BaseModel):
    """Chunking behaviour."""

    max_words_per_chunk: int = 1024  # soft upper bound; ignored when preserve_sections is True
    chunk_overlap_words: int = 0  # overlap for final word-window fallback
    dedup_enabled: bool = True
    dedup_similarity: float = 0.92
    format_aware: bool = True
    preserve_sections: bool = True  # keep detected sections whole instead of splitting at max_words


class RAGConfig(BaseModel):
    """Top-level runtime configuration for raggit."""

    database_url: str = "postgresql+asyncpg://raggit:raggit@localhost:5432/raggit"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "raggit_chunks"
    qdrant_api_key: str | None = None
    log_level: str = "INFO"
    # Back-compat flat chunk fields (mirrored into chunking as words)
    chunk_size: int = 1024
    chunk_overlap: int = 0
    min_top_k: int = 5
    max_top_k: int = 50
    top_k_ratio: float = 0.01
    rrf_k: int = 60
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig | None = None
    default_tenant_id: str | None = None
