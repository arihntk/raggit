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
    token_count: int | None = None
    embedding_model: str | None = None
    vector_id: UUID | None = None
    created_at: datetime


class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval pipeline."""

    chunk: Chunk
    score: float
    rank_bm25: int | None = None
    rank_semantic: int | None = None


class QueryResult(BaseModel):
    """Result of a user query."""

    query: str
    sanitized_keywords: list[str]
    chunks: list[RetrievedChunk]
    answer: str | None = None
    total_chunks_considered: int = Field(..., description="Total chunks in the index")


class StorageConfig(BaseModel):
    """Configuration for a storage backend."""

    source_type: SourceType
    uri: str
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
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 32


class RAGConfig(BaseModel):
    """Top-level runtime configuration for raggit."""

    database_url: str = "postgresql+asyncpg://raggit:raggit@localhost:5432/raggit"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "raggit_chunks"
    qdrant_api_key: str | None = None
    log_level: str = "INFO"
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_top_k: int = 5
    max_top_k: int = 50
    top_k_ratio: float = 0.01
    rrf_k: int = 60
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig | None = None
