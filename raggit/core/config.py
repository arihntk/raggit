"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from raggit.api.models import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    RAGConfig,
    RetrievalConfig,
    SourceType,
    StorageConfig,
)


class Settings(BaseSettings):
    """Settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://raggit:raggit@localhost:5432/raggit"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "raggit_chunks"
    qdrant_api_key: str | None = None

    # Logging
    log_level: str = "INFO"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 0
    chunking_dedup_enabled: bool = True
    chunking_dedup_similarity: float = 0.92
    chunking_format_aware: bool = True
    chunking_preserve_sections: bool = True

    # Retrieval
    min_top_k: int = 5
    max_top_k: int = 50
    top_k_ratio: float = 0.01
    rrf_k: int = 60
    retrieval_traversal_enabled: bool = True
    retrieval_traversal_max_steps: int = 10
    retrieval_traversal_min_score: float = 0.01
    retrieval_traversal_drop_ratio: float = 0.5

    # Embedding
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_batch_size: int = 32

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # Storage
    storage_source_type: str = "local"
    storage_uri: str = "./data/documents"
    storage_bucket: str | None = None
    storage_container: str | None = None
    storage_prefix: str | None = None
    storage_region: str | None = None
    storage_aws_access_key_id: str | None = None
    storage_aws_secret_access_key: str | None = None
    storage_gcs_service_account_path: str | None = None
    storage_azure_connection_string: str | None = None
    storage_poll_interval_seconds: int = 30

    @property
    def rag_config(self) -> RAGConfig:
        """Build the public RAGConfig from settings."""
        return RAGConfig(
            database_url=self.database_url,
            qdrant_url=self.qdrant_url,
            qdrant_collection=self.qdrant_collection,
            qdrant_api_key=self.qdrant_api_key,
            log_level=self.log_level,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunking=ChunkingConfig(
                max_words_per_chunk=self.chunk_size,
                chunk_overlap_words=self.chunk_overlap,
                dedup_enabled=self.chunking_dedup_enabled,
                dedup_similarity=self.chunking_dedup_similarity,
                format_aware=self.chunking_format_aware,
                preserve_sections=self.chunking_preserve_sections,
            ),
            min_top_k=self.min_top_k,
            max_top_k=self.max_top_k,
            top_k_ratio=self.top_k_ratio,
            rrf_k=self.rrf_k,
            retrieval=RetrievalConfig(
                min_top_k=self.min_top_k,
                max_top_k=self.max_top_k,
                top_k_ratio=self.top_k_ratio,
                rrf_k=self.rrf_k,
                traversal_enabled=self.retrieval_traversal_enabled,
                traversal_max_steps=self.retrieval_traversal_max_steps,
                traversal_min_score=self.retrieval_traversal_min_score,
                traversal_drop_ratio=self.retrieval_traversal_drop_ratio,
            ),
            embedding=EmbeddingConfig(
                provider=self.embedding_provider,
                model=self.embedding_model,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
                batch_size=self.embedding_batch_size,
            ),
            llm=LLMConfig(
                provider=self.llm_provider,
                model=self.llm_model,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            ),
            storage=StorageConfig(
                source_type=SourceType(self.storage_source_type),
                uri=self.storage_uri,
                bucket=self.storage_bucket,
                container=self.storage_container,
                prefix=self.storage_prefix,
                region=self.storage_region,
                aws_access_key_id=self.storage_aws_access_key_id,
                aws_secret_access_key=self.storage_aws_secret_access_key,
                gcs_service_account_path=self.storage_gcs_service_account_path,
                azure_connection_string=self.storage_azure_connection_string,
                poll_interval_seconds=self.storage_poll_interval_seconds,
            ),
        )


def config_file_path() -> Path:
    """Return the path to the user config directory."""
    path = Path.home() / ".config" / "raggit"
    path.mkdir(parents=True, exist_ok=True)
    return path / "raggit.env"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    env_path = config_file_path()
    env_file = env_path if env_path.exists() else Path(".env")
    # pydantic-settings accepts _env_file at runtime; mypy stubs omit it.
    return Settings(_env_file=env_file)  # type: ignore[call-arg]
