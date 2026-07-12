"""raggit setup command: write configuration and bootstrap the system."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from alembic.config import Config
from rich.console import Console
from rich.panel import Panel

from alembic import command
from raggit.api.models import SourceType
from raggit.cli._env import write_env_file
from raggit.cli.options import config_option, source_type_option
from raggit.core.config import get_settings
from raggit.core.logging import configure_logging

app_setup = typer.Typer()
console = Console()


def _collect_env_values(**kwargs: Any) -> dict[str, Any]:
    """Map CLI option names to environment variable names."""
    mapping = {
        "database_url": "DATABASE_URL",
        "qdrant_url": "QDRANT_URL",
        "qdrant_collection": "QDRANT_COLLECTION",
        "qdrant_api_key": "QDRANT_API_KEY",
        "log_level": "LOG_LEVEL",
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "chunking_dedup_enabled": "CHUNKING_DEDUP_ENABLED",
        "chunking_dedup_similarity": "CHUNKING_DEDUP_SIMILARITY",
        "chunking_format_aware": "CHUNKING_FORMAT_AWARE",
        "chunking_preserve_sections": "CHUNKING_PRESERVE_SECTIONS",
        "min_top_k": "MIN_TOP_K",
        "max_top_k": "MAX_TOP_K",
        "top_k_ratio": "TOP_K_RATIO",
        "rrf_k": "RRF_K",
        "retrieval_parent_window": "RETRIEVAL_PARENT_WINDOW",
        "retrieval_min_score": "RETRIEVAL_MIN_SCORE",
        "retrieval_query_rewrite": "RETRIEVAL_QUERY_REWRITE",
        "retrieval_multi_query_count": "RETRIEVAL_MULTI_QUERY_COUNT",
        "retrieval_traversal_enabled": "RETRIEVAL_TRAVERSAL_ENABLED",
        "retrieval_traversal_max_steps": "RETRIEVAL_TRAVERSAL_MAX_STEPS",
        "retrieval_traversal_min_score": "RETRIEVAL_TRAVERSAL_MIN_SCORE",
        "retrieval_traversal_drop_ratio": "RETRIEVAL_TRAVERSAL_DROP_RATIO",
        "reranker_enabled": "RERANKER_ENABLED",
        "reranker_model": "RERANKER_MODEL",
        "reranker_top_n": "RERANKER_TOP_N",
        "embedding_provider": "EMBEDDING_PROVIDER",
        "embedding_model": "EMBEDDING_MODEL",
        "embedding_api_key": "EMBEDDING_API_KEY",
        "embedding_base_url": "EMBEDDING_BASE_URL",
        "embedding_batch_size": "EMBEDDING_BATCH_SIZE",
        "llm_provider": "LLM_PROVIDER",
        "llm_model": "LLM_MODEL",
        "llm_base_url": "LLM_BASE_URL",
        "llm_api_key": "LLM_API_KEY",
        "llm_temperature": "LLM_TEMPERATURE",
        "llm_max_tokens": "LLM_MAX_TOKENS",
        "storage_source_type": "STORAGE_SOURCE_TYPE",
        "storage_uri": "STORAGE_URI",
        "storage_bucket": "STORAGE_BUCKET",
        "storage_container": "STORAGE_CONTAINER",
        "storage_prefix": "STORAGE_PREFIX",
        "storage_region": "STORAGE_REGION",
        "storage_aws_access_key_id": "STORAGE_AWS_ACCESS_KEY_ID",
        "storage_aws_secret_access_key": "STORAGE_AWS_SECRET_ACCESS_KEY",
        "storage_gcs_service_account_path": "STORAGE_GCS_SERVICE_ACCOUNT_PATH",
        "storage_azure_connection_string": "STORAGE_AZURE_CONNECTION_STRING",
        "storage_poll_interval_seconds": "STORAGE_POLL_INTERVAL_SECONDS",
        "safety_refuse_on_empty": "SAFETY_REFUSE_ON_EMPTY",
        "safety_refuse_on_low_score": "SAFETY_REFUSE_ON_LOW_SCORE",
        "safety_min_answer_score": "SAFETY_MIN_ANSWER_SCORE",
        "safety_groundedness_check": "SAFETY_GROUNDEDNESS_CHECK",
        "safety_pii_redaction": "SAFETY_PII_REDACTION",
        "safety_prompt_injection_hardening": "SAFETY_PROMPT_INJECTION_HARDENING",
        "default_tenant_id": "DEFAULT_TENANT_ID",
    }
    return {env_name: kwargs[param_name] for param_name, env_name in mapping.items()}


async def _check_database(database_url: str) -> None:
    """Verify PostgreSQL connectivity."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url, future=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


async def _check_qdrant(qdrant_url: str, qdrant_api_key: str | None) -> None:
    """Verify Qdrant connectivity."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key or None)
    try:
        await client.get_collections()
    finally:
        await client.close()


async def _ensure_qdrant_collection(config: Any) -> None:
    """Create the Qdrant collection for the configured embedding model."""
    from raggit.db.vector import VectorStore
    from raggit.ingestion.embedder import create_embedder

    embedder = create_embedder(config.embedding)
    vector_size = embedder.vector_size
    vector_store = VectorStore(config)
    try:
        await vector_store.ensure_collection(vector_size)
        console.print(
            f"[green]Qdrant collection '{config.qdrant_collection}' ready "
            f"(vector size: {vector_size}).[/green]"
        )
    finally:
        await vector_store.close()


def _run_migrations() -> None:
    """Run Alembic migrations programmatically."""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


def _ensure_local_storage(uri: str) -> None:
    """Create the local document directory if needed."""
    path = Path(uri).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Local storage directory ready: {path}[/green]")


async def _bootstrap_system(
    skip_db_check: bool,
    skip_qdrant_setup: bool,
    skip_storage_setup: bool,
) -> None:
    """Run first-time system setup steps (migrations run separately)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    if not skip_db_check:
        with console.status("[bold green]Checking PostgreSQL connection..."):
            await _check_database(settings.database_url)
        console.print("[green]PostgreSQL connection OK.[/green]")

    if not skip_qdrant_setup:
        with console.status("[bold green]Checking Qdrant connection..."):
            await _check_qdrant(settings.qdrant_url, settings.qdrant_api_key)
        console.print("[green]Qdrant connection OK.[/green]")
        with console.status("[bold green]Ensuring Qdrant collection..."):
            await _ensure_qdrant_collection(settings.rag_config)

    if not skip_storage_setup and settings.storage_source_type == "local":
        _ensure_local_storage(settings.storage_uri)


def register_setup(app: typer.Typer) -> None:
    """Register the setup command with the CLI application."""

    @app.command()
    def setup(
        # System setup control
        skip_system_setup: bool = typer.Option(
            False,
            "--skip-system-setup",
            help="Only write the env file; do not bootstrap services.",
        ),
        skip_db_check: bool = typer.Option(
            False, "--skip-db-check", help="Skip the PostgreSQL connectivity check."
        ),
        skip_migrations: bool = typer.Option(
            False, "--skip-migrations", help="Skip Alembic database migrations."
        ),
        skip_qdrant_setup: bool = typer.Option(
            False, "--skip-qdrant-setup", help="Skip Qdrant connectivity and collection setup."
        ),
        skip_storage_setup: bool = typer.Option(
            False, "--skip-storage-setup", help="Skip local storage directory creation."
        ),
        # Database
        database_url: str = config_option("database_url", "PostgreSQL connection URL"),
        # Vector store
        qdrant_url: str = config_option("qdrant_url", "Qdrant URL"),
        qdrant_collection: str = config_option("qdrant_collection", "Qdrant collection name"),
        qdrant_api_key: str | None = config_option("qdrant_api_key", "Qdrant API key"),
        # Logging
        log_level: str = config_option("log_level", "Log level (DEBUG, INFO, WARNING, ERROR)"),
        # Chunking
        chunk_size: int = config_option("chunk_size", "Target chunk size in words/tokens"),
        chunk_overlap: int = config_option(
            "chunk_overlap", "Number of overlapping words/tokens between chunks"
        ),
        chunking_dedup_enabled: bool = config_option(
            "chunking_dedup_enabled", "Enable near-duplicate chunk removal"
        ),
        chunking_dedup_similarity: float = config_option(
            "chunking_dedup_similarity", "Jaccard similarity threshold for deduplication"
        ),
        chunking_format_aware: bool = config_option(
            "chunking_format_aware", "Use format-aware chunk boundaries"
        ),
        chunking_preserve_sections: bool = config_option(
            "chunking_preserve_sections",
            "Keep detected sections whole instead of splitting by chunk size",
        ),
        # Retrieval
        min_top_k: int = config_option("min_top_k", "Minimum number of retrieved chunks"),
        max_top_k: int = config_option("max_top_k", "Maximum number of retrieved chunks"),
        top_k_ratio: float = config_option(
            "top_k_ratio", "Fraction of total chunks used to scale top-k"
        ),
        rrf_k: int = config_option("rrf_k", "Reciprocal rank fusion constant"),
        retrieval_parent_window: int = config_option(
            "retrieval_parent_window", "Expand hits by +/- N sibling chunks"
        ),
        retrieval_min_score: float | None = config_option(
            "retrieval_min_score", "Drop chunks below this fusion/rerank score"
        ),
        retrieval_query_rewrite: str = config_option(
            "retrieval_query_rewrite",
            "Query rewrite mode: none, multi_query, hyde",
        ),
        retrieval_multi_query_count: int = config_option(
            "retrieval_multi_query_count", "Number of variants for multi_query rewrite"
        ),
        retrieval_traversal_enabled: bool = config_option(
            "retrieval_traversal_enabled", "Enable relevance-chain traversal"
        ),
        retrieval_traversal_max_steps: int = config_option(
            "retrieval_traversal_max_steps", "Maximum traversal steps from a hit"
        ),
        retrieval_traversal_min_score: float = config_option(
            "retrieval_traversal_min_score", "Minimum score to continue traversal"
        ),
        retrieval_traversal_drop_ratio: float = config_option(
            "retrieval_traversal_drop_ratio", "Score ratio that stops traversal"
        ),
        # Reranker
        reranker_enabled: bool = config_option(
            "reranker_enabled", "Enable cross-encoder reranking"
        ),
        reranker_model: str = config_option("reranker_model", "Cross-encoder model name"),
        reranker_top_n: int = config_option("reranker_top_n", "Number of candidates to rerank"),
        # Embedding
        embedding_provider: str = config_option(
            "embedding_provider", "Embedding provider: sentence-transformers, openai"
        ),
        embedding_model: str = config_option("embedding_model", "Embedding model name"),
        embedding_api_key: str | None = config_option("embedding_api_key", "Embedding API key"),
        embedding_base_url: str | None = config_option(
            "embedding_base_url", "OpenAI-compatible embedding base URL"
        ),
        embedding_batch_size: int = config_option(
            "embedding_batch_size", "Texts per embedding batch"
        ),
        # LLM
        llm_provider: str = config_option("llm_provider", "LLM provider: openai, ollama"),
        llm_model: str = config_option("llm_model", "LLM model name"),
        llm_base_url: str | None = config_option("llm_base_url", "OpenAI-compatible LLM base URL"),
        llm_api_key: str | None = config_option("llm_api_key", "LLM API key"),
        llm_temperature: float = config_option("llm_temperature", "LLM sampling temperature"),
        llm_max_tokens: int = config_option("llm_max_tokens", "Maximum LLM response tokens"),
        # Storage
        storage_source_type: SourceType = source_type_option("Storage backend"),
        storage_uri: str = config_option("storage_uri", "Storage URI or local path"),
        storage_bucket: str | None = config_option("storage_bucket", "S3/GCS bucket name"),
        storage_container: str | None = config_option(
            "storage_container", "Azure Blob container name"
        ),
        storage_prefix: str | None = config_option("storage_prefix", "Object key prefix"),
        storage_region: str | None = config_option("storage_region", "S3 region"),
        storage_aws_access_key_id: str | None = config_option(
            "storage_aws_access_key_id", "AWS access key ID"
        ),
        storage_aws_secret_access_key: str | None = config_option(
            "storage_aws_secret_access_key", "AWS secret access key"
        ),
        storage_gcs_service_account_path: str | None = config_option(
            "storage_gcs_service_account_path", "Path to GCS service account JSON"
        ),
        storage_azure_connection_string: str | None = config_option(
            "storage_azure_connection_string", "Azure Blob connection string"
        ),
        storage_poll_interval_seconds: int = config_option(
            "storage_poll_interval_seconds", "Watcher poll interval in seconds"
        ),
        # Safety
        safety_refuse_on_empty: bool = config_option(
            "safety_refuse_on_empty", "Refuse when no chunks are retrieved"
        ),
        safety_refuse_on_low_score: bool = config_option(
            "safety_refuse_on_low_score", "Refuse when scores are below threshold"
        ),
        safety_min_answer_score: float | None = config_option(
            "safety_min_answer_score", "Minimum score required for an answer"
        ),
        safety_groundedness_check: bool = config_option(
            "safety_groundedness_check", "Enable groundedness check"
        ),
        safety_pii_redaction: bool = config_option(
            "safety_pii_redaction", "Redact PII before embedding"
        ),
        safety_prompt_injection_hardening: bool = config_option(
            "safety_prompt_injection_hardening", "Harden chunks against prompt injection"
        ),
        # Multi-tenancy
        default_tenant_id: str | None = config_option(
            "default_tenant_id", "Default tenant id for documents without one"
        ),
    ) -> None:
        """Configure raggit and bootstrap the system for first-time use."""
        # Ensure the latest env file state is reflected in option defaults and
        # that the in-memory cache is clear before we persist new values.
        get_settings.cache_clear()

        env_values = _collect_env_values(**locals())
        config_path = write_env_file(env_values)
        console.print(f"[green]Configuration written to {config_path}[/green]")

        # Reload settings so subsequent steps see the just-written values.
        get_settings.cache_clear()

        if skip_system_setup:
            console.print(
                "[yellow]Skipped system setup. Run again without "
                "--skip-system-setup to bootstrap services.[/yellow]"
            )
            return

        try:
            if not skip_migrations:
                with console.status("[bold green]Running database migrations..."):
                    _run_migrations()
                console.print("[green]Database migrations up to date.[/green]")

            asyncio.run(
                _bootstrap_system(
                    skip_db_check=skip_db_check,
                    skip_qdrant_setup=skip_qdrant_setup,
                    skip_storage_setup=skip_storage_setup,
                )
            )
        except Exception as exc:
            console.print(Panel(f"[red]Setup failed: {exc}[/red]", title="Error"))
            raise typer.Exit(1) from exc

        console.print(Panel("[green]raggit is configured and ready.[/green]", title="Setup"))
