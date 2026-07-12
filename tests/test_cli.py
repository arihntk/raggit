"""Tests for the raggit CLI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from raggit.cli.main import app

runner = CliRunner()


@pytest.fixture
def isolated_config(tmp_path: Path) -> Iterator[Path]:
    """Use a temporary raggit config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    env_file = config_dir / "raggit.env"
    with patch("raggit.cli._env.config_file_path") as mock_env_path:
        mock_env_path.return_value = env_file
        yield env_file


def test_setup_writes_env_file(isolated_config: Path) -> None:
    """setup writes configuration values to the env file."""
    result = runner.invoke(
        app,
        [
            "setup",
            "--skip-system-setup",
            "--database-url",
            "postgresql+asyncpg://u:p@host/db",
            "--qdrant-url",
            "http://qdrant:6333",
            "--storage-source-type",
            "local",
            "--storage-uri",
            "/docs",
            "--llm-provider",
            "ollama",
            "--llm-model",
            "llama3.2",
            "--chunk-size",
            "512",
            "--reranker-enabled",
        ],
    )
    assert result.exit_code == 0, result.output
    content = isolated_config.read_text()
    assert "DATABASE_URL=postgresql+asyncpg://u:p@host/db" in content
    assert "QDRANT_URL=http://qdrant:6333" in content
    assert "STORAGE_SOURCE_TYPE=local" in content
    assert "STORAGE_URI=/docs" in content
    assert "LLM_PROVIDER=ollama" in content
    assert "LLM_MODEL=llama3.2" in content
    assert "CHUNK_SIZE=512" in content
    assert "RERANKER_ENABLED=true" in content


def test_setup_preserves_existing_values(isolated_config: Path) -> None:
    """setup keeps existing env file values when not overridden."""
    isolated_config.write_text("CUSTOM_KEY=keep\nDATABASE_URL=old\n")
    result = runner.invoke(
        app,
        [
            "setup",
            "--skip-system-setup",
            "--database-url",
            "postgresql+asyncpg://new/db",
        ],
    )
    assert result.exit_code == 0, result.output
    content = isolated_config.read_text()
    assert "DATABASE_URL=postgresql+asyncpg://new/db" in content
    assert "CUSTOM_KEY=keep" in content


def test_setup_shell_quotes_values(isolated_config: Path) -> None:
    """setup shell-quotes values with special characters."""
    result = runner.invoke(
        app,
        [
            "setup",
            "--skip-system-setup",
            "--storage-uri",
            "./data/my documents",
        ],
    )
    assert result.exit_code == 0, result.output
    content = isolated_config.read_text()
    assert 'STORAGE_URI="./data/my documents"' in content


def test_ingest_requires_local_path() -> None:
    """ingest exits when local storage is used without a path."""
    with patch("raggit.core.config.get_settings") as mock_settings:
        from raggit.core.config import Settings

        settings = Settings(
            storage_source_type="local",
            storage_uri="./data/documents",
        )
        mock_settings.return_value = settings
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code == 1
        assert "path is required" in result.output.lower()


def test_query_invalid_document_id() -> None:
    """query exits on malformed document id filters."""
    result = runner.invoke(app, ["query", "--document-id", "not-a-uuid", "hello"])
    assert result.exit_code == 1
    assert "Invalid document id" in result.output


def test_status_runs_against_empty_index() -> None:
    """status prints tables even when the index is empty."""
    with (
        patch("raggit.cli.status_cmd.AsyncSessionLocal"),
        patch("raggit.db.repository.DocumentRepository") as mock_doc_repo,
        patch("raggit.db.repository.EmbeddingCollectionRepository") as mock_coll_repo,
    ):
        mock_doc_repo.return_value.list_all = AsyncMock(return_value=[])
        mock_coll_repo.return_value.list_all = AsyncMock(return_value=[])
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0, result.output
        assert "Total documents: 0" in result.output
