"""Shared CLI option factories for raggit commands."""

from __future__ import annotations

from typing import Any

import typer

from raggit.api.models import QueryRewriteMode, SourceType
from raggit.core.config import Settings, get_settings


def _settings() -> Settings:
    """Return current settings (env file, .env, or built-in defaults)."""
    return get_settings()


def config_option(attr: str, help_text: str, **kwargs: Any) -> Any:
    """Create a typer.Option whose default comes from the current setting value."""
    return typer.Option(
        default=getattr(_settings(), attr),
        help=help_text,
        **kwargs,
    )


def source_type_option(help_text: str = "Storage backend") -> Any:
    return typer.Option(
        default=SourceType(_settings().storage_source_type),
        help=help_text,
        case_sensitive=False,
        show_choices=True,
    )


def query_rewrite_option(help_text: str = "Query rewrite strategy") -> Any:
    return typer.Option(
        default=QueryRewriteMode(_settings().retrieval_query_rewrite or "none"),
        help=help_text,
        case_sensitive=False,
        show_choices=True,
    )
