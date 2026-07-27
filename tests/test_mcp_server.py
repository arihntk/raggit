"""Tests for the optional MCP server."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


from collections.abc import Iterator


@pytest.fixture
def no_mcp_module() -> Iterator[None]:
    """Temporarily remove the mcp module from sys.modules."""
    real_mcp = sys.modules.pop("mcp", None)
    real_fastmcp = sys.modules.pop("mcp.server.fastmcp", None)
    with patch.dict(sys.modules, {"mcp": None}):
        yield
    if real_mcp is not None:
        sys.modules["mcp"] = real_mcp
    if real_fastmcp is not None:
        sys.modules["mcp.server.fastmcp"] = real_fastmcp


def test_mcp_module_imports_without_mcp_installed(no_mcp_module: None) -> None:
    """The mcp_server module must import even when mcp is not installed."""
    # Remove cached module to force re-import.
    sys.modules.pop("raggit.mcp_server", None)
    import raggit.mcp_server as mcp_server

    assert mcp_server is not None


def test_check_mcp_raises_without_mcp(no_mcp_module: None) -> None:
    """Runtime error must explain how to install the mcp extra."""
    sys.modules.pop("raggit.mcp_server", None)
    from raggit.mcp_server import _check_mcp

    with pytest.raises(RuntimeError, match="raggit\\[mcp\\]"):
        _check_mcp()


def test_get_mcp_server_returns_fastmcp_instance() -> None:
    """When mcp is installed, the server instance is created lazily."""
    from raggit.mcp_server import _get_mcp_server

    server = _get_mcp_server()
    assert server is not None
    assert server.name == "raggit"


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    """The server exposes the expected raggit tools."""
    from raggit.mcp_server import _get_mcp_server

    server = _get_mcp_server()
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    expected = {
        "query",
        "get_status",
        "list_documents",
        "get_document",
        "list_chunks",
        "get_chunk",
        "list_logs",
        "get_config",
        "ingest",
        "run_eval",
    }
    assert expected.issubset(tool_names)


@pytest.mark.asyncio
async def test_get_sse_app_returns_asgi_app() -> None:
    """The SSE ASGI app is available when mcp is installed."""
    from raggit.mcp_server import get_sse_app

    app = await get_sse_app()
    assert app is not None


def test_cli_mcp_command_available() -> None:
    """The CLI main module registers the mcp command."""
    from raggit.cli.main import app

    commands = {cmd.callback.__name__ for cmd in app.registered_commands if cmd.callback is not None}
    assert "mcp" in commands
