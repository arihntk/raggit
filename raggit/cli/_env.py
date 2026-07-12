"""Environment-file helpers for raggit setup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from raggit.core.config import config_file_path


def _shell_quote(value: str) -> str:
    """Quote a value for safe use in a POSIX shell source file."""
    if any(ch in value for ch in (" ", "#", '"', "'", "\\", "\n", "$")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _parse_existing_env(path: Path) -> dict[str, str]:
    """Parse an existing env file into a key-value dictionary.

    Only simple KEY=VALUE lines are handled; comments and blank lines are
    ignored.
    """
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        parsed[key.strip()] = raw_value.strip().strip('"').strip("'")
    return parsed


def write_env_file(values: dict[str, Any]) -> Path:
    """Write configuration values to the raggit env file.

    Existing values in the file are preserved unless overridden by ``values``.
    Values are converted to strings and shell-quoted when necessary.
    The file is created with 0o600 permissions.
    """
    config_path = config_file_path()
    merged = _parse_existing_env(config_path)
    for key, raw in values.items():
        if raw is None:
            continue
        value = ("true" if raw else "false") if isinstance(raw, bool) else str(raw)
        merged[key] = _shell_quote(value)

    lines = [f"{key}={value}" for key, value in sorted(merged.items())]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(config_path, 0o600)
    return config_path
