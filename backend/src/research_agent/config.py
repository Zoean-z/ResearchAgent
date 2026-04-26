"""Environment-file loading helpers for local development."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load key/value pairs from a local `.env`-style file into `os.environ`.

    The default path resolution is:
    1. explicit `path`
    2. `RESEARCH_AGENT_ENV_FILE`
    3. repository-root `.env`

    Existing environment variables are preserved unless `override=True`.
    """

    env_path = _resolve_env_path(path)
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return env_path


def _resolve_env_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.getenv("RESEARCH_AGENT_ENV_FILE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / ".env"


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


__all__ = ["load_env_file"]
