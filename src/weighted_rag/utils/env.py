"""Lightweight .env loader for configurable model overrides."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False


def load_env_file(env_path: Path | None = None) -> None:
    """Loads key=value pairs from a .env file into os.environ (without overriding)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidate = env_path or _default_env_path()
    if candidate and candidate.exists():
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    _ENV_LOADED = True


def _default_env_path() -> Path | None:
    current = Path(__file__).resolve()
    # WeightedRAG src/weighted_rag/utils/env.py -> project root is parents[3]
    for parent in current.parents:
        env_path = parent / ".env"
        if env_path.exists():
            return env_path
        if parent == parent.parent:
            break
    return None
