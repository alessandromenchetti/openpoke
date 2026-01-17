# server/services/telemetry/llm_usage.py
from __future__ import annotations

from typing import Any, Dict

from ...logging_config import logger
from .llm_usage_store import get_llm_usage_store

_store = get_llm_usage_store()


def record_llm_usage(payload: Dict[str, Any]) -> None:
    """Append a single usage record to SQLite (best-effort)."""
    try:
        _store.insert(payload)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to record llm usage", extra={"error": str(exc)})
