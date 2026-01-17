# server/services/telemetry/llm_usage.py
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ...logging_config import logger

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_TELEMETRY_DIR = _DATA_DIR / "telemetry"
_LLM_USAGE_PATH = _TELEMETRY_DIR / "llm_usage.jsonl"

_write_lock = threading.Lock()


def record_llm_usage(payload: Dict[str, Any]) -> None:
    """Append a single JSON record to server/data/telemetry/llm_usage.jsonl."""
    try:
        _TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

        # Add timestamp if caller didn't provide it
        payload.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())

        line = json.dumps(payload, ensure_ascii=False)

        with _write_lock:
            with open(_LLM_USAGE_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        logger.warning("failed to record llm usage", extra={"error": str(exc)})
