from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ...logging_config import logger

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_TELEMETRY_DIR = _DATA_DIR / "telemetry"
_LLM_USAGE_DB_PATH = _TELEMETRY_DIR / "llm_usage.sqlite"


class LLMUsageStore:
    """Append-only persistence for LLM usage stats backed by SQLite."""

    _COLUMNS = (
        "ts_utc",
        "trace_id",
        "root_source",
        "component",
        "agent_name",
        "purpose",
        "span_id",
        "parent_span_id",
        "model",
        "duration_ms",
        "status",
        "http_status",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "error",
    )

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_directory()
        self._ensure_schema()

    def _ensure_directory(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("telemetry directory creation failed", extra={"error": str(exc)})

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        # ts_utc first, as requested
        schema_sql = """
                     CREATE TABLE IF NOT EXISTS llm_usage (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         ts_utc TEXT NOT NULL,
                         trace_id TEXT,
                         root_source TEXT,
                         component TEXT,
                         agent_name TEXT,
                         purpose TEXT,
                         span_id TEXT,
                         parent_span_id TEXT,
                         model TEXT,
                         duration_ms INTEGER,
                         status TEXT,
                         http_status INTEGER,
                         prompt_tokens INTEGER,
                         completion_tokens INTEGER,
                         total_tokens INTEGER,
                         cost REAL,
                         error TEXT
                     );
                     """
        index_sql = """
                    CREATE INDEX IF NOT EXISTS idx_llm_usage_ts ON llm_usage (ts_utc);
                    CREATE INDEX IF NOT EXISTS idx_llm_usage_trace ON llm_usage (trace_id);
                    CREATE INDEX IF NOT EXISTS idx_llm_usage_root ON llm_usage (root_source);
                    CREATE INDEX IF NOT EXISTS idx_llm_usage_purpose ON llm_usage (purpose);
                    """

        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(schema_sql)
            conn.executescript(index_sql)

    def insert(self, payload: Dict[str, Any]) -> Optional[int]:
        try:
            record = self._normalize(payload)

            with self._lock, self._connect() as conn:
                columns = ", ".join(self._COLUMNS)
                placeholders = ", ".join([":" + c for c in self._COLUMNS])
                sql = f"INSERT INTO llm_usage ({columns}) VALUES ({placeholders})"
                conn.execute(sql, record)
                row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                return int(row_id)

        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to record llm usage", extra={"error": str(exc)})
            return None

    def _normalize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts_utc = payload.get("ts_utc") or datetime.now(timezone.utc).isoformat()

        err = payload.get("error")
        if err is not None and not isinstance(err, str):
            try:
                err = json.dumps(err, ensure_ascii=False)
            except Exception:
                err = str(err)

        row: Dict[str, Any] = {c: payload.get(c) for c in self._COLUMNS}
        row["ts_utc"] = ts_utc
        row["error"] = err

        # light type coercions (SQLite is permissive, but this helps consistency)
        for k in ("duration_ms", "http_status", "prompt_tokens", "completion_tokens", "total_tokens"):
            if row.get(k) is not None:
                try:
                    row[k] = int(row[k])
                except Exception:
                    row[k] = None

        if row.get("cost") is not None:
            try:
                row["cost"] = float(row["cost"])
            except Exception:
                row["cost"] = None

        return row


    # TODO: Add clear method for when user wants to clear all data


_llm_usage_store = LLMUsageStore(_LLM_USAGE_DB_PATH)


def get_llm_usage_store() -> LLMUsageStore:
    return _llm_usage_store


__all__ = ["LLMUsageStore", "get_llm_usage_store"]