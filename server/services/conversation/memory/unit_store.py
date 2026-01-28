from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ....logging_config import logger
from .models import MemoryUnit


_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "memory" / "memory_units.sqlite"


class MemoryUnitStore:
    """SQLite-backed storage for memory units."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("memory store directory creation failed", extra={"error": str(exc)})
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_units (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL,
                        unit_type TEXT NOT NULL,
                        entities_json TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        source_refs_json TEXT NOT NULL,
                        is_active INTEGER NOT NULL,
                        dedupe_key TEXT UNIQUE
                    )
                    """
                )
                con.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_units_active ON memory_units(is_active)"
                )
                con.commit()
            finally:
                con.close()

    def add_unit(
        self,
        text: str,
        unit_type: str = "fact",
        entities: Optional[List[str]] = None,
        confidence: float = 0.8,
        source_refs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """Insert a new memory unit. Returns id, or None when deduped."""
        clean_text = (text or "").strip()
        if not clean_text:
            return None
        ents = [e.strip() for e in (entities or []) if isinstance(e, str) and e.strip()]
        refs = [r for r in (source_refs or []) if isinstance(r, dict)]
        created_at = datetime.now(timezone.utc).isoformat()

        # Very simple dedupe key.
        dedupe_key = f"{unit_type}|{clean_text.lower()}|{','.join(sorted([e.lower() for e in ents]))}"

        with self._lock:
            con = self._connect()
            try:
                cur = con.execute(
                    """
                    INSERT INTO memory_units (text, unit_type, entities_json, confidence, created_at, source_refs_json, is_active, dedupe_key)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        clean_text,
                        unit_type or "fact",
                        json.dumps(ents, ensure_ascii=False),
                        float(confidence),
                        created_at,
                        json.dumps(refs, ensure_ascii=False),
                        dedupe_key,
                    ),
                )
                con.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError:
                # deduped
                return None
            except Exception as exc:  # pragma: no cover
                logger.warning("failed to insert memory unit", extra={"error": str(exc)})
                return None
            finally:
                con.close()

    def get_units(self, ids: Iterable[int]) -> List[MemoryUnit]:
        id_list = [int(i) for i in ids if i is not None]
        if not id_list:
            return []
        placeholders = ",".join(["?"] * len(id_list))
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    f"SELECT * FROM memory_units WHERE id IN ({placeholders}) AND is_active=1",
                    tuple(id_list),
                ).fetchall()
            finally:
                con.close()
        units: List[MemoryUnit] = []
        for row in rows:
            try:
                entities = json.loads(row["entities_json"]) or []
            except Exception:
                entities = []
            try:
                refs = json.loads(row["source_refs_json"]) or []
            except Exception:
                refs = []
            units.append(
                MemoryUnit(
                    id=int(row["id"]),
                    text=str(row["text"]),
                    unit_type=str(row["unit_type"]),
                    entities=[e for e in entities if isinstance(e, str)],
                    confidence=float(row["confidence"]),
                    created_at=str(row["created_at"]),
                    source_refs=[r for r in refs if isinstance(r, dict)],
                    is_active=bool(int(row["is_active"])),
                )
            )
        return units

    def list_recent(self, limit: int = 200) -> List[MemoryUnit]:
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT * FROM memory_units WHERE is_active=1 ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            finally:
                con.close()
        return self.get_units([int(r["id"]) for r in rows])

    def clear_all(self) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute("DELETE FROM memory_units")
                con.commit()
                logger.info("Cleared all memory units from store.")
            except Exception as e:
                logger.error(f"Failed to clear memory units: {e}")
            finally:
                con.close()


_store: Optional[MemoryUnitStore] = None
_store_lock = threading.Lock()


def get_memory_unit_store() -> MemoryUnitStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryUnitStore(_DB_PATH)
    return _store


__all__ = ["MemoryUnitStore", "get_memory_unit_store"]