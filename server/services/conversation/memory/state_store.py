from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ....logging_config import logger
from .models import UserState


_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_STATE_PATH = _DATA_DIR / "memory" / "user_state.json"


class UserStateStore:
    """Small durable user state stored as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("user state directory creation failed", extra={"error": str(exc)})

    def load_state(self) -> UserState:
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return UserState.empty()
            except Exception as exc:  # pragma: no cover
                logger.warning("user state read failed", extra={"error": str(exc)})
                return UserState.empty()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return UserState.empty()
        if not isinstance(data, dict):
            return UserState.empty()
        return UserState.from_dict(data)

    def write_state(self, state: UserState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        tmp = self._path.with_suffix(".tmp")
        with self._lock:
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self._path)
            except Exception as exc:  # pragma: no cover
                logger.error("user state write failed", extra={"error": str(exc)})
                raise
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass


_store: Optional[UserStateStore] = None
_store_lock = threading.Lock()


def get_user_state_store() -> UserStateStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = UserStateStore(_STATE_PATH)
    return _store


__all__ = ["UserStateStore", "get_user_state_store"]