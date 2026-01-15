import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from ...logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EXECUTION_META_DIR = _DATA_DIR / "execution_agents"


def _slugify(name: str) -> str:
    """Convert agent name to filesystem-safe slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name.strip()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "agent"


class ExecutionAgentMetadataStore:
    """
    Manages per-agent metadata in .json format.
    """

    def __init__(self, meta_dir: Path):
        self._meta_dir = meta_dir
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        try:
            self._meta_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create directory: {e}")

    def _lock_for(self, agent_name: str) -> threading.Lock:
        """
        Get or create a lock for an agent.
        """
        slug = _slugify(agent_name)
        with self._global_lock:
            if slug not in self._locks:
                self._locks[slug] = threading.Lock()
            return self._locks[slug]

    def _meta_path(self, agent_name: str) -> Path:
        """
        Get meta file path for an agent.
        """
        return self._meta_dir / f"{_slugify(agent_name)}.json"

    def load_metadata(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Load metadata for an agent.
        """
        path = self._meta_path(agent_name)
        if not path.exists():
            return None
        with self._lock_for(agent_name):
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata for {agent_name}: {e}")
                return None

    def save_metadata(self, agent_name: str, description: Optional[str] = None):
        """
        Save metadata for an agent.
        """
        path = self._meta_path(agent_name)

        with self._lock_for(agent_name):
            try:
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {
                        "name": agent_name,
                        "description": agent_name,
                        "last_used": None,
                        "usage_count": 0,
                    }

                meta["name"] = agent_name
                if description:
                    meta["description"] = description
                meta["last_used"] = datetime.now(timezone.utc).isoformat()
                meta["usage_count"] = meta.get("usage_count", 0) + 1

                with path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"Failed to save metadata for {agent_name}: {e}")

    #TODO: Add method to clear metadata if user clicks clear button


_execution_agent_metadata = ExecutionAgentMetadataStore(_EXECUTION_META_DIR)


def get_execution_agent_metadata() -> ExecutionAgentMetadataStore:
    """Get the singleton metadata store instance."""
    return _execution_agent_metadata