import json
import threading
from pathlib import Path
from typing import List

from ...logging_config import logger
from ...config import get_settings


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_LRU_CACHE_PATH = _DATA_DIR / "execution_agents" / "lru_cache.json"


class ExecutionAgentLRUCache:
    """LRU cache for execution agents"""

    def __init__(self, cache_path: Path, size: int = 5):
        """Initialize the LRU cache."""
        self._cache_path = cache_path
        self._size = size
        self._cache: List[str] = []
        self._lock = threading.Lock()
        self.load()

    def _initialize_from_metadata(self) -> List[str]:
        """Build initial cache from metadata files stored by last_used timestamp."""
        try:
            from .metadata import get_execution_agent_metadata

            metadata_store = get_execution_agent_metadata()
            meta_dir = metadata_store.meta_dir()

            if not meta_dir.exists():
                return []

            agents_with_timestamps = []
            for meta_file in meta_dir.glob("*-meta.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        agent_name = meta.get("name")
                        last_used = meta.get("last_used")

                        if agent_name and last_used:
                            agents_with_timestamps.append((agent_name, last_used))

                except Exception as e:
                    logger.warning(f"Failed to read metadata file {meta_file.name}: {e}")
                    continue

            agents_with_timestamps.sort(key=lambda x: x[1])
            initialized_cache = [name for name, _ in agents_with_timestamps[-self._size:]]

            return initialized_cache

        except Exception as e:
            logger.error(f"Failed to initialize LRU cache from metadata: {str(e)}")
            return []

    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        try:
            data = {
                "agents": self._cache,
                "size": self._size
            }

            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                logger.info(f"Saved {len(self._cache)} agents to cache.")

        except Exception as e:
            logger.error(f"Failed to save LRU cache: {e}.")

    def load(self) -> None:
        """Load cache from disk."""
        if not self._cache_path.exists():
            self._cache = self._initialize_from_metadata()
            self._save_to_disk()
            return

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cache = data.get("agents", [])

        except Exception as e:
            logger.warning(f"Failed to load lru_cache.json: {e}")
            self._cache = []

    def get_hot_list(self) -> List[str]:
        """Get the list of recently used agents."""
        with self._lock:
            return list(reversed(self._cache))

    def update(self, agent_name: str) -> None:
        """Update the cache after agent execution."""
        with self._lock:
            if agent_name in self._cache:
                self._cache.remove(agent_name)

            self._cache.append(agent_name)

            if len(self._cache) > self._size:
                removed = self._cache.pop(0)
                logger.debug(f"Removed least recently used agent from cache: {removed}")

            self._save_to_disk()

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache = []
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
            logger.info("Cleared LRU cache.")
        except Exception as e:
            logger.error(f"Failed to clear LRU cache: {e}.")

_settings = get_settings()
_execution_agent_lru_cache = ExecutionAgentLRUCache(_LRU_CACHE_PATH, _settings.lru_cache_size)


def get_execution_agent_lru_cache() -> ExecutionAgentLRUCache:
    """Get the singleton LRU cache instance."""
    return _execution_agent_lru_cache