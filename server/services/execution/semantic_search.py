import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ...logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EMBEDDINGS_DIR = _DATA_DIR / "execution_agents" / "embeddings"


class AgentSemanticSearch:
    """Semantic search over agents using FAISS."""

    def __init__(self, embeddings_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading FAISS index from {embeddings_dir}")
        self._embeddings_dir = embeddings_dir
        self._index_path = embeddings_dir / "faiss.index"
        logger.info(f"Loading embeddings model")
        self._model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatIP = None
        self._lock = threading.Lock()

        logger.info(f"Loading FAISS index")
        self._ensure_directory()
        self._load()
        logger.info("Done")

    def _ensure_directory(self):
        """Create embeddings dir if it doesn't exist."""
        try:
            self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create embeddings directory: {e}")

    def _get_agent_names(self) -> List[str]:
        """Get agent names from roster."""
        try:
            from . import get_agent_roster
            roster = get_agent_roster()
            return roster.get_agents()
        except Exception as e:
            logger.error(f"Failed to get agent names from roster: {e}")
            return []

    def _rebuild_from_roster(self, agent_names: List[str]) -> None:
        """Rebuild the FAISS index from the current roster."""
        if not agent_names:
            return

        try:
            logger.info(f"Rebuilding FAISS index from {len(agent_names)} agents.")

            dimension = self._model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatIP(dimension)

            embeddings = self._model.encode(agent_names, normalize_embeddings=True)
            embeddings = embeddings.astype(np.float32)

            new_index.add(embeddings)
            self._index = new_index

            logger.info(f"Rebuilt FAISS index from {len(agent_names)} agents.")
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")

    def _load(self) -> None:
        """Load FAISS index from disk, rebuild from roster if needed, or create empty."""
        dimension = self._model.get_sentence_embedding_dimension()

        if self._index_path.exists():
            try:
                with self._lock:
                    self._index = faiss.read_index(str(self._index_path))
                logger.debug(f"Loaded FAISS index with {self._index.ntotal} agents")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

        agent_names = self._get_agent_names()
        if agent_names:
            logger.info(f"No index file found, rebuilding from {len(agent_names)} agents in roster")

            with self._lock:
                self._index = faiss.IndexFlatIP(dimension)
                self._rebuild_from_roster(agent_names)
                self._save()

            return

        with self._lock:
            self._index = faiss.IndexFlatIP(dimension)
        logger.debug("Initialized empty FAISS index (no roster data)")

    def _save(self) -> None:
        try:
            faiss.write_index(self._index, str(self._index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_agent_to_index(self, agent_name: str) -> None:
        """Add an agent's embedding to the index if not already present."""
        agent_names = self._get_agent_names()

        if agent_name not in agent_names:
            logger.warning(f"Agent {agent_name} not found in roster, skipping indexing.")
            return

        should_save = False

        with self._lock:
            current_size = self._index.ntotal
            roster_size = len(agent_names)

            if current_size < roster_size:
                embedding = self._model.encode([agent_name], normalize_embeddings=True)
                embedding = embedding.astype(np.float32)

                self._index.add(embedding)
                self._save()
                logger.debug(f"Indexed agent: {agent_name}.")
            else:
                logger.debug(f"Agent {agent_name} already indexed.")

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform semantic search for agents similar to query."""
        agent_names = self._get_agent_names()

        if not agent_names:
            return []

        with self._lock:
            if self._index.ntotal == 0:
                self._rebuild_from_roster(agent_names)
                self._save()

            if self._index.ntotal == 0:
                return []

            query_embedding = self._model.encode([query], normalize_embeddings=True)
            query_embedding = query_embedding.astype(np.float32)

            k = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(query_embedding, k)

        results: List[Tuple[str, float]] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(agent_names):
                results.append((agent_names[idx], float(distance)))
            else:
                logger.warning(f"FAISS idx {idx} out of range for roster size {len(agent_names)}")

        return results

    def clear(self) -> None:
        """Clear the FAISS index and delete from disk."""
        try:
            dimension = self._model.get_sentence_embedding_dimension()

            with self._lock:
                self._index = faiss.IndexFlatIP(dimension)
                if self._index_path.exists():
                    self._index_path.unlink()

            logger.info(f"Cleared FAISS index.")

        except Exception as e:
            logger.error(f"Failed to clear FAISS index: {e}")


_agent_semantic_search = AgentSemanticSearch(_EMBEDDINGS_DIR)


def get_agent_semantic_search() -> AgentSemanticSearch:
    """Get the singleton AgentSemanticSearch instance."""
    return _agent_semantic_search