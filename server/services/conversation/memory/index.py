from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ....logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
_INDEX_DIR = _DATA_DIR / "memory" / "embeddings"


class MemorySemanticIndex:
    """FAISS index over memory units (by integer id)."""

    def __init__(self, index_dir: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._index_dir = index_dir
        self._index_path = index_dir / "faiss.index"
        self._model = SentenceTransformer(model_name)
        self._lock = threading.Lock()
        self._ensure_dir()
        self._index = self._load_or_create()

    def _ensure_dir(self) -> None:
        try:
            self._index_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("memory index directory creation failed", extra={"error": str(exc)})

    def _new_index(self) -> faiss.Index:
        dim = self._model.get_sentence_embedding_dimension()
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def _load_or_create(self) -> faiss.Index:
        if self._index_path.exists():
            try:
                return faiss.read_index(str(self._index_path))
            except Exception as exc:
                logger.warning("failed to load memory faiss index; recreating", extra={"error": str(exc)})
        return self._new_index()

    def _save(self) -> None:
        try:
            faiss.write_index(self._index, str(self._index_path))
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to save memory faiss index", extra={"error": str(exc)})

    def add(self, ids: List[int], texts: List[str]) -> None:
        if not ids or not texts or len(ids) != len(texts):
            return
        embeddings = self._model.encode(texts, normalize_embeddings=True).astype(np.float32)
        id_arr = np.array(ids, dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(embeddings, id_arr)
            self._save()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        q = (query or "").strip()
        if not q:
            return []
        emb = self._model.encode([q], normalize_embeddings=True).astype(np.float32)
        with self._lock:
            if self._index.ntotal == 0:
                return []
            distances, ids = self._index.search(emb, min(int(top_k), int(self._index.ntotal)))
        results: List[Tuple[int, float]] = []
        for _id, score in zip(ids[0].tolist(), distances[0].tolist()):
            if int(_id) < 0:
                continue
            results.append((int(_id), float(score)))
        return results

    def clear(self) -> None:
        try:
            with self._lock:
                self._index = self._new_index()
                if self._index_path.exists():
                    self._index_path.unlink()

            logger.info(f"Cleared memory unit index.")

        except Exception as e:
            logger.error(f"Failed to clear memory unit index: {e}")


_index: Optional[MemorySemanticIndex] = None
_index_lock = threading.Lock()


def get_memory_index() -> MemorySemanticIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = MemorySemanticIndex(_INDEX_DIR)

                # Best-effort bootstrap: if the index is empty but we have stored units,
                # embed a recent window so retrieval works after restarts.
                try:
                    if getattr(_index._index, "ntotal", 0) == 0:
                        from .unit_store import get_memory_unit_store

                        store = get_memory_unit_store()
                        recent = store.list_recent(limit=200)
                        if recent:
                            _index.add([u.id for u in recent], [u.text for u in recent])
                except Exception:
                    pass
    return _index


__all__ = ["MemorySemanticIndex", "get_memory_index"]