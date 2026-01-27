from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .index import get_memory_index
from .models import MemoryUnit
from .state_store import get_user_state_store
from .unit_store import get_memory_unit_store
from .utils import build_alias_index, entity_match


@dataclass
class RetrievalSettings:
    # Overfetch then filter/dedupe
    overfetch_per_query: int = 30
    top_k_per_query: int = 6
    global_max: int = 12
    min_after_filter: int = 2  # if entity filter yields less, relax


class MemoryRetriever:
    """Retrieve relevant memory units for a set of decomposed queries."""

    def __init__(self, settings: Optional[RetrievalSettings] = None) -> None:
        self.settings = settings or RetrievalSettings()
        self._index = get_memory_index()
        self._store = get_memory_unit_store()
        self._state_store = get_user_state_store()

    async def retrieve(self, decomposed: List[Dict[str, object]]) -> List[MemoryUnit]:
        state = self._state_store.load_state()
        alias_index = build_alias_index(state.entities)

        per_query_hits: List[List[Tuple[int, float]]] = []
        all_ids: List[int] = []
        for item in decomposed:
            if not isinstance(item, dict):
                per_query_hits.append([])
                continue
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                per_query_hits.append([])
                continue
            hits = self._index.search(query, top_k=self.settings.overfetch_per_query)
            per_query_hits.append(hits)
            all_ids.extend([i for i, _ in hits])

        units = self._store.get_units(list(dict.fromkeys(all_ids)))
        units_by_id = {u.id: u for u in units}

        final: List[MemoryUnit] = []

        for item, hits in zip(decomposed, per_query_hits):
            if not isinstance(item, dict):
                continue
            query = item.get("query")
            if not isinstance(query, str) or not query.strip():
                continue
            query_entities = item.get("entities")
            q_ents = [e for e in (query_entities or []) if isinstance(e, str) and e.strip()]

            per_q: List[MemoryUnit] = []
            for _id, _score in hits:
                u = units_by_id.get(_id)
                if not u:
                    continue
                if q_ents:
                    if entity_match(q_ents, u.entities, alias_index=alias_index):
                        per_q.append(u)
                else:
                    per_q.append(u)
                if len(per_q) >= self.settings.top_k_per_query:
                    break

            if q_ents and len(per_q) < self.settings.min_after_filter:
                per_q = []
                for _id, _score in hits:
                    u = units_by_id.get(_id)
                    if u:
                        per_q.append(u)
                    if len(per_q) >= self.settings.top_k_per_query:
                        break

            for u in per_q[: self.settings.top_k_per_query]:
                if u.id not in {x.id for x in final}:
                    final.append(u)
                if len(final) >= self.settings.global_max:
                    break
            if len(final) >= self.settings.global_max:
                break

        return final


_retriever: Optional[MemoryRetriever] = None


def get_memory_retriever() -> MemoryRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MemoryRetriever()
    return _retriever


__all__ = ["RetrievalSettings", "MemoryRetriever", "get_memory_retriever"]