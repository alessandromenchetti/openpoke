import json
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ...logging_config import logger


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EMBEDDINGS_DIR = _DATA_DIR / "execution_agents" / "embeddings"


def _extract_first_json_array(text: str):
    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if start is None:
            if ch == "[":
                start = i
                depth = 1
            continue

        # Inside JSON candidate
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


async def decompose_user_query(user_query: str, conversation_history: str) -> List[Dict[str, object]]:
    """Use LLM to decompose user query into search query(ies).

    ### NEW ###
    Returns a list of objects:
    [{"query": str, "entities": [str, ...]}, ...]
    """
    import re
    from ...config import get_settings
    from ...openrouter_client import request_chat_completion

    settings = get_settings()

    def _truncate(s: str, n: int = 220) -> str:  # NEW
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else (s[: n - 3] + "...")

    system_prompt = """
    You are an expert at decomposing user requests into semantic search queries.

    Given the a new user request with a snippet of conversation history as context, extract distinct intents or targets
    from the user request that should be searched for separately.

    Rules:
    - Each item should focus on ONE specific intent, person, or action
    - Resolve pronouns using conversation history (e.g., "she" → "alice")
    - If you cannot resolve a pronoun, omit entities and focus on the intent/action
    - Resolve any temporal references (e.g., "tomorrow 8pm" → specific date at 8pm) using timestamps in the conversation history
    - Keep queries focused and specific
    - Return ONLY a JSON array of objects with keys: "query" and "entities"
    - "entities" must be a JSON array of strings (often empty; ideally <= 1 entity)
    - Prefer full names or email addresses when available in context

    Output format:
    [{"query": "...", "entities": ["..."]}, ...]

    Examples:
    Input: "send email to bob and alice"
    Output: [{"query":"send email to bob","entities":["bob"]},{"query":"send email to alice","entities":["alice"]}]

    Input: "has she replied?" (with history mentioning Alice)
    Output: [{"query":"alice email reply","entities":["alice"]}]

    Input: "has she replied?" (no clear referent in history)
    Output: [{"query":"email reply check","entities":[]}]

    Input: "check for grade email and remind me in 10 minutes"
    Output: [{"query":"check grade email","entities":[]},{"query":"reminder in 10 minutes","entities":[]}]

    Input: "what about ryan or vince?" (with previous message in context saying something like "has alice replied?")
    Output: [{"query":"ryan email reply","entities":["ryan"]},{"query":"vince email reply","entities":["vince"]}]

    Input: "search my emails"
    Output: [{"query":"search emails","entities":[]}]""".strip()

    history_lines = conversation_history.strip().split("\n")
    filtered_lines = [line for line in history_lines if line.strip() and not line.startswith("<agent_message")]

    recent_history = "\n".join(filtered_lines[-10:]) if filtered_lines else "None"

    logger.info(
        f"[DECOMP] starting | user_query={_truncate(user_query, 140)} | recent_history_excerpt={_truncate(recent_history, 200)}"
    )  # NEW

    messages = [
        {
            "role": "user",
            "content": f"""
                <conversation_history>
                {recent_history}
                </conversation_history>

                <user_query>
                {user_query}
                </user_query>

                Decompose into semantic search queries (JSON array only):""".strip()
        }
    ]

    try:
        response = await request_chat_completion(
            model=settings.interaction_agent_model,
            messages=messages,
            system=system_prompt,
            api_key=settings.openrouter_api_key,
        )

        content = response["choices"][0]["message"]["content"].strip()

        logger.info(f"[DECOMP] raw model output excerpt: {_truncate(content, 240)}")  # NEW

        arr_text = _extract_first_json_array(content)
        if arr_text:
            items = json.loads(arr_text)
            if isinstance(items, list) and items:
                normalized: List[Dict[str, object]] = []
                for item in items:
                    if isinstance(item, str):
                        normalized.append({"query": item, "entities": []})
                        continue
                    if not isinstance(item, dict):
                        continue
                    q = item.get("query")
                    ents = item.get("entities")
                    if isinstance(q, str) and q.strip():
                        if isinstance(ents, list):
                            entities = [e for e in ents if isinstance(e, str) and e.strip()]
                        else:
                            entities = []
                        normalized.append({"query": q.strip(), "entities": entities})

                if normalized:
                    logger.info(f"[DECOMP] parsed items: {normalized}")  # NEW
                    return normalized

        logger.warning(f"[DECOMP] parse failed; fallback to original | excerpt={_truncate(content, 240)}")  # NEW
        return [{"query": user_query, "entities": []}]

    except Exception as e:
        logger.warning(f"[DECOMP] exception; fallback to original | error={e}", exc_info=True)  # NEW
        return [{"query": user_query, "entities": []}]


class AgentSemanticSearch:
    """Semantic search over agents using FAISS."""

    def __init__(self, embeddings_dir: Path, model_name: str = "all-MiniLM-L6-v2"):
        self._embeddings_dir = embeddings_dir
        self._index_path = embeddings_dir / "faiss.index"
        self._model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatIP = None
        self._lock = threading.Lock()

        self._ensure_directory()
        self._load()

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
            dimension = self._model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatIP(dimension)

            embeddings = self._model.encode(agent_names, normalize_embeddings=True)
            embeddings = embeddings.astype(np.float32)

            new_index.add(embeddings)
            self._index = new_index

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

    def semantic_search(self, queries: List[str], top_k: int = 15, boost_weight: float = 0.2) -> List[Tuple[str, float]]:
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

            agent_scores: Dict[str, float] = {}

            query_embeddings = self._model.encode(queries, normalize_embeddings=True)
            query_embeddings = query_embeddings.astype(np.float32)

            for query, query_embedding in zip(queries, query_embeddings):
                query_embedding = query_embedding.reshape(1, -1)

                k = min(top_k*2, self._index.ntotal)
                distances, indices = self._index.search(query_embedding, k)

                query_keywords = set(query.lower().split())

                for idx, distance in zip(indices[0], distances[0]):
                    if idx < len(agent_names):
                        agent_name = agent_names[idx]
                        agent_keywords = set(agent_name.lower().replace("-", " ").replace(
                            "_", " ").split())

                        keyword_overlap = len(query_keywords & agent_keywords)
                        boosted_score = float(distance) + (boost_weight * keyword_overlap)

                        agent_scores[agent_name] = max(agent_scores.get(agent_name, 0.0), boosted_score)

        sorted_results = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

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