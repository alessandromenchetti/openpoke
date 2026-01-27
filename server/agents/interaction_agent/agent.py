"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...logging_config import logger
from ...services.execution import get_execution_agent_lru_cache, get_agent_semantic_search

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
async def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    user_query: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))

    def _truncate(s: str, n: int = 220) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else (s[: n - 3] + "...")

    decomposed: Optional[List[Dict[str, object]]] = None
    agent_search_queries: List[str] = []

    if user_query:
        from ...services.execution.semantic_search import decompose_user_query
        from ...services.telemetry.trace_context import bind_span_context

        with bind_span_context(
                component="interaction_agent",
                purpose="interaction_agent.query_decomposition",
        ):
            decomposed = await decompose_user_query(user_query, transcript)

        # ### NEW: Build list of plain queries for agent search (no entities) ###
        agent_search_queries = [
            str(item.get("query")).strip()
            for item in (decomposed or [])
            if isinstance(item, dict) and isinstance(item.get("query"), str) and str(item.get("query")).strip()
        ]
        if not agent_search_queries:
            agent_search_queries = [user_query]

        logger.info(f"[IA] decomposed_queries | count={len(decomposed or [])} ")

    if decomposed is not None:
        try:
            from ...services.conversation.memory import get_memory_retriever, get_user_state_store

            state_store = get_user_state_store()
            user_state = state_store.load_state()

            logger.info(
                f"[IA] loaded user_state | profile={len(getattr(user_state, 'profile', []) or [])} "
                f"open_loops={len(getattr(user_state, 'open_loops', []) or [])} "
                f"commitments={len(getattr(user_state, 'commitments', []) or [])} "
                f"entities={len(getattr(user_state, 'entities', []) or [])}"
            )

            sections.append(_render_user_state(user_state))

            retriever = get_memory_retriever()
            retrieved = await retriever.retrieve(decomposed)

            preview = []
            for u in (retrieved or [])[:6]:
                preview.append(
                    {
                        "id": getattr(u, "id", None),
                        "type": getattr(u, "unit_type", None),
                        "entities": getattr(u, "entities", None),
                        "text": _truncate(getattr(u, "text", ""), 120),
                    }
                )
            logger.info(f"[IA] retrieved_memory | count={len(retrieved or [])} preview={preview}")

            sections.append(_render_retrieved_memory(retrieved))

        except Exception as exc:
            logger.warning(f"[IA] memory injection failed (continuing) | error={exc}", exc_info=True)

    active_agent_shortlist = await _render_active_agent_shortlist(agent_search_queries, transcript)
    sections.append(active_agent_shortlist)

    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Format user state (profile, open loops, commitments) into XML for LLM context
def _render_user_state(user_state) -> str:
    try:
        profile = getattr(user_state, "profile", []) or []
        open_loops = getattr(user_state, "open_loops", []) or []
        commitments = getattr(user_state, "commitments", []) or []
    except Exception:
        return "<user_state>\nNone\n</user_state>"

    parts: List[str] = []
    if profile:
        parts.append("<profile>\n" + "\n".join([f"- {escape(str(x))}" for x in profile[:12]]) + "\n</profile>")
    if open_loops:
        lines = []
        for ol in open_loops[:10]:
            if isinstance(ol, dict):
                txt = ol.get("text")
                if isinstance(txt, str) and txt.strip():
                    lines.append(f"- {escape(txt.strip())}")
        if lines:
            parts.append("<open_loops>\n" + "\n".join(lines) + "\n</open_loops>")
    if commitments:
        lines = []
        for c in commitments[:10]:
            if isinstance(c, dict):
                txt = c.get("text")
                due = c.get("due_at")
                if isinstance(txt, str) and txt.strip():
                    suffix = f" (due: {due})" if isinstance(due, str) and due.strip() else ""
                    lines.append(f"- {escape(txt.strip())}{escape(suffix)}")
        if lines:
            parts.append("<commitments>\n" + "\n".join(lines) + "\n</commitments>")

    if not parts:
        return "<user_state>\nNone\n</user_state>"
    return "<user_state>\n" + "\n".join(parts) + "\n</user_state>"


# Render retrieved memory units into XML format for LLM context
def _render_retrieved_memory(units) -> str:
    if not units:
        return "<retrieved_memory>\nNone\n</retrieved_memory>"
    rendered: List[str] = []
    for u in units[:20]:
        try:
            unit_type = escape(getattr(u, "unit_type", "fact"), quote=True)
            text = escape(getattr(u, "text", ""))
            ents = getattr(u, "entities", []) or []
            ent_text = ""
            if ents:
                ent_text = " entities=\"" + escape(", ".join([str(e) for e in ents[:5]]), quote=True) + "\""
            rendered.append(f"<memory_unit type=\"{unit_type}\"{ent_text}>{text}</memory_unit>")
        except Exception:
            continue
    if not rendered:
        return "<retrieved_memory>\nNone\n</retrieved_memory>"
    return "<retrieved_memory>\n" + "\n".join(rendered) + "\n</retrieved_memory>"


# Render both hot and relevant agents for inclusion in the prompt
async def _render_active_agent_shortlist(queries: List[str], conversation_history: str) -> str:
    hot_list = _get_hot_agents()
    relevant_agents = await _get_relevant_agents(queries, conversation_history)

    sections: List[str] = []

    if hot_list:
        rendered_hot = [
            f'<agent name="{escape(name, quote=True)}" />' for name in hot_list
        ]
        sections.append(
            "<recently_used_agents>\n" + "\n".join(rendered_hot) + "\n</recently_used_agents>"
        )

    if relevant_agents:
        unique_relevant = [name for name in relevant_agents if name not in hot_list]
        if unique_relevant:
            rendered_relevant = [
                f'<agent name="{escape(name, quote=True)}" />' for name in unique_relevant
            ]
            sections.append(
                "<relevant_agents>\n" + "\n".join(rendered_relevant) + "\n</relevant_agents>"
            )

    if not sections:
        return "<active_agents>\nNone\n</active_agents>"

    shortlist = "\n\n".join(sections)

    return shortlist


# Get the hot list of most recently used active execution agents
def _get_hot_agents() -> List[str]:
    lru_cache = get_execution_agent_lru_cache()
    results = lru_cache.get_hot_list()
    return results


# Get the most semantically relevant agents based on user query
async def _get_relevant_agents(queries: List[str], conversation_history: str) -> List[str]:
    cleaned_queries = [q.strip() for q in (queries or []) if isinstance(q, str) and q.strip()]
    if not cleaned_queries:
        return []

    from ...config import get_settings

    settings = get_settings()

    agent_semantic_search = get_agent_semantic_search()
    results = agent_semantic_search.semantic_search(
        cleaned_queries,
        top_k=settings.semantic_search_top_k,
        boost_weight=settings.semantic_search_keyword_boost_weight
    )

    return [name for name, score in results]


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
