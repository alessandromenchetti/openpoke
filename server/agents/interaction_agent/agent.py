"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...services.execution import get_agent_roster
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
    active_agent_shortlist = await _render_active_agent_shortlist(user_query, transcript)
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


# Format currently active execution agents into XML tags for LLM awareness
# NOT USED ANYMORE - kept for reference
def _render_active_agents() -> str:
    roster = get_agent_roster()
    roster.load()
    agents = roster.get_agents()

    if not agents:
        return "None"

    rendered: List[str] = []
    for agent_name in agents:
        name = escape(agent_name or "agent", quote=True)
        rendered.append(f'<agent name="{name}" />')

    return "\n".join(rendered)


# Render both hot and relevant agents for inclusion in the prompt
async def _render_active_agent_shortlist(user_query: Optional[str], conversation_history: str) -> str:
    hot_list = _get_hot_agents()
    relevant_agents = await _get_relevant_agents(user_query, conversation_history)

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
async def _get_relevant_agents(user_query: Optional[str], conversation_history: str) -> List[str]:
    if not user_query:
        return []

    from ...config import get_settings
    from ...services.execution.semantic_search import decompose_user_query
    from ...services.telemetry.trace_context import bind_span_context

    settings = get_settings()

    with bind_span_context(
            component="interaction_agent",
            purpose="interaction_agent.query_decomposition",
    ):
        queries = await decompose_user_query(user_query, conversation_history)

    agent_semantic_search = get_agent_semantic_search()
    results = agent_semantic_search.semantic_search(
        queries, top_k=settings.semantic_search_top_k, boost_weight=settings.semantic_search_keyword_boost_weight
    )

    return [name for name, score in results]


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
