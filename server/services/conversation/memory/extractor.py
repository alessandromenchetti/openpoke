from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ....config import get_settings
from ....openrouter_client import request_chat_completion
from ..summarization.state import LogEntry
from .models import UserState


@dataclass(frozen=True)
class MemoryExtractionResult:
    user_state: UserState
    memory_units: List[Dict[str, Any]]


_SYSTEM_PROMPT = """
You are an AI assistant's long-term memory curator.

Your job is to (1) maintain a small, durable user_state JSON and (2) emit a few atomic memory_units
based ONLY on the provided state + new logs.

Output MUST be a single JSON object with keys:
- "user_state": object
- "memory_units": array of objects

user_state schema (keep small):
{
  "profile": [string ...],
  "open_loops": [{"text": string, "entities": [string], "status": "open|done"?} ...],
  "commitments": [{"text": string, "entities": [string], "due_at": string?} ...],
  "entities": [{"canonical": string, "type": "person|org|project|id|unknown", "aliases": [string ...]} ...]
}

Rules:
- Do NOT invent facts.
- Prefer stable, durable items (preferences, recurring details, important commitments, identifiers).
- Remove obsolete/completed items when the new logs clearly indicate completion.
- Entities: include full names/emails/IDs where available. Add aliases for shortened forms.
- memory_units: 0-8 items. Each item:
  {"text": string, "unit_type": string, "entities": [string], "confidence": 0.0-1.0}
  Use unit_type from: preference, decision, commitment, fact, config, other.
  Keep text short but specific.
""".strip()


def _format_entries(entries: List[LogEntry]) -> str:
    lines: List[str] = []
    for e in entries:
        label = e.tag.replace("_", " ")
        payload = (e.payload or "").strip()
        idx = e.index if e.index >= 0 else "?"
        if payload:
            lines.append(f"[{idx}] {label}: {payload}")
        else:
            lines.append(f"[{idx}] {label}: (empty)")
    return "\n".join(lines) if lines else "(no new logs)"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


async def extract_long_term_memory(
    previous_state: UserState,
    previous_summary: str,
    new_entries: List[LogEntry],
) -> MemoryExtractionResult:
    settings = get_settings()

    content = {
        "previous_user_state": previous_state.to_dict(),
        "previous_conversation_summary": (previous_summary or "").strip() or "None",
        "new_logs": _format_entries(new_entries),
    }

    messages = [{"role": "user", "content": json.dumps(content, ensure_ascii=False, indent=2)}]

    response = await request_chat_completion(
        model=settings.summarizer_model,
        messages=messages,
        system=_SYSTEM_PROMPT,
        api_key=settings.openrouter_api_key,
    )

    raw = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(raw, str):
        raise RuntimeError("Memory extraction did not return text")

    obj = _extract_json_object(raw.strip())
    if not obj:
        raise RuntimeError("Failed to parse memory extraction JSON")

    state_obj = obj.get("user_state")
    units_obj = obj.get("memory_units")
    if not isinstance(state_obj, dict):
        state_obj = {}
    if not isinstance(units_obj, list):
        units_obj = []

    user_state = UserState.from_dict(state_obj)
    memory_units: List[Dict[str, Any]] = [u for u in units_obj if isinstance(u, dict)]
    return MemoryExtractionResult(user_state=user_state, memory_units=memory_units)


__all__ = ["MemoryExtractionResult", "extract_long_term_memory"]