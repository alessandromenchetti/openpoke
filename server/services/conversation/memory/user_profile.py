from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ....config import get_settings
from ....logging_config import logger
from ....openrouter_client import request_chat_completion
from ..summarization.state import LogEntry
from .models import UserState
from .utils import ensure_entity_in_registry


@dataclass(frozen=True)
class UserProfileUpdateInput:
    """Inputs used to update the durable user profile."""
    now_iso: str
    timezone: str
    previous_state: UserState
    new_logs: List[LogEntry]
    signals: Dict[str, Any]


# NEW: prompt is responsible for removing resolved items.
# Pruning is purely deterministic (caps/dedupe/entity rebuild).
_SYSTEM_PROMPT = """
You are maintaining a small, durable USER PROFILE JSON for a tool-enabled assistant.

You will be given:
- previous_user_state: the current stored user_state JSON
- timezone_name: the user's timezone name (e.g., America/Toronto)
- now: the current date/time in ISO-8601 format in the user's timezone
- new_logs: recent conversation-log lines since the last update (each line includes a timestamp)
- limits: maximum sizes you MUST obey
- signals: system signals (e.g., created_triggers)

Return EXACTLY one JSON object with a single key: "user_state".
No markdown, no commentary.

USER_STATE schema (keep SMALL):
{
  "profile": [string ...],
  "open_loops": [
    {"text": string, "entities": [string ...], "due_at": string?}
  ],
  "commitments": [
    {"text": string, "entities": [string ...], "due_at": string?}
  ]
}

HARD RULES (must follow):
1) DO NOT invent facts. Use only what is supported by the new_logs.
2) Keep ONLY ACTIVE open_loops and commitments.
   - If an item was opened and then resolved/completed in the new_logs, it MUST NOT appear.
   - If a previous open_loop/commitment is resolved in the new_logs, REMOVE it.
3) PROFILE STRICTNESS:
   - The "profile" section MUST NOT contain tasks, plans, scheduled events, reminders, action items, or anything time-bound.
   - If a line contains an action ("check", "email", "remind", "draft", "send", "book", "schedule", "set") OR a time reference ("tomorrow", "today", "tonight", "at 7pm", dates),
     it MUST NOT go in "profile". It belongs in open_loops or commitments (or is omitted if resolved).
   - Profile is ONLY: stable facts + explicitly stated preferences + active projects/goals + collaboration conventions.
4) COMMITMENTS are strict:
   - Only add a commitment if the user explicitly confirmed/committed to a future obligation OR explicitly stated a confirmed future event.
   - Invitations or tentative plans ("ask Ryan if he's free") are NOT commitments; they are open_loops.
5) TRIGGERS/REMINDERS:
   - If logs or signals indicate a reminder trigger was created, DO NOT add a commitment that duplicates that reminder.
   Example trigger evidence:
   "[SUCCESS] Reminder Agent: ... created a reminder trigger ... assigned ID ..."
6) ENTITIES:
   - The "entities" list should contain only entities that appear in ACTIVE open_loops/commitments.
   - If an open_loop/commitment is removed, entities tied only to it should disappear too (you may omit entities entirely and let code rebuild).
7) LIMITS:
   - You MUST obey limits.profile_max, limits.open_loops_max, limits.commitments_max.
   - Order each list by importance/urgency (most important first). If you need to drop items, drop the least important.

TIME NORMALIZATION (very important):
- You MUST NOT write relative time words in stored text ("tomorrow", "today", "tonight", "next week").
- Convert time references to ABSOLUTE dates/times using the timestamps included in new_logs as the reference point.
- If you cannot confidently resolve a date/time, OMIT the time rather than guessing.
- If you include due_at, use ISO-8601 (e.g., "2026-01-27T19:00:00") and interpret it in timezone_name.

Open loop vs commitment examples:
- "Draft email to Ryan is waiting for user confirmation to send" → open_loops
- "User booked dinner reservation with Ryan for 2026-01-27 19:00" → commitments
- "Remind me tomorrow at 5pm" AND trigger created → do NOT store as commitment (redundant)
""".strip()


def _format_entries(entries: List[LogEntry]) -> str:
    lines: List[str] = []
    for e in entries:
        label = e.tag.replace("_", " ")
        ts = e.timestamp or ""
        payload = (e.payload or "").strip()
        idx = e.index if e.index >= 0 else "?"
        stamp = f" {ts}" if ts else ""
        if payload:
            lines.append(f"[{idx}]{stamp} {label}: {payload}")
        else:
            lines.append(f"[{idx}]{stamp} {label}: (empty)")
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


def prune_user_profile(state: UserState) -> UserState:
    """
    Deterministic pruning + entity rebuild.

    NOTE: We do NOT rely on an explicit status field.
    The model is instructed to remove resolved items; pruning is for:
      - deduping
      - hard caps
      - ensuring entities registry stays bounded (derived from loops/commitments)
    """

    def _dedupe_text(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in items:
            if not isinstance(s, str):
                continue
            key = s.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s.strip())
        return out

    state.profile = _dedupe_text([s for s in (state.profile or []) if isinstance(s, str)])[:max_profile]

    def _clean_items(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for d in items or []:
            if not isinstance(d, dict):
                continue
            text = d.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            key = text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ents = d.get("entities")
            entities = [e.strip() for e in (ents or []) if isinstance(e, str) and e.strip()]
            due = d.get("due_at")
            out.append({"text": text.strip(), "entities": entities, "due_at": due if isinstance(due, str) else None})
            if len(out) >= limit:
                break
        return out

    state.open_loops = _clean_items(state.open_loops, max_loops)
    state.commitments = _clean_items(state.commitments, max_commitments)

    # NEW: Entities registry is derived ONLY from open_loops + commitments.
    state.entities = []
    for item in (state.open_loops or []) + (state.commitments or []):
        ents = item.get("entities") if isinstance(item, dict) else None
        if isinstance(ents, list):
            for ent in ents:
                if isinstance(ent, str) and ent.strip():
                    ensure_entity_in_registry(state.entities, ent.strip())

    return state


async def update_user_profile(inputs: UserProfileUpdateInput) -> UserState:
    """LLM-backed update of the durable user profile."""
    settings = get_settings()
    model = getattr(settings, "user_profile_model", getattr(settings, "summarizer_model", "")) or getattr(settings, "summarizer_model", "")
    max_profile = int(getattr(settings, "user_profile_max_profile_lines", 12))
    max_loops = int(getattr(settings, "user_profile_max_open_loops", 10))
    max_commitments = int(getattr(settings, "user_profile_max_commitments", 10))

    content = {
        "previous_user_state": inputs.previous_state.to_dict(),
        "timezone_name": inputs.timezone,
        "now": inputs.now_iso,
        "new_logs": _format_entries(inputs.new_logs),
        "limits": {
            "profile_max": max_profile,
            "open_loops_max": max_loops,
            "commitments_max": max_commitments,
        },
        "signals": inputs.signals or {},
    }

    logger.info(f"New logs used for profile update: {content['new_logs']}")
    logger.info(f"Signals used for profile update: {content['signals']}")

    messages = [{"role": "user", "content": json.dumps(content, ensure_ascii=False, indent=2)}]

    logger.info("user profile update started", extra={"new_logs": len(inputs.new_logs), "model": model})

    response = await request_chat_completion(
        model=model,
        messages=messages,
        system=_SYSTEM_PROMPT,
        api_key=settings.openrouter_api_key,
    )

    raw = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    if not isinstance(raw, str):
        raise RuntimeError("User profile update did not return text")

    obj = _extract_json_object(raw.strip())
    if not obj:
        raise RuntimeError("Failed to parse user profile JSON")

    state_obj = obj.get("user_state")
    if not isinstance(state_obj, dict):
        raise RuntimeError("User profile JSON missing user_state")

    new_state = UserState.from_dict(state_obj)
    pruned = new_state
    # pruned = prune_user_profile(new_state)

    logger.info(
        "user profile update completed",
        extra={
            "profile": len(pruned.profile or []),
            "open_loops": len(pruned.open_loops or []),
            "commitments": len(pruned.commitments or []),
            "entities": len(getattr(pruned, "entities", []) or []),
        },
    )

    return pruned


__all__ = ["UserProfileUpdateInput", "update_user_profile", "prune_user_profile"]