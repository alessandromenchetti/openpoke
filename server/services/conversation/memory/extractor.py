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
class MemoryUnitExtractionResult:
    """Output of summarization-time memory unit extraction."""
    memory_units: List[Dict[str, Any]]


_SYSTEM_PROMPT = """
You are an AI assistant's long-term memory UNIT extractor.

You will be given:
- timezone_name: the user's timezone name (e.g., America/Toronto)
- conversation_summary: existing running summary (may be empty)
- new_logs: a batch of log lines that are about to be compressed (each line includes a timestamp)

Your job is to emit a small set of atomic memory_units capturing durable, reusable information.

Output MUST be a single JSON object with a single key:
{
  "memory_units": [
     {"text": string, "unit_type": string, "entities": [string ...], "confidence": number}
  ]
}

Constraints:
- memory_units must be 0 to 8 items.
- unit_type must be one of: preference, decision, commitment, fact, config, other.
- confidence must be between 0.0 and 1.0.

Hard rules:
1) DO NOT invent facts. Use only what is supported by new_logs.
2) Prefer durable items that help future behavior:
   - explicit user preferences ("I prefer…")
   - stable identifiers and facts (name, timezone, key people)
   - confirmed decisions and confirmed commitments (not tentative)
   - important constraints or conventions
3) Avoid duplicating what the trigger/reminder system already tracks.
   - If logs show a reminder trigger was created for an event, do NOT store a memory unit that simply restates the reminder.
4) TIME NORMALIZATION (very important):
   - You MUST NOT write relative time words in memory_units text ("tomorrow", "today", "tonight", "next week").
   - Convert time references to ABSOLUTE dates/times using the timestamp on the relevant log line(s) as the reference point.
   - Use timezone_name for interpretation.
   - If you cannot confidently resolve an absolute time/date, OMIT the time rather than guessing.
   Examples:
     BAD: "Dinner with Ryan tomorrow at 7"
     GOOD: "Dinner with Ryan on 2026-01-27 at 19:00 (America/Toronto)"
5) Keep unit text short but specific (one sentence).
6) Entities should be specific strings (full names/emails/IDs where possible) and only if clearly supported.

Return valid JSON only. No markdown.

""".strip()


def _format_entries(entries: List[LogEntry]) -> str:
    lines: List[str] = []
    for e in entries:
        label = e.tag.replace("_", " ")
        payload = (e.payload or "").strip()
        idx = e.index if e.index >= 0 else "?"
        if payload:
            lines.append(f"[{idx}] {e.timestamp} {label}: {payload}")
        else:
            lines.append(f"[{idx}] {e.timestamp} {label}: (empty)")
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


async def extract_memory_units(
    *,
    previous_summary: str,
    new_entries: List[LogEntry],
    timezone_name: str,
) -> MemoryUnitExtractionResult:
    settings = get_settings()

    content = {
        "conversation_summary": (previous_summary or "").strip() or "None",
        "new_logs": _format_entries(new_entries),
        "timezone_name": timezone_name,
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

    units_obj = obj.get("memory_units")
    if not isinstance(units_obj, list):
        units_obj = []

    memory_units: List[Dict[str, Any]] = [u for u in units_obj if isinstance(u, dict)]
    return MemoryUnitExtractionResult(memory_units=memory_units)


__all__ = ["MemoryUnitExtractionResult", "extract_memory_units"]