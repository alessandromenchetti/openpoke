from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ....logging_config import logger
from ....utils.timezones import now_in_user_timezone
from ..summarization.state import LogEntry
from ...telemetry.trace_context import bind_span_context


@dataclass(frozen=True)
class UserProfileUpdateResult:
    updated: bool
    last_index: int
    error: Optional[str] = None


def _collect_entries(conversation_log) -> List[LogEntry]:
    entries: List[LogEntry] = []
    for index, (tag, timestamp, payload) in enumerate(conversation_log.iter_entries()):
        entries.append(LogEntry(tag=tag, payload=payload, index=index, timestamp=timestamp or None))
    return entries


def _detect_signals(entries: List[LogEntry]) -> Dict[str, Any]:
    """
    Lightweight deterministic signals to prevent redundant commitments.

    Example trigger creation log:
    "[SUCCESS] Reminder Agent: I've successfully created a reminder trigger ... assigned ID ..."
    """
    signals: Dict[str, Any] = {"created_triggers": []}

    for e in entries:
        if e.tag != "agent_message":
            continue
        payload = (e.payload or "")
        if "created" in payload.lower() and "trigger" in payload.lower() and "id" in payload.lower():
            signals["created_triggers"].append(payload.strip())

    return signals


async def update_user_profile_after_turn(
    *,
    conversation_log,
    timezone_name: str,
) -> UserProfileUpdateResult:
    """
    Update durable user profile after the assistant completes a turn.

    Policy:
    - Update after every user-request completion (best-effort).
    - Only use new log lines since last profile update.
    """

    from .state_store import get_user_state_store
    from .user_profile import UserProfileUpdateInput, update_user_profile

    store = get_user_state_store()
    previous = store.load_state()

    last_idx = int(getattr(previous, "last_state_update_index", -1))

    all_entries = _collect_entries(conversation_log)
    new_entries = [e for e in all_entries if e.index > last_idx]

    if not new_entries:
        return UserProfileUpdateResult(updated=False, last_index=last_idx)

    signals = _detect_signals(new_entries)
    now_iso = now_in_user_timezone("%Y-%m-%d %H:%M:%S")

    try:
        with bind_span_context(
            component="interaction_agent",
            purpose="user_profile.update_after_turn",
        ):
            updated_state = await update_user_profile(
                UserProfileUpdateInput(
                    now_iso=now_iso,
                    timezone=timezone_name,
                    previous_state=previous,
                    new_logs=new_entries,
                    signals=signals,
                )
            )

        # NEW: advance the cursor so we only process new logs next time.
        setattr(updated_state, "last_state_update_index", new_entries[-1].index)

        store.write_state(updated_state)

        logger.info(
            "user profile persisted",
            extra={
                "last_state_update_index": getattr(updated_state, "last_state_update_index", None),
                "new_entries": len(new_entries),
            },
        )
        return UserProfileUpdateResult(updated=True, last_index=new_entries[-1].index)

    except Exception as exc:
        logger.warning("user profile update failed", extra={"error": str(exc)})
        return UserProfileUpdateResult(updated=False, last_index=last_idx, error=str(exc))


__all__ = ["update_user_profile_after_turn", "UserProfileUpdateResult"]