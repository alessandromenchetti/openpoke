from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

from ....config import get_settings
from ....logging_config import logger
from ....openrouter_client import OpenRouterError, request_chat_completion
from .prompt_builder import SummaryPrompt, build_summarization_prompt
from .state import LogEntry, SummaryState
from .working_memory_log import get_working_memory_log
from ...telemetry.trace_context import bind_span_context

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..log import ConversationLog


def _resolve_conversation_log() -> "ConversationLog":
    from ..log import get_conversation_log

    return get_conversation_log()


def _collect_entries(log) -> List[LogEntry]:
    entries: List[LogEntry] = []
    for index, (tag, timestamp, payload) in enumerate(log.iter_entries()):
        entries.append(LogEntry(tag=tag, payload=payload, index=index, timestamp=timestamp or None))
    return entries


async def _call_openrouter(prompt: SummaryPrompt, model: str, api_key: Optional[str]) -> str:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = await request_chat_completion(
                model=model,
                messages=prompt.messages,
                system=prompt.system_prompt,
                api_key=api_key,
            )
            choices = response.get("choices") or []
            if not choices:
                raise OpenRouterError("OpenRouter response missing choices")
            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                return content
            raise OpenRouterError("OpenRouter response missing content")
        except OpenRouterError as exc:
            last_error = exc
            if attempt == 0:
                logger.warning(
                    "conversation summarization attempt failed; retrying",
                    extra={"error": str(exc)},
                )
                continue
            logger.error(
                "conversation summarization failed",
                extra={"error": str(exc)},
            )
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            logger.error(
                "conversation summarization unexpected failure",
                extra={"error": str(exc)},
            )
            break
    if last_error:
        raise last_error
    raise OpenRouterError("Conversation summarization failed")


async def summarize_conversation() -> bool:
    settings = get_settings()
    if not settings.summarization_enabled:
        return False

    conversation_log = _resolve_conversation_log()
    working_memory_log = get_working_memory_log()

    entries = _collect_entries(conversation_log)
    state = working_memory_log.load_summary_state()

    threshold = settings.conversation_summary_threshold
    tail_size = max(settings.conversation_summary_tail_size, 0)

    if threshold <= 0:
        return False

    unsummarized_entries = [entry for entry in entries if entry.index > state.last_index]
    if len(unsummarized_entries) < threshold + tail_size:
        return False

    batch = unsummarized_entries[:threshold]
    cutoff_index = batch[-1].index

    prompt = build_summarization_prompt(state.summary_text, batch)

    logger.info(
        "conversation summarization started",
        extra={
            "entries_total": len(entries),
            "unsummarized": len(unsummarized_entries),
            "batch_size": len(batch),
            "last_index_before": state.last_index,
            "cutoff_index": cutoff_index,
        },
    )

    summary_text = await _call_openrouter(prompt, settings.summarizer_model, settings.openrouter_api_key)
    summary_body = summary_text if summary_text else state.summary_text

    try:
        from ..memory import get_memory_unit_store, get_user_state_store
        from ..memory.extractor import extract_memory_units
        from ..memory.index import get_memory_index
        from ....utils.timezones import get_user_timezone_name

        tz_name = get_user_timezone_name("UTC")

        with bind_span_context(
            component="conversation_summarization",
            purpose="memory_extraction",
        ):
            extraction = await extract_memory_units(
                previous_summary=state.summary_text,
                new_entries=batch,
                timezone_name=tz_name,
            )

        mem_store = get_memory_unit_store()
        mem_index = get_memory_index()
        new_ids = []
        new_texts = []
        for mu in extraction.memory_units:
            if not isinstance(mu, dict):
                continue
            text = mu.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            unit_type = mu.get("unit_type")
            if not isinstance(unit_type, str) or not unit_type.strip():
                unit_type = "fact"
            ents = mu.get("entities")
            entities = [e for e in (ents or []) if isinstance(e, str) and e.strip()]
            conf = mu.get("confidence")
            confidence = float(conf) if isinstance(conf, (int, float)) else 0.7

            new_id = mem_store.add_unit(
                text=text.strip(),
                unit_type=unit_type.strip(),
                entities=entities,
                confidence=confidence,
            )
            if new_id is not None:
                new_ids.append(new_id)
                new_texts.append(text.strip())

        if new_ids:
            mem_index.add(new_ids, new_texts)
    except Exception as exc:
        logger.warning("long-term memory update failed", extra={"error": str(exc)})

    refreshed_entries = _collect_entries(conversation_log)
    remaining_entries = [entry for entry in refreshed_entries if entry.index > cutoff_index]

    new_state = SummaryState(
        summary_text=summary_body,
        last_index=cutoff_index,
        updated_at=datetime.now(timezone.utc),
        unsummarized_entries=remaining_entries,
    )

    working_memory_log.write_summary_state(new_state)

    logger.info(
        "conversation summarization completed",
        extra={
            "last_index_after": new_state.last_index,
            "remaining_unsummarized": len(new_state.unsummarized_entries),
        },
    )
    return True


__all__ = ["summarize_conversation"]
