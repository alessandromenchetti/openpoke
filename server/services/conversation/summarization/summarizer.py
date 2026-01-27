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
        from ..memory import get_memory_unit_store, get_user_state_store  # NEW
        from ..memory.extractor import extract_long_term_memory  # NEW
        from ..memory.index import get_memory_index  # NEW
        from ..memory.utils import ensure_entity_in_registry  # NEW

        def _truncate(s: str, n: int = 160) -> str:  # NEW
            s = (s or "").replace("\n", " ").strip()
            return s if len(s) <= n else (s[: n - 3] + "...")

        logger.info(
            f"[LTM] starting extraction | batch_size={len(batch)} "
            f"(summarizing entries {batch[0].index}..{batch[-1].index})"
        )  # NEW

        user_state_store = get_user_state_store()  # NEW
        previous_user_state = user_state_store.load_state()  # NEW

        logger.info(
            f"[LTM] loaded previous user_state | profile={len(previous_user_state.profile)} "
            f"open_loops={len(previous_user_state.open_loops)} commitments={len(previous_user_state.commitments)} "
            f"entities={len(previous_user_state.entities)}"
        )  # NEW

        with bind_span_context(
            component="conversation_summarizer",
            purpose="conversation.ltm_extraction"
        ):
            extraction = await extract_long_term_memory(  # NEW
                previous_state=previous_user_state,
                previous_summary=state.summary_text,
                new_entries=batch,
            )

        logger.info(
            f"[LTM] extraction complete | new_profile={len(extraction.user_state.profile)} "
            f"new_open_loops={len(extraction.user_state.open_loops)} "
            f"new_commitments={len(extraction.user_state.commitments)} "
            f"new_entities={len(extraction.user_state.entities)} "
            f"memory_units_out={len(extraction.memory_units)}"
        )  # NEW

        # Preview extracted units (so you can see the flow)
        preview_units = []  # NEW
        for mu in extraction.memory_units[:8]:  # NEW
            if isinstance(mu, dict):
                preview_units.append(
                    {
                        "type": _truncate(str(mu.get("unit_type", "fact")), 32),
                        "entities": mu.get("entities", []),
                        "text": _truncate(str(mu.get("text", "")), 140),
                        "conf": mu.get("confidence", None),
                    }
                )
        logger.info(f"[LTM] memory_units preview (up to 8): {preview_units}")  # NEW

        # Keep entity registry in sync with memory units.
        # (If the extractor emits an entity but state omitted it, we still add it.)
        added_entity_mentions = 0  # NEW
        for mu in extraction.memory_units:
            ents = mu.get("entities") if isinstance(mu, dict) else None
            if isinstance(ents, list):
                for ent in ents:
                    if isinstance(ent, str) and ent.strip():
                        ensure_entity_in_registry(extraction.user_state.entities, ent.strip())
                        added_entity_mentions += 1
        logger.info(f"[LTM] synced entity registry from units | entity_mentions_processed={added_entity_mentions}")  # NEW

        # Persist user_state
        user_state_store.write_state(extraction.user_state)  # NEW
        logger.info("[LTM] wrote user_state.json")  # NEW

        # Persist memory units + index embeddings
        mem_store = get_memory_unit_store()  # NEW
        mem_index = get_memory_index()  # NEW

        attempted = 0  # NEW
        inserted_ids = []  # NEW
        inserted_texts = []  # NEW

        for mu in extraction.memory_units:
            if not isinstance(mu, dict):
                continue
            text = mu.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            attempted += 1  # NEW

            unit_type = mu.get("unit_type")
            if not isinstance(unit_type, str) or not unit_type.strip():
                unit_type = "fact"

            ents = mu.get("entities")
            entities = [e for e in (ents or []) if isinstance(e, str) and e.strip()]

            conf = mu.get("confidence")
            confidence = float(conf) if isinstance(conf, (int, float)) else 0.8

            new_id = mem_store.add_unit(
                text=text.strip(),
                unit_type=unit_type.strip(),
                entities=entities,
                confidence=confidence,
            )

            if new_id is not None:
                inserted_ids.append(new_id)
                inserted_texts.append(text.strip())

        deduped = attempted - len(inserted_ids)  # NEW
        logger.info(
            f"[LTM] persisted memory units | attempted={attempted} inserted={len(inserted_ids)} deduped={deduped} "
            f"inserted_ids={inserted_ids}"
        )  # NEW

        if inserted_ids:
            mem_index.add(inserted_ids, inserted_texts)
            logger.info(f"[LTM] indexed embeddings | count={len(inserted_ids)}")  # NEW
        else:
            logger.info("[LTM] indexing skipped (no new units inserted)")  # NEW

    except Exception as exc:
        logger.warning(f"[LTM] update failed (continuing) | error={exc}", exc_info=True)  # NEW

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
