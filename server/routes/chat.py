from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import ChatHistoryClearResponse, ChatHistoryResponse, ChatRequest
from ..services import get_conversation_log, get_trigger_service, handle_chat_request, get_memory_index

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/send", response_class=JSONResponse, summary="Submit a chat message and receive a completion")
# Handle incoming chat messages and route them to the interaction agent
async def chat_send(
    payload: ChatRequest,
) -> JSONResponse:
    return await handle_chat_request(payload)


@router.get("/history", response_model=ChatHistoryResponse)
# Retrieve the conversation history from the log
def chat_history() -> ChatHistoryResponse:
    log = get_conversation_log()
    return ChatHistoryResponse(messages=log.to_chat_messages())


@router.delete("/history", response_model=ChatHistoryClearResponse)
def clear_history() -> ChatHistoryClearResponse:
    from ..services import (
        get_execution_agent_logs,
        get_agent_roster,
        get_execution_agent_metadata,
        get_execution_agent_lru_cache,
        get_agent_semantic_search,
        get_memory_index,
        get_memory_unit_store,
        get_user_state_store,
    )

    # Clear conversation log
    log = get_conversation_log()
    log.clear()

    # Clear execution agent logs
    execution_logs = get_execution_agent_logs()
    execution_logs.clear_all()

    # Clear agent roster
    roster = get_agent_roster()
    roster.clear()

    # Clear stored triggers
    trigger_service = get_trigger_service()
    trigger_service.clear_all()

    # Clear execution agent metadata
    metadata_store = get_execution_agent_metadata()
    metadata_store.clear_all()

    # Clear LRU cache
    lru_cache = get_execution_agent_lru_cache()
    lru_cache.clear()

    # Clear FAISS index
    semantic_search = get_agent_semantic_search()
    semantic_search.clear()

    # Clear memory unit index
    memory_index = get_memory_index()
    memory_index.clear()

    # Clear memory unit store
    memory_unit_store = get_memory_unit_store()
    memory_unit_store.clear_all()

    # Clear user state store
    user_state_store = get_user_state_store()
    user_state_store.clear()

    return ChatHistoryClearResponse()


__all__ = ["router"]
