from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from ..config import get_settings
from ..services.telemetry.trace_context import current_span_id_var, get_trace_context
from ..services.telemetry.llm_usage import record_llm_usage

OpenRouterBaseURL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload."""

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system),
        "stream": False,
        "usage": {"include": True},
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url.rstrip('/')}/chat/completions"

    ctx = get_trace_context()
    parent_span_id = current_span_id_var.get()
    span_id = uuid.uuid4().hex[:16]
    current_span_id_var.set(span_id)

    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=_headers(api_key=api_key),
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()

            data = response.json()
            duration_ms = int((time.monotonic() - t0) * 1000)

            usage = (data or {}).get("usage") or {}
            record_llm_usage(
                {
                    "trace_id": ctx.trace_id,
                    "root_source": ctx.root_source,
                    "component": ctx.component,
                    "agent_name": ctx.agent_name,
                    "purpose": ctx.purpose,
                    "span_id": span_id,
                    "parent_span_id": parent_span_id,
                    "model": model,
                    "duration_ms": duration_ms,
                    "status": "ok",
                    "http_status": response.status_code,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cost": usage.get("cost") or usage.get("total_cost"),
                }
            )
            return data

    except httpx.HTTPStatusError as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        try:
            err_detail = exc.response.json()
        except Exception:
            err_detail = exc.response.text

        record_llm_usage(
            {
                "trace_id": ctx.trace_id,
                "root_source": ctx.root_source,
                "component": ctx.component,
                "agent_name": ctx.agent_name,
                "purpose": ctx.purpose,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "model": model,
                "duration_ms": duration_ms,
                "status": "error",
                "http_status": exc.response.status_code,
                "error": err_detail,
            }
        )
        _handle_response_error(exc)

    except httpx.HTTPError as exc:
        duration_ms = int((time.monotonic() - t0) * 1000)
        record_llm_usage(
            {
                "trace_id": ctx.trace_id,
                "root_source": ctx.root_source,
                "component": ctx.component,
                "agent_name": ctx.agent_name,
                "purpose": ctx.purpose,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "model": model,
                "duration_ms": duration_ms,
                "status": "error",
                "http_status": None,
                "error": str(exc),
            }
        )
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL"]
