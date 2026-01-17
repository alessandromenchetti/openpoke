from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


trace_id_var = contextvars.ContextVar("trace_id", default=None)
root_source_var = contextvars.ContextVar("root_source", default=None)
component_var = contextvars.ContextVar("component", default=None)
agent_name_var = contextvars.ContextVar("agent_name", default=None)
purpose_var = contextvars.ContextVar("purpose", default=None)
current_span_id_var = contextvars.ContextVar("current_span_id", default=None)


@dataclass(frozen=True)
class TraceContext:
    trace_id: Optional[str]
    root_source: Optional[str]
    component: Optional[str]
    agent_name: Optional[str]
    purpose: Optional[str]
    current_span_id: Optional[str]


def new_trace_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def get_trace_context() -> TraceContext:
    return TraceContext(
        trace_id=trace_id_var.get(),
        root_source=root_source_var.get(),
        component=component_var.get(),
        agent_name=agent_name_var.get(),
        purpose=purpose_var.get(),
        current_span_id=current_span_id_var.get(),
    )


@contextmanager
def bind_trace(
    *,
    trace_id: Optional[str] = None,
    root_source: Optional[str] = None,
    component: Optional[str] = None,
    agent_name: Optional[str] = None,
    purpose: Optional[str] = None,
) -> Iterator[str]:
    """Bind trace fields to the current asyncio task context."""
    if trace_id is None:
        trace_id = new_trace_id(root_source or "trace")

    t_trace = trace_id_var.set(trace_id)
    t_root = root_source_var.set(root_source)
    t_comp = component_var.set(component)
    t_agent = agent_name_var.set(agent_name)
    t_purpose = purpose_var.set(purpose)
    t_span = current_span_id_var.set(None)

    try:
        yield trace_id
    finally:
        current_span_id_var.reset(t_span)
        purpose_var.reset(t_purpose)
        agent_name_var.reset(t_agent)
        component_var.reset(t_comp)
        root_source_var.reset(t_root)
        trace_id_var.reset(t_trace)


@contextmanager
def bind_span_context(
    *,
    component: Optional[str] = None,
    agent_name: Optional[str] = None,
    purpose: Optional[str] = None,
) -> Iterator[None]:
    """Temporarily override labeling fields for the current asyncio task context."""
    t_comp = component_var.set(component_var.get() if component is None else component)
    t_agent = agent_name_var.set(agent_name_var.get() if agent_name is None else agent_name)
    t_purpose = purpose_var.set(purpose_var.get() if purpose is None else purpose)

    try:
        yield
    finally:
        purpose_var.reset(t_purpose)
        agent_name_var.reset(t_agent)
        component_var.reset(t_comp)