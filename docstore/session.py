"""rag/session.py — binds the active conversation to the async context.

Worker nodes are dispatched with `Send(agent_name, payload)` and the supervisor
authors that payload, so a node cannot see anything the supervisor didn't think
to put there — and the supervisor has no reason to know about conversation ids.
Rather than widen every worker's TypedDict with a field the LLM would have to
populate correctly on every call, the conversation is bound out-of-band via a
ContextVar, exactly as `workflow/inflight.py` already does for run ids.

Set once per request in the endpoint; every node inside the graph inherits it.
"""
from __future__ import annotations

from contextvars import ContextVar

_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def set_conversation(conversation_id: str | None) -> None:
    _conversation_id.set(conversation_id)


def get_conversation() -> str | None:
    return _conversation_id.get()
