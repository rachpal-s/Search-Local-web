"""routers/conversations.py — thread list, thread load, rename, delete.

Thin by design: every route is a direct projection of docstore.store. Business logic
that belongs to the graph stays in the graph; this module only shapes JSON.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from docstore import store
from observability import trace_url

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class CreateConversation(BaseModel):
    title: str = Field(default="New chat", max_length=500)


class RenameConversation(BaseModel):
    title: str = Field(min_length=1, max_length=500)


@router.get("")
async def list_conversations(q: str | None = Query(default=None, max_length=200),
                             limit: int = Query(default=100, ge=1, le=500)):
    """Thread list for the sidebar, newest activity first."""
    return {"conversations": store.list_conversations(limit=limit, q=q)}


@router.post("", status_code=201)
async def create_conversation(body: CreateConversation | None = None):
    return store.create_conversation((body.title if body else "New chat"))


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Full thread: metadata, messages, attached documents, corpus stats."""
    conv = store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {
        "conversation": conv,
        "messages": store.get_messages(conversation_id),
        "documents": store.list_documents(conversation_id),
        "corpus": store.corpus_stats(conversation_id),
    }


@router.patch("/{conversation_id}")
async def rename_conversation(conversation_id: str, body: RenameConversation):
    if not store.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    store.rename_conversation(conversation_id, body.title)
    return {"id": conversation_id, "title": body.title}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Removes the thread and everything scoped to it, including uploaded files."""
    if not store.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")

    import shutil
    from docstore.ingest import upload_dir
    store.delete_conversation(conversation_id)
    try:
        shutil.rmtree(upload_dir(conversation_id), ignore_errors=True)
    except OSError:
        pass
    return {"status": "deleted", "id": conversation_id}


@router.get("/{conversation_id}/trace-url")
async def conversation_trace_url(conversation_id: str):
    """The LangSmith trace link for this thread, or null when telemetry is
    disabled — the frontend hides the "View trace" link entirely on null
    rather than showing one that goes nowhere. No conversation-existence
    check needed: this is a pure URL template substitution, harmless to
    call for any id including one that never existed."""
    return {"url": trace_url(conversation_id)}
