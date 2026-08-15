"""agents/doc_retriever.py — worker that searches the user's uploaded documents.

Two paths feed uploaded content into an answer, and both are needed:

  1. A pre-pass in the endpoint retrieves against the raw user query before the
     graph starts, so the supervisor's very first decision is already informed by
     what the user attached. Without it, the supervisor would reliably dispatch a
     web search for something sitting in the PDF in front of it.

  2. This worker, which the supervisor can dispatch mid-run with a *reformulated*
     query. That matters because the useful query is often not the user's
     wording: "does this contradict the policy?" needs a lookup for the policy
     clause, not for the word "contradict".

Returns results on `context` so the existing operator.add reducer appends them,
and on `action_logs` so they surface in the live trace like every other worker.
"""
from __future__ import annotations

from typing import Any, Dict

from docstore import collections as doc_collections
from docstore import session, store
from docstore.retrieve import context_for_query
from workflow.state import DocRetrieveTaskState


async def doc_retriever_node(state: DocRetrieveTaskState) -> Dict[str, Any]:
    query = (state.get("query") or "").strip()
    conversation_id = session.get_conversation()

    print(f"\n[DOC_RETRIEVER] 📚 Searching for: '{query[:120]}'")

    if not conversation_id:
        msg = "📚 Document search skipped — no conversation bound to this run."
        print(f"[DOC_RETRIEVER] ⚠️ {msg}")
        return {"context": [], "action_logs": [msg]}

    if not query:
        return {"context": [],
                "action_logs": ["📚 Document search skipped — empty query."]}

    # Scope is this conversation's own uploads PLUS any attached collections
    # (docstore/collections.py). Checking store.corpus_stats(conversation_id)
    # alone — as this used to — only sees uploads: a chat with no uploads of
    # its own but a collection attached would report zero chunks here and
    # bail before ever calling context_for_query(), even though that function
    # was already scope-aware and would have found the collection's content.
    # That mismatch is what produced "No uploaded documents in this
    # conversation to search" for a conversation that plainly had documents,
    # right after the pre-pass had just retrieved from the same scope.
    scope_ids = doc_collections.scopes_for(conversation_id)
    stats = store.corpus_stats_scoped(scope_ids) if scope_ids else {"chunks": 0}
    if not stats["chunks"]:
        msg = "📚 No documents or attached collections to search in this conversation."
        print(f"[DOC_RETRIEVER] {msg}")
        return {"context": [], "action_logs": [msg]}

    try:
        blocks, hits = await context_for_query(conversation_id, query)
    except Exception as e:  # noqa: BLE001 — a retrieval failure must not kill the run
        msg = f"📚 Document search failed: {type(e).__name__}: {e}"
        print(f"[DOC_RETRIEVER] 🔥 {msg}")
        return {"context": [], "action_logs": [msg]}

    if not hits:
        msg = f"📚 No relevant passages found across {stats['chunks']} chunks."
        print(f"[DOC_RETRIEVER] {msg}")
        return {"context": [], "action_logs": [msg]}

    files = sorted({h.get("file_name", "?") for h in hits})
    top = hits[0].get("score")
    log = (f"📚 Retrieved {len(hits)} passage(s) from {len(files)} document(s) "
           f"[{', '.join(files[:3])}{'…' if len(files) > 3 else ''}], top score {top}.")
    print(f"[DOC_RETRIEVER] ✅ {log}")

    return {"context": blocks, "action_logs": [log]}
