"""docstore/collections.py — named corpora that live outside any one chat.

The problem this solves
-----------------------
Every document, chunk and vector in this application is keyed by
`conversation_id`, and docstore/retrieve.py says so explicitly: "Scope is always
a single conversation. Cross-thread retrieval is deliberately not offered: a
user who uploads a contract in one thread has not consented to it surfacing in
another."

That is the right rule for uploads. It is the wrong rule for a batch job that
ingests ten thousand policy documents from a shared drive — those exist
precisely so that many chats can search them, and there is no conversation they
naturally belong to.

The design
----------
A **collection** is a scope that holds documents but has no messages. It reuses
the `conversations` table with `kind='collection'`, which is a pragmatic choice
worth being explicit about: it means zero changes to the foreign keys and
cascade rules on documents, doc_chunks and chunk_vectors, all of which already
point at `conversations(id)` and already delete cleanly. The alternative — a
separate table with a nullable `collection_id` on three child tables — would
have touched every query in store.py to gain a naming distinction.

`list_conversations()` filters to `kind='chat'`, so collections never appear in
the chat rail.

Consent is preserved by making attachment explicit. A collection is searchable
from a chat only if a row exists in `conversation_collections`. Nothing is
attached implicitly, and an upload into a chat is never promoted to a
collection. The rule changes from "one conversation" to "this conversation plus
what it has explicitly been given", which is still a consent boundary — just one
the operator can widen deliberately.
"""
from __future__ import annotations

import asyncio
from typing import Any

from docstore import store


# ------------------------------------------------------------------ CRUD

def create_collection(name: str, description: str = "") -> dict[str, Any]:
    cid, ts = store.new_id(), store._now()
    with store.conn() as c:
        c.execute(
            "INSERT INTO conversations (id,title,created_at,updated_at,kind,description) "
            "VALUES (?,?,?,?,'collection',?)",
            (cid, (name or "Untitled collection")[:500], ts, ts, description[:2000]),
        )
    return {"id": cid, "name": name, "description": description,
            "created_at": ts, "updated_at": ts,
            "documents": 0, "chunks": 0, "vectors": 0}


def get_collection(collection_id: str) -> dict | None:
    with store.conn() as c:
        r = c.execute("SELECT * FROM conversations WHERE id=? AND kind='collection'",
                      (collection_id,)).fetchone()
    if not r:
        return None
    row = dict(r)
    return {"id": row["id"], "name": row["title"],
            "description": row.get("description") or "",
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            **store.corpus_stats(collection_id)}


def list_collections() -> list[dict]:
    with store.conn() as c:
        rows = c.execute(
            "SELECT id, title, description, created_at, updated_at "
            "FROM conversations WHERE kind='collection' ORDER BY title"
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        out.append({"id": d["id"], "name": d["title"],
                    "description": d.get("description") or "",
                    "created_at": d["created_at"], "updated_at": d["updated_at"],
                    **store.corpus_stats(d["id"])})
    return out


def find_collection_by_name(name: str) -> dict | None:
    with store.conn() as c:
        r = c.execute(
            "SELECT id FROM conversations WHERE kind='collection' AND title=?",
            (name,)).fetchone()
    return get_collection(r["id"]) if r else None


def ensure_collection(name: str, description: str = "") -> dict:
    """Get-or-create by name. Job runs are re-runnable without duplicating scopes."""
    return find_collection_by_name(name) or create_collection(name, description)


def delete_collection(collection_id: str) -> bool:
    """Removes the collection and, by cascade, its documents, chunks and vectors."""
    if not get_collection(collection_id):
        return False
    store.delete_conversation(collection_id)
    return True


def rename_collection(collection_id: str, name: str, description: str | None = None) -> None:
    with store.conn() as c:
        if description is None:
            c.execute("UPDATE conversations SET title=?, updated_at=? "
                      "WHERE id=? AND kind='collection'",
                      (name[:500], store._now(), collection_id))
        else:
            c.execute("UPDATE conversations SET title=?, description=?, updated_at=? "
                      "WHERE id=? AND kind='collection'",
                      (name[:500], description[:2000], store._now(), collection_id))


# ------------------------------------------------------------------ attachment

def attach(conversation_id: str, collection_id: str) -> bool:
    """Make a collection searchable from a chat. Idempotent."""
    if not get_collection(collection_id):
        return False
    with store.conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO conversation_collections "
            "(conversation_id, collection_id, attached_at) VALUES (?,?,?)",
            (conversation_id, collection_id, store._now()),
        )
    return True


def detach(conversation_id: str, collection_id: str) -> None:
    with store.conn() as c:
        c.execute("DELETE FROM conversation_collections "
                  "WHERE conversation_id=? AND collection_id=?",
                  (conversation_id, collection_id))


def attached_ids(conversation_id: str) -> list[str]:
    """Collection ids this chat may search. Order is stable for cache friendliness."""
    with store.conn() as c:
        rows = c.execute(
            "SELECT cc.collection_id FROM conversation_collections cc "
            "JOIN conversations c2 ON c2.id = cc.collection_id AND c2.kind='collection' "
            "WHERE cc.conversation_id=? ORDER BY cc.attached_at",
            (conversation_id,)).fetchall()
    return [r["collection_id"] for r in rows]


def attached(conversation_id: str) -> list[dict]:
    return [c for c in (get_collection(i) for i in attached_ids(conversation_id)) if c]


def scopes_for(conversation_id: str) -> list[str]:
    """The full retrieval scope for a chat: itself, then its attached collections.

    The conversation comes first so that when two scopes hold near-identical
    text, ties in the fused ranking fall toward the document the user attached
    to this thread — which is the one they are more likely to be asking about.
    """
    if not conversation_id:
        return []
    return [conversation_id, *attached_ids(conversation_id)]


# ------------------------------------------------------------------ discovery

async def search_all_collections(query: str, top_k_per_collection: int = 3) -> list[dict]:
    """Search every non-empty collection, without attaching or reading any of
    them into a chat's scope.

    Exists because attachment is deliberately opt-in — nothing is searched
    silently — but that means a person has no way to find out *which*
    collection holds what they're after without guessing. This is the
    discovery step for that: run the query everywhere, show counts and a
    snippet per collection, let the operator attach the ones that matched.
    Nothing here changes what any chat can retrieve; it only reveals metadata
    (which collection matched, roughly how well) plus short excerpts, the same
    information a person would see anyway by attaching and asking one by one.
    """
    from docstore import retrieve

    candidates = [c for c in list_collections() if c["chunks"]]
    if not candidates or not query.strip():
        return []

    async def one(c: dict) -> tuple[dict, list[dict]]:
        hits = await retrieve.search([c["id"]], query, top_k=top_k_per_collection)
        return c, hits

    pairs = await asyncio.gather(*(one(c) for c in candidates))

    out = []
    for c, hits in pairs:
        if not hits:
            continue
        out.append({
            "collection": c,
            "hit_count": len(hits),
            "top_score": hits[0]["score"],
            "snippets": [{"file_name": h.get("file_name"),
                         "text": (h.get("text") or "")[:220],
                         "score": h.get("score")} for h in hits],
        })
    out.sort(key=lambda r: -r["top_score"])
    return out
