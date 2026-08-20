"""docstore/graph_hydrate.py — supplements text-search results with graph facts.

Two outputs from the same underlying Neo4j lookup, serving two different
consumers:

  text blocks   fed into the LLM's context, exactly as before this trace
                feature existed — additive, appended after
                format_for_prompt()'s output, fails open to [] with no graph.

  trace         a structured object for the ON-DEMAND visual trace shown in
                the frontend. None when there is nothing to show, which is
                what drives the "hide the button, don't show it disabled"
                behavior on the client — the trace's mere presence/absence
                IS the availability signal, no separate check needed.

Both come from ONE Neo4j query per collection, not two — the visual isn't an
extra round-trip, it's the same data already being fetched for the LLM,
kept in structured form for slightly longer before being thrown away.

Where the candidate entity names come from is worth being explicit about:
NOT a fresh NER pass over the query text. The top text-search hits already
ran through spaCy at ingestion time and carry their own `entities` — chunks
that just matched "does Rachpal carry a driving license" almost certainly
mention "Rachpal Singh" in their entities list already. Seeding the graph
lookup from THAT is more reliable than re-parsing the query in isolation.
"""
from __future__ import annotations

import asyncio
import json

from config import get_settings
from docstore import graph_store, store

cfg = get_settings()


def _entity_names_from_hits(hits: list[dict], limit: int = 8) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for h in hits:
        raw = h.get("entities")
        if not raw:
            continue
        try:
            entities = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            continue
        for e in entities or []:
            text = (e.get("text") or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                names.append(text)
            if len(names) >= limit:
                return names
    return names


def _format_block(collection_name: str, facts: list[dict], stale_note: str) -> str:
    lines = [f"[KNOWLEDGE GRAPH — collection: {collection_name}]{stale_note}"]
    for f in facts:
        lines.append(f"  {f['entity']} ({f['entity_label']}) — related to — "
                     f"{f['related']} ({f['related_label']})")
    return "\n".join(lines)


def _cap_trace_nodes(collections: list[dict], max_nodes: int) -> list[dict]:
    """Trim the VISUAL trace to a total node budget across all collections,
    keeping the highest-weight facts first. The text blocks above are NOT
    capped this way — an LLM tolerates more volume than a diagram does, so
    this only affects what gets drawn, never what the model reads."""
    all_facts = sorted(
        ((c, f) for c in collections for f in c["facts"]),
        key=lambda cf: -cf[1]["weight"])

    kept_names: set[str] = set()
    out_by_coll: dict[str, dict] = {}
    for c, f in all_facts:
        names_needed = {f["entity"].lower(), f["related"].lower()}
        new_names = names_needed - kept_names
        if len(kept_names) + len(new_names) > max_nodes and kept_names:
            continue
        kept_names |= new_names
        bucket = out_by_coll.setdefault(c["collection_id"], {**c, "facts": []})
        bucket["facts"].append(f)

    return [out_by_coll[c["collection_id"]] for c in collections
            if c["collection_id"] in out_by_coll]


def _filter_to_retrieved_docs(facts: list[dict], retrieved_doc_ids: set[str]) -> list[dict]:
    """Keep only facts whose relationship was actually observed in a document
    that got retrieved for THIS query.

    Without this, related_facts() returns collection-wide relationships for
    a matched entity — correct in general, but not what "reasoning graph for
    this answer" should mean. An entity matched from the retrieved chunks
    could have dozens of co-occurrence edges scattered across a 500-document
    collection; showing all of them looks like noise unrelated to the
    citations in front of the person, because most of it genuinely is
    unrelated to this specific answer. This is the step that makes the
    visual actually correspond to what the answer is grounded in.
    """
    if not retrieved_doc_ids:
        return facts
    kept = []
    for f in facts:
        doc_ids = set(f.get("doc_ids") or [])
        if doc_ids & retrieved_doc_ids:
            kept.append(f)
    return kept


def _hydrate_sync(scope_ids: list[str], hits: list[dict]) -> tuple[list[str], dict | None]:
    """The actual work. Genuinely synchronous — every call here (Neo4j via
    graph_store, SQLite via store) is blocking, which is exactly why the
    async wrapper below runs this in a thread rather than calling it
    directly on the event loop."""
    if not scope_ids or not hits or not graph_store.is_available():
        return [], None

    entity_names = _entity_names_from_hits(hits)
    if not entity_names:
        return [], None

    retrieved_doc_ids = {h.get("doc_id") for h in hits if h.get("doc_id")}

    collections: list[dict] = []
    for collection_id in scope_ids:
        state = store.get_graph_state(collection_id)
        if state["status"] not in ("ready", "stale"):
            continue

        raw_facts = graph_store.related_facts(collection_id, entity_names)
        facts = _filter_to_retrieved_docs(raw_facts, retrieved_doc_ids)
        # A cheap but real fallback: if grounding to retrieved docs left
        # nothing (the matched entity's graph relationships all happen to
        # live in OTHER documents this query didn't retrieve), that is
        # honestly "nothing relevant to show" — not "show it anyway".
        facts = facts[:int(getattr(cfg, "graph_hydrate_facts_per_collection", 12))]
        if not facts:
            continue

        with store.conn() as c:
            row = c.execute("SELECT title FROM conversations WHERE id=?",
                            (collection_id,)).fetchone()
        name = row["title"] if row else collection_id

        collections.append({
            "collection_id": collection_id, "collection_name": name,
            "stale": state["stale"], "facts": facts,
        })

    if not collections:
        return [], None

    text_blocks = [
        _format_block(c["collection_name"], c["facts"],
                     " (may be out of date — this collection has changed "
                     "since the graph was last built; treat as approximate)"
                     if c["stale"] else "")
        for c in collections
    ]

    max_nodes = int(getattr(cfg, "graph_trace_max_nodes", 15))
    trace = {"collections": _cap_trace_nodes(collections, max_nodes)}
    return text_blocks, trace


async def hydrate(scope_ids: list[str], hits: list[dict]) -> tuple[list[str], dict | None]:
    """Returns (text_blocks, trace). trace is None when there is nothing to
    show — no graph, Neo4j unavailable, no graph-linked entities found, OR
    hydration took too long and was abandoned.

    Runs the actual (blocking) work in a thread, off the event loop, capped
    by graph_hydrate_timeout_s. This is the fix for graph hydration being
    able to freeze an entire query — before this, is_available()'s first
    connection attempt to Neo4j was a blocking network call sitting directly
    in the async request path, with no bound on how long it could hang the
    whole server for. Now the worst case is exactly graph_hydrate_timeout_s
    seconds, after which this degrades to "no graph" and the answer proceeds
    with whatever text-search context it already has — identical behaviour
    to any other hydration failure.
    """
    if not scope_ids or not hits:
        return [], None
    timeout = float(getattr(cfg, "graph_hydrate_timeout_s", 2.5))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_hydrate_sync, scope_ids, hits), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[graph_hydrate] ⚠️ timed out after {timeout}s (Neo4j slow or "
              f"unreachable) — continuing without graph context for this query.")
        return [], None
    except Exception as e:  # noqa: BLE001 — graph hydration must never break retrieval
        print(f"[graph_hydrate] ⚠️ hydration failed, continuing without it: {e}")
        return [], None
