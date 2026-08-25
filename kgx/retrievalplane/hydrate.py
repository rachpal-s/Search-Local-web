"""kgx/retrievalplane/hydrate.py — the read path.

Drop-in replacement for docstore.graph_hydrate.hydrate, with an identical
contract:

    async hydrate(scope_ids, hits) -> (text_blocks, trace | None)

Same fail-open discipline as the original: any failure, any timeout, any
missing Neo4j, and the caller gets ([], None) and the answer proceeds on text
search alone. Graph context is additive; it must never be load-bearing.

WHAT CHANGES VERSUS THE LEGACY PATH
-----------------------------------
Two things, and both matter more than the graph swap itself.

SEEDS. Legacy reads h["entities"] — the doc_chunks column that analysis
showed is 200,541 mentions all labelled PROPN, whose top entries are "The",
"This" and "Text". It then looks those up in a graph built from the same
data. Broken at both ends. This path seeds from the gazetteer instead: real
typed entities, matched with verified spans.

FACTS. Legacy emits "X — related to — Y", an untyped co-occurrence with no
direction and no provenance. This emits the predicate, the direction, the
source document and the tier, so the model can cite and weigh it.

PROVENANCE DEVIATION, STATED DELIBERATELY: legacy filters facts down to the
documents already retrieved. That is safer for grounding but defeats the
purpose — a graph earns its place by surfacing what dense retrieval did NOT
return. So facts here are unfiltered and each carries its source doc_id and
tier instead, which keeps them citable without confining them to what was
already found.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from kgx import config as kgx_config
from kgx.dataplane.gazetteer_matcher import Gazetteer
from kgx.repositories import graph_repo as repo

MAX_SEEDS = 10
MAX_FACTS = 14
TIMEOUT_S = 2.5


@lru_cache(maxsize=1)
def _gaz() -> Gazetteer:
    return Gazetteer.load()


@lru_cache(maxsize=64)
def _namespace_for_scope(scope_id: str) -> str | None:
    """Resolve a scope to its namespace via the collection title. Returns
    None for unmapped scopes rather than guessing — an unmapped collection
    must not silently inherit another domain's graph."""
    try:
        from docstore import store
        with store.conn() as c:
            row = c.execute("SELECT title FROM conversations WHERE id=?",
                            (scope_id,)).fetchone()
        return kgx_config.namespace_for(row["title"]) if row else None
    except Exception:                                          # noqa: BLE001
        return None


def _seed_entities(hits: list[dict]) -> list[str]:
    """Canonical entity names present in the retrieved text."""
    g = _gaz()
    seen: list[str] = []
    for h in hits[:12]:
        text = h.get("text") or ""
        if not text:
            continue
        for m in g.find(text):
            if not m.ambiguous and m.canonical not in seen:
                seen.append(m.canonical)
                if len(seen) >= MAX_SEEDS:
                    return seen
    return seen


def _format(namespace: str, facts: list[dict]) -> str:
    lines = [f"[KNOWLEDGE GRAPH — {namespace}, ontology v"
             f"{kgx_config.get_settings().ontology_version}]"]
    for f in facts:
        neg = "does NOT " if f.get("negated") else ""
        conf = f.get("confidence") or 0.0
        lines.append(
            f"  {f['subject']} ({f['subject_class']}) {neg}{f['predicate']} "
            f"{f['object']} ({f['object_class']})"
            f"  [source: {str(f.get('doc_id') or '?')[:12]}, "
            f"tier: {f.get('tier') or 'unknown'}, conf: {conf:.2f}]")
    return "\n".join(lines)


def _hydrate_sync(scope_ids: list[str], hits: list[dict]) -> tuple[list[str], dict | None]:
    if not repo.is_available():
        return [], None
    seeds = _seed_entities(hits)
    if not seeds:
        return [], None

    blocks: list[str] = []
    trace: dict = {"seeds": seeds, "namespaces": [], "facts": 0}

    for scope_id in scope_ids:
        ns = _namespace_for_scope(scope_id)
        if not ns:
            continue
        ents = repo.find_by_label(ns, seeds, limit=MAX_SEEDS)
        if not ents:
            continue
        facts = repo.neighbourhood(ns, [e["id"] for e in ents], limit=MAX_FACTS)
        if not facts:
            continue
        blocks.append(_format(ns, facts))
        trace["namespaces"].append({"namespace": ns, "entities": len(ents),
                                    "facts": len(facts)})
        trace["facts"] += len(facts)

    return blocks, (trace if blocks else None)


async def hydrate(scope_ids: list[str], hits: list[dict]) -> tuple[list[str], dict | None]:
    if not scope_ids or not hits:
        return [], None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_hydrate_sync, scope_ids, hits), timeout=TIMEOUT_S)
    except asyncio.TimeoutError:
        print(f"[kgx.hydrate] timed out after {TIMEOUT_S}s — continuing without graph.")
        return [], None
    except Exception as e:                                     # noqa: BLE001
        print(f"[kgx.hydrate] failed, continuing without it: {e}")
        return [], None


async def hydrate_compare(scope_ids: list[str], hits: list[dict],
                          legacy) -> tuple[list[str], dict | None]:
    """Run BOTH paths and return both, each labelled.

    This is the mode the whole plane was built for: the same query, the same
    hits, two graphs, side by side, so the quality difference is observed
    rather than argued about. It costs two hydrations per query — fine for an
    admin view or an eval harness, not for the user hot path.
    """
    legacy_task = asyncio.create_task(legacy(scope_ids, hits))
    kgx_task = asyncio.create_task(hydrate(scope_ids, hits))
    (l_blocks, l_trace), (k_blocks, k_trace) = await asyncio.gather(
        legacy_task, kgx_task, return_exceptions=False)

    out: list[str] = []
    if l_blocks:
        out.append("[COMPARE: LEGACY co-occurrence graph]\n" + "\n".join(l_blocks))
    if k_blocks:
        out.append("[COMPARE: ONTOLOGY graph]\n" + "\n".join(k_blocks))
    trace = {"mode": "compare",
             "legacy_facts": (l_trace or {}).get("facts_total")
                             or len(l_blocks),
             "kgx_facts": (k_trace or {}).get("facts", 0),
             "legacy": l_trace, "kgx": k_trace}
    return out, trace
