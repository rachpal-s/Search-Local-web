"""docstore/graph_store.py — thin Neo4j client, fail-soft by construction.

Every function here either succeeds or returns an empty/None result and logs
a warning. Nothing raises past this module into the retrieval or job-running
code, because the entire feature is opt-in and per-collection: a person who
never touches knowledge graphs should never see this module cause a single
different behaviour anywhere else in the app, including when Neo4j is simply
not installed (a genuinely optional dependency) or unreachable.
"""
from __future__ import annotations

from config import get_settings

cfg = get_settings()

_driver = None
_unavailable_reason: str | None = None


def _get_driver():
    """Lazy singleton. Returns None (never raises) if unavailable."""
    global _driver, _unavailable_reason
    if _driver is not None:
        return _driver
    if _unavailable_reason is not None:
        return None
    try:
        from neo4j import GraphDatabase
    except ImportError:
        _unavailable_reason = "the neo4j package is not installed"
        print(f"[graph_store] ⚠️ {_unavailable_reason}; graph features disabled.")
        return None
    try:
        _driver = GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password),
            # Bounds the blocking call ITSELF, not just how long a caller is
            # willing to wait for it. Without this, an unreachable host can
            # take 30s+ (OS-level TCP retry defaults) to give up, and the
            # asyncio.wait_for() timeout in graph_hydrate.py only stops the
            # CALLER from waiting — the underlying thread keeps running for
            # the full duration regardless, piling up in the thread pool
            # every time this happens. Kept slightly under
            # graph_hydrate_timeout_s so the connection attempt itself
            # fails before the outer timeout would have abandoned it anyway.
            connection_timeout=max(1.0, float(getattr(cfg, "graph_hydrate_timeout_s", 2.5)) - 0.5),
        )
        _driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001 — any connection failure, fail soft
        _unavailable_reason = f"{type(e).__name__}: {e}"
        print(f"[graph_store] ⚠️ Neo4j unreachable at {cfg.neo4j_uri} "
              f"({_unavailable_reason}); graph features disabled.")
        _driver = None
    return _driver


def is_available() -> bool:
    return _get_driver() is not None


def unavailable_reason() -> str | None:
    """None if available; otherwise why, for surfacing in a job's error log."""
    _get_driver()
    return _unavailable_reason


def ensure_constraints() -> None:
    """Uniqueness constraint per (collection, label, canonical name).

    Without this, MERGE on a not-yet-indexed property is a full label scan —
    fine at hundreds of nodes, painful at hundreds of thousands. One-time,
    idempotent (IF NOT EXISTS), safe to call at the start of every graph job.
    """
    driver = _get_driver()
    if not driver:
        return
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            s.run(
                "CREATE CONSTRAINT entity_key IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.collection_id, e.label, e.canonical_name) IS UNIQUE"
            )
    except Exception as e:  # noqa: BLE001
        print(f"[graph_store] ⚠️ could not ensure constraints: {e}")


def clear_collection(collection_id: str) -> None:
    """Wipe a collection's nodes/edges before a rebuild, so a rebuild never
    leaves orphaned nodes from documents that were later excluded or removed."""
    driver = _get_driver()
    if not driver:
        return
    with driver.session(database=cfg.neo4j_database) as s:
        s.run("MATCH (e:Entity {collection_id: $cid}) DETACH DELETE e",
              cid=collection_id)


def upsert_entities(collection_id: str, entities: list[dict]) -> int:
    """Batch upsert. Each dict: {canonical_name, label, aliases, doc_ids,
    mention_count}. Returns how many were written, 0 on any failure."""
    driver = _get_driver()
    if not driver or not entities:
        return 0
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            s.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {collection_id: $cid, label: row.label,
                                  canonical_name: row.canonical_name})
                SET e.aliases = row.aliases,
                    e.doc_ids = row.doc_ids,
                    e.mention_count = row.mention_count
                """,
                cid=collection_id, rows=entities,
            )
        return len(entities)
    except Exception as e:  # noqa: BLE001
        print(f"[graph_store] ⚠️ upsert_entities failed: {e}")
        return 0


def upsert_cooccurrence_edges(collection_id: str, edges: list[dict]) -> int:
    """Each dict: {a, b, label_a, label_b, weight, doc_ids}. Undirected —
    written once per unordered pair by the caller."""
    driver = _get_driver()
    if not driver or not edges:
        return 0
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            s.run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {collection_id: $cid, label: row.label_a, canonical_name: row.a})
                MATCH (b:Entity {collection_id: $cid, label: row.label_b, canonical_name: row.b})
                MERGE (a)-[r:CO_OCCURS_WITH]-(b)
                SET r.weight = row.weight, r.doc_ids = row.doc_ids
                """,
                cid=collection_id, rows=edges,
            )
        return len(edges)
    except Exception as e:  # noqa: BLE001
        print(f"[graph_store] ⚠️ upsert_cooccurrence_edges failed: {e}")
        return 0


def collection_stats(collection_id: str) -> dict:
    driver = _get_driver()
    if not driver:
        return {"nodes": 0, "edges": 0}
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            nodes = s.run(
                "MATCH (e:Entity {collection_id: $cid}) RETURN count(e) AS n",
                cid=collection_id).single()["n"]
            edges = s.run(
                "MATCH (:Entity {collection_id: $cid})-[r:CO_OCCURS_WITH]-"
                "(:Entity {collection_id: $cid}) RETURN count(r) AS n",
                cid=collection_id).single()["n"]
        return {"nodes": nodes, "edges": edges // 2}   # undirected, counted twice
    except Exception as e:  # noqa: BLE001
        print(f"[graph_store] ⚠️ collection_stats failed: {e}")
        return {"nodes": 0, "edges": 0}


def related_facts(collection_id: str, entity_names: list[str],
                  max_hops: int = 1, limit: int = 40) -> list[dict]:
    """1-hop neighbourhood for a set of candidate entity names.

    Exact match against canonical_name or an alias (case-folded), not a
    substring CONTAINS match. The entity_resolver already tracks name
    variants properly as aliases — "R Singh" as an alias of "Rachpal Singh"
    — so a fuzzy CONTAINS was redundant with that AND riskier: it could
    match "Rachpal Singh" against an unrelated node whose name happened to
    contain it as a substring (e.g. "Rachpal Singh Sharma").

    `limit` here is deliberately higher than what actually gets shown —
    docstore/graph_hydrate.py filters this down further to only facts whose
    doc_ids overlap with the documents retrieved for the specific query, so
    the caller needs enough raw candidates to filter FROM, not just the
    final display count.

    Returns [] on any failure or if nothing matches — the caller treats an
    empty list identically to "no graph", which is the point: this can never
    be the reason an answer fails.
    """
    driver = _get_driver()
    if not driver or not entity_names:
        return []
    try:
        with driver.session(database=cfg.neo4j_database) as s:
            rows = s.run(
                """
                MATCH (e:Entity {collection_id: $cid})
                WHERE any(name IN $names WHERE
                    toLower(e.canonical_name) = toLower(name)
                    OR any(alias IN e.aliases WHERE toLower(alias) = toLower(name)))
                MATCH (e)-[r:CO_OCCURS_WITH]-(o:Entity {collection_id: $cid})
                RETURN e.canonical_name AS entity, e.label AS entity_label,
                       o.canonical_name AS related, o.label AS related_label,
                       r.weight AS weight, r.doc_ids AS doc_ids
                ORDER BY r.weight DESC LIMIT $limit
                """,
                cid=collection_id, names=entity_names, limit=limit,
            )
            return [dict(r) for r in rows]
    except Exception as e:  # noqa: BLE001
        print(f"[graph_store] ⚠️ related_facts failed: {e}")
        return []
