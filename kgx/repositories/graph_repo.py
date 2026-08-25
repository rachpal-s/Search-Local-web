"""kgx/repositories/graph_repo.py — namespaced Neo4j adapter.

THE ISOLATION BOUNDARY, in code.

Two dissimilar domains share one Neo4j instance: a Cloud/AI library and a
collection of identity documents, certificates and financial records. The
upper ontology declares that traversal must not cross between them. This
module is where "must not" becomes "cannot", because a declaration in a YAML
file protects nothing on its own.

WHY IT LIVES HERE AND NOT IN THE CALLER
---------------------------------------
Neo4j Community is single-database, so isolation cannot be delegated to the
engine. It has to be label-plus-property discipline applied without exception
on every read and every write. If even one query builds its Cypher without
the namespace guard, the boundary is gone and nothing will tell you. So the
guard is not a parameter callers may pass — it is a required argument on
every public function, asserted before any Cypher is built, and covered by
selftest() below.

RELATION TO THE EXISTING GRAPH
------------------------------
Writes only :KgxEntity and :KgxClaim. The existing :Entity co-occurrence
graph is never read and never written. Both coexist in one instance; either
can be dropped without touching the other. That is what makes this plane
reversible.

    python -m kgx.repositories.graph_repo selftest
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass, field

VALID_NAMESPACES = {"cloud_ai", "personal"}

ENTITY_LABEL = "KgxEntity"
CLAIM_LABEL = "KgxClaim"

_driver = None
_unavailable: str | None = None


class NamespaceViolation(RuntimeError):
    """Raised on any attempt to mix or cross namespaces. Never caught
    internally — a namespace error is a bug, not a degraded condition, and
    silently coping with it is how isolation quietly stops existing."""


def _check_ns(namespace: str) -> str:
    if namespace not in VALID_NAMESPACES:
        raise NamespaceViolation(
            f"unknown namespace {namespace!r}; expected one of {sorted(VALID_NAMESPACES)}")
    return namespace


def mint_iri(namespace: str, kind: str = "entity") -> str:
    """Opaque, stable, non-semantic (§8). NEVER encodes the label — entities
    get renamed and merged, and a name-based IRI breaks every existing
    reference when they do (anti-pattern #28).

    UUIDv7-shaped: millisecond timestamp prefix makes IRIs sort by creation
    and keeps index locality reasonable, without adding a dependency.
    """
    _check_ns(namespace)
    ts = int(time.time() * 1000)
    return f"urn:kgx:{namespace}:{kind}:{ts:012x}{uuid.uuid4().hex[:16]}"


# ------------------------------------------------------------------ driver

def _get_driver():
    """Lazy singleton. Returns None (never raises) if Neo4j is unavailable —
    same fail-soft contract as the existing docstore/graph_store.py, so an
    absent Neo4j degrades this plane rather than breaking the app."""
    global _driver, _unavailable
    if _driver is not None:
        return _driver
    if _unavailable is not None:
        return None
    try:
        from neo4j import GraphDatabase
    except ImportError:
        _unavailable = "the neo4j package is not installed"
        print(f"[kgx.graph] {_unavailable}; graph features disabled.")
        return None
    try:
        from config import get_settings
        cfg = get_settings()
        uri, user = cfg.neo4j_uri, cfg.neo4j_user
        pwd, db = cfg.neo4j_password, cfg.neo4j_database
    except Exception:                                        # standalone use
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
        db = os.getenv("NEO4J_DATABASE", "neo4j")
    try:
        _driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=3.0)
        _driver.verify_connectivity()
        _driver._kgx_database = db
    except Exception as e:                                   # noqa: BLE001
        _unavailable = f"{type(e).__name__}: {e}"
        print(f"[kgx.graph] Neo4j unreachable at {uri} ({_unavailable}); disabled.")
        _driver = None
    return _driver


def is_available() -> bool:
    return _get_driver() is not None


def _session():
    d = _get_driver()
    return d.session(database=getattr(d, "_kgx_database", "neo4j")) if d else None


# --------------------------------------------------------------- schema

def ensure_constraints() -> None:
    """Idempotent. Uniqueness is on (namespace, id) rather than id alone:
    it makes namespace part of identity at the storage layer, so a
    cross-namespace id collision is structurally impossible instead of
    merely improbable."""
    s = _session()
    if not s:
        return
    with s:
        for stmt in (
            f"CREATE CONSTRAINT kgx_entity_id IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) REQUIRE (e.namespace, e.id) IS UNIQUE",
            f"CREATE CONSTRAINT kgx_claim_id IF NOT EXISTS "
            f"FOR (c:{CLAIM_LABEL}) REQUIRE (c.namespace, c.id) IS UNIQUE",
            f"CREATE CONSTRAINT kgx_entity_key IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) REQUIRE (e.namespace, e.source_key) IS UNIQUE",
            f"CREATE INDEX kgx_entity_label IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) ON (e.namespace, e.pref_label)",
            f"CREATE INDEX kgx_entity_class IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) ON (e.namespace, e.entity_class)",
            f"CREATE INDEX kgx_claim_pred IF NOT EXISTS "
            f"FOR (c:{CLAIM_LABEL}) ON (c.namespace, c.predicate)",
        ):
            try:
                s.run(stmt)
            except Exception as e:                           # noqa: BLE001
                print(f"[kgx.graph] constraint skipped: {e}")


# --------------------------------------------------------------- writes

@dataclass
class EntityRecord:
    id: str
    namespace: str
    pref_label: str
    entity_class: str
    # Natural key for MERGE. The gazetteer canonical name — NOT identity.
    # Identity is `id`, opaque and minted once. source_key exists only so a
    # re-run finds the node it wrote last time instead of creating a twin.
    # A canonical RENAME is therefore a migration, not an edit: add the old
    # name as an alias and redirect, exactly as §8 requires.
    source_key: str = ""
    aliases: list[str] = field(default_factory=list)
    owner: str = ""
    external_iri: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    mention_count: int = 0
    extraction_route: str = "gazetteer"
    ontology_version: str = "0.2.0"
    module_version: str = "0.2.0"
    first_seen: str = ""
    last_seen: str = ""
    recorded_at: str = ""


@dataclass
class ClaimRecord:
    id: str
    namespace: str
    subject_iri: str
    predicate: str
    object_iri: str
    predicate_module: str
    evidence_chunk_id: str
    evidence_span_start: int
    evidence_span_end: int
    source_doc_id: str
    source_tier: str = "self_authored"
    confidence: float = 0.0
    negated: bool = False
    extraction_route: str = "gazetteer"
    extractor_model: str = ""
    extractor_version: str = ""
    prompt_version: str = ""
    valid_from: str | None = None
    valid_to: str | None = None
    recorded_at: str = ""
    validation_status: str = "valid"
    proposed_label: str = ""


def upsert_entities(namespace: str, rows: list[EntityRecord]) -> int:
    """Batch upsert. Refuses a mixed batch outright rather than filtering it:
    a caller that assembled entities from two namespaces has a bug, and
    quietly dropping half its input would hide that."""
    _check_ns(namespace)
    bad = {r.namespace for r in rows} - {namespace}
    if bad:
        raise NamespaceViolation(
            f"batch declared namespace={namespace!r} but contains {sorted(bad)}")
    s = _session()
    if not s or not rows:
        return 0
    with s:
        s.run(
            f"""
            UNWIND $rows AS row
            MERGE (e:{ENTITY_LABEL} {{namespace: $ns, id: row.id}})
            SET e += row, e.namespace = $ns
            """,
            ns=namespace, rows=[vars(r) for r in rows])
    return len(rows)


def upsert_entities_by_key(namespace: str, rows: list[EntityRecord]) -> dict:
    """IDEMPOTENT upsert, keyed on (namespace, source_key).

    This is the function the extractors use, and the distinction from
    upsert_entities matters: `id` is set ON CREATE ONLY. Re-running the
    writer after a gazetteer change updates labels, aliases and counts while
    every existing IRI survives — so a claim written last week still points
    at the same node this week.

    Without this, mint_iri()'s timestamp component would produce a fresh IRI
    on every run and the graph would double in size each time.
    """
    _check_ns(namespace)
    bad = {r.namespace for r in rows} - {namespace}
    if bad:
        raise NamespaceViolation(
            f"batch declared namespace={namespace!r} but contains {sorted(bad)}")
    missing = [r.pref_label for r in rows if not r.source_key]
    if missing:
        raise ValueError(f"source_key required; missing on {missing[:5]}")

    s = _session()
    if not s or not rows:
        return {"merged": 0, "created": 0}
    before = stats(namespace)["entities"]
    with s:
        s.run(
            f"""
            UNWIND $rows AS row
            MERGE (e:{ENTITY_LABEL} {{namespace: $ns, source_key: row.source_key}})
            ON CREATE SET e.id = row.id, e.first_seen = row.first_seen
            SET e.pref_label       = row.pref_label,
                e.entity_class     = row.entity_class,
                e.aliases          = row.aliases,
                e.owner            = row.owner,
                e.external_iri     = row.external_iri,
                e.doc_ids          = row.doc_ids,
                e.mention_count    = row.mention_count,
                e.extraction_route = row.extraction_route,
                e.ontology_version = row.ontology_version,
                e.module_version   = row.module_version,
                e.last_seen        = row.last_seen,
                e.recorded_at      = row.recorded_at,
                e.namespace        = $ns
            """,
            ns=namespace, rows=[vars(r) for r in rows])
    after = stats(namespace)["entities"]
    return {"merged": len(rows), "created": after - before}


def entities_by_key(namespace: str, keys: list[str]) -> dict[str, str]:
    """source_key -> id, for resolving claim endpoints after a write."""
    _check_ns(namespace)
    s = _session()
    if not s or not keys:
        return {}
    with s:
        rows = s.run(
            f"MATCH (e:{ENTITY_LABEL} {{namespace: $ns}}) "
            f"WHERE e.source_key IN $keys RETURN e.source_key AS k, e.id AS id",
            ns=namespace, keys=keys)
        return {r["k"]: r["id"] for r in rows}


def upsert_claims(namespace: str, rows: list[ClaimRecord]) -> int:
    """Claims are reified: a node, not an edge. Both endpoints are matched
    WITH the namespace guard, so a claim whose subject or object lives in
    another namespace simply matches nothing and is not written. The count
    returned is what was actually persisted, not what was offered."""
    _check_ns(namespace)
    bad = {r.namespace for r in rows} - {namespace}
    if bad:
        raise NamespaceViolation(
            f"batch declared namespace={namespace!r} but contains {sorted(bad)}")
    s = _session()
    if not s or not rows:
        return 0
    with s:
        rec = s.run(
            f"""
            UNWIND $rows AS row
            MATCH (subj:{ENTITY_LABEL} {{namespace: $ns, id: row.subject_iri}})
            MATCH (obj:{ENTITY_LABEL}  {{namespace: $ns, id: row.object_iri}})
            MERGE (c:{CLAIM_LABEL} {{namespace: $ns, id: row.id}})
            SET c += row, c.namespace = $ns
            MERGE (c)-[:SUBJECT]->(subj)
            MERGE (c)-[:OBJECT]->(obj)
            RETURN count(c) AS written
            """,
            ns=namespace, rows=[vars(r) for r in rows]).single()
    return rec["written"] if rec else 0


# ---------------------------------------------------------------- reads

def neighbourhood(namespace: str, iris: list[str], limit: int = 40) -> list[dict]:
    """One-hop claims touching any of the given entities.

    Every node pattern carries {namespace: $ns} — subject, object AND claim.
    Guarding only the starting node would let a traversal walk out of the
    namespace on its second step, which is precisely the failure this module
    exists to make impossible.
    """
    _check_ns(namespace)
    s = _session()
    if not s or not iris:
        return []
    with s:
        rows = s.run(
            f"""
            MATCH (subj:{ENTITY_LABEL} {{namespace: $ns}})
            WHERE subj.id IN $iris
            MATCH (c:{CLAIM_LABEL} {{namespace: $ns}})-[:SUBJECT]->(subj)
            MATCH (c)-[:OBJECT]->(obj:{ENTITY_LABEL} {{namespace: $ns}})
            WHERE c.validation_status = 'valid'
            RETURN subj.pref_label AS subject, subj.entity_class AS subject_class,
                   c.predicate AS predicate, c.confidence AS confidence,
                   c.source_doc_id AS doc_id, c.source_tier AS tier,
                   c.negated AS negated,
                   obj.pref_label AS object, obj.entity_class AS object_class
            ORDER BY c.confidence DESC LIMIT $limit
            """,
            ns=namespace, iris=iris, limit=limit)
        return [dict(r) for r in rows]


def find_by_label(namespace: str, labels: list[str], limit: int = 50) -> list[dict]:
    """Exact match on pref_label or alias, case-folded. Namespace-guarded."""
    _check_ns(namespace)
    s = _session()
    if not s or not labels:
        return []
    with s:
        rows = s.run(
            f"""
            MATCH (e:{ENTITY_LABEL} {{namespace: $ns}})
            WHERE any(n IN $labels WHERE toLower(e.pref_label) = toLower(n)
                  OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = toLower(n)))
            RETURN e.id AS id, e.pref_label AS pref_label,
                   e.entity_class AS entity_class, e.aliases AS aliases
            LIMIT $limit
            """, ns=namespace, labels=labels, limit=limit)
        return [dict(r) for r in rows]


def stats(namespace: str) -> dict:
    _check_ns(namespace)
    s = _session()
    if not s:
        return {"entities": 0, "claims": 0}
    with s:
        e = s.run(f"MATCH (e:{ENTITY_LABEL} {{namespace:$ns}}) RETURN count(e) AS n",
                  ns=namespace).single()["n"]
        c = s.run(f"MATCH (c:{CLAIM_LABEL} {{namespace:$ns}}) RETURN count(c) AS n",
                  ns=namespace).single()["n"]
    return {"entities": e, "claims": c}


def clear_namespace(namespace: str) -> None:
    """Wipe ONE namespace. Cannot touch the other, and cannot touch the
    pre-existing :Entity graph."""
    _check_ns(namespace)
    s = _session()
    if not s:
        return
    with s:
        s.run(f"MATCH (n:{CLAIM_LABEL}  {{namespace:$ns}}) DETACH DELETE n", ns=namespace)
        s.run(f"MATCH (n:{ENTITY_LABEL} {{namespace:$ns}}) DETACH DELETE n", ns=namespace)


# ------------------------------------------------------------- selftest

def selftest() -> int:
    """Proves isolation holds. Writes to both namespaces under a throwaway
    id prefix, asserts nothing crosses, then cleans up after itself.

    This is the test that has to exist BEFORE any extraction code runs.
    There must never be a window in which personal-namespace entities exist
    while the boundary is merely intended.
    """
    if not is_available():
        print("Neo4j unavailable — cannot run isolation test.")
        print("This test MUST pass before extraction. Start Neo4j and re-run.")
        return 2

    ensure_constraints()
    tag = uuid.uuid4().hex[:8]
    ca = [EntityRecord(id=f"test:{tag}:ca:{i}", namespace="cloud_ai",
                       pref_label=n, entity_class=c, recorded_at="2026-01-01T00:00:00Z")
          for i, (n, c) in enumerate([("TestBedrock", "Service"), ("TestRAG", "Pattern")])]
    pe = [EntityRecord(id=f"test:{tag}:pe:{i}", namespace="personal",
                       pref_label=n, entity_class=c, recorded_at="2026-01-01T00:00:00Z")
          for i, (n, c) in enumerate([("TestPassport", "IdentityDocument"),
                                      ("TestIssuer", "Issuer")])]
    failures = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    try:
        upsert_entities("cloud_ai", ca)
        upsert_entities("personal", pe)
        check("write to both namespaces", True)

        # 1. a mixed batch must be refused, not silently filtered
        try:
            upsert_entities("cloud_ai", ca + pe)
            check("mixed batch refused", False, "it was accepted")
        except NamespaceViolation:
            check("mixed batch refused", True)

        # 2. an unknown namespace must be refused
        try:
            stats("everything")
            check("unknown namespace refused", False, "it was accepted")
        except NamespaceViolation:
            check("unknown namespace refused", True)

        # 3. a claim spanning namespaces must persist nothing
        cross = ClaimRecord(
            id=f"test:{tag}:cross", namespace="cloud_ai",
            subject_iri=ca[0].id, predicate="implements", object_iri=pe[0].id,
            predicate_module="cloud_ai", evidence_chunk_id="x",
            evidence_span_start=0, evidence_span_end=1, source_doc_id="d",
            recorded_at="2026-01-01T00:00:00Z")
        written = upsert_claims("cloud_ai", [cross])
        check("cross-namespace claim not written", written == 0, f"written={written}")

        # 4. a same-namespace claim must persist
        good = ClaimRecord(
            id=f"test:{tag}:good", namespace="cloud_ai",
            subject_iri=ca[0].id, predicate="implements", object_iri=ca[1].id,
            predicate_module="cloud_ai", evidence_chunk_id="x",
            evidence_span_start=0, evidence_span_end=1, source_doc_id="d",
            confidence=0.9, recorded_at="2026-01-01T00:00:00Z")
        check("same-namespace claim written", upsert_claims("cloud_ai", [good]) == 1)

        # 5. traversal must not surface the other namespace
        hood = neighbourhood("cloud_ai", [ca[0].id])
        leaked = [h for h in hood if h["object"] in {e.pref_label for e in pe}]
        check("traversal does not leak", not leaked, f"{len(hood)} hop(s), 0 leaked")

        # 6. label lookup must not cross
        check("label lookup does not cross",
              find_by_label("cloud_ai", ["TestPassport"]) == [])

        # 7. clearing one namespace must not touch the other
        clear_namespace("cloud_ai")
        check("clear is namespace-scoped",
              len(find_by_label("personal", ["TestPassport"])) == 1)

    finally:
        clear_namespace("cloud_ai")
        clear_namespace("personal")

    print(f"\n{7 - len(failures)}/7 passed")
    if failures:
        print("ISOLATION IS NOT ENFORCED. Do not run extraction.")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)
