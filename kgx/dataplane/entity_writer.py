"""kgx/dataplane/entity_writer.py — mentions to typed entities.

The first module in this plane that writes to the graph. No LLM: every node
it produces comes from a curated gazetteer match with a verified character
span, so the output is auditable line by line.

WHY THIS RUNS ALONE, BEFORE THE RELATION PASS
---------------------------------------------
Relations are built on top of entities. If canonicalization is wrong, or
ambiguity resolves badly, or personal-namespace documents leak in, then
every claim extracted afterwards inherits the fault — and you would discover
it two GPU-hours later instead of two minutes in. Running the entity layer
alone gives a complete typed graph to inspect for the cost of a coffee.

It is also independently useful. A 440-entity typed graph with real classes
and observed aliases already answers more than the 14K-node co-occurrence
graph it replaces, whose top entities were "The", "This" and "Text".

SCOPE: cloud_ai namespace only. The personal collection needs a different
extractor entirely — dates, typed identifiers and issuers, no gazetteer —
and writing Service-classed nodes into the personal namespace would violate
its module's class list. That extractor is a separate build.

DRY RUN BY DEFAULT. Nothing is written without --commit.

    python -m kgx.dataplane.entity_writer                 # report only
    python -m kgx.dataplane.entity_writer --commit
    python -m kgx.dataplane.entity_writer --commit --collection Library
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from kgx import config as kgx_config
from kgx.dataplane.gazetteer_matcher import Gazetteer
from kgx.repositories import graph_repo as repo

OUT_DIR = Path("kgx/out")
BATCH = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _superseded(path: Path) -> set[str]:
    """doc_ids retired by the duplicate scan. Extracting from a superseded
    document inflates mention counts for content a newer version already
    carries — the same distortion that made the old co-occurrence graph a
    popularity contest duplicates won."""
    if not path.exists():
        print(f"  WARNING: no duplicate report at {path}")
        print("  Superseded documents will NOT be excluded; counts will be inflated.")
        return set()
    out: set[str] = set()
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    hdr = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == len(hdr):
            row = dict(zip(hdr, parts))
            if row.get("role") in ("SUPERSEDED", "ABSORBED"):
                out.add(row.get("doc_id", ""))
    return out - {""}


def _scopes(collection: str | None):
    """Resolve collections and their namespaces. Refuses anything unmapped —
    config.namespace_for fails closed by design."""
    from docstore import store
    with store.conn() as c:
        if collection:
            rows = c.execute(
                "SELECT id,title FROM conversations WHERE id=? OR title=?",
                (collection, collection)).fetchall()
            if not rows:
                sys.exit(f"no collection {collection!r}")
        else:
            rows = c.execute("SELECT id,title FROM conversations "
                             "WHERE kind='collection' ORDER BY title").fetchall()
    out = []
    for r in rows:
        try:
            ns = kgx_config.namespace_for(r["title"])
        except kgx_config.ConfigError as e:
            print(f"  SKIP {r['title']!r}: {e}")
            continue
        out.append((r["id"], r["title"], ns))
    return out


def build(collection: str | None, dupes_path: Path):
    """Aggregate verified mentions into entity records. Pure read."""
    from docstore import store

    scopes = [s for s in _scopes(collection) if s[2] == "cloud_ai"]
    skipped_ns = [s for s in _scopes(collection) if s[2] != "cloud_ai"]
    for _, title, ns in skipped_ns:
        print(f"  SKIP {title!r} (namespace={ns}; needs the personal extractor)")
    if not scopes:
        sys.exit("no cloud_ai collections in scope")
    print("building from: " + ", ".join(t for _, t, _ in scopes))

    skip_docs = _superseded(dupes_path)
    print(f"superseded documents excluded: {len(skip_docs)}")

    ids = [i for i, _, _ in scopes]
    ph = ",".join("?" * len(ids))
    with store.conn() as c:
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    g = Gazetteer.load()
    agg: dict[str, dict] = {}
    ambiguous: dict[str, dict] = {}
    chunks_used = 0

    for r in rows:
        if r["doc_id"] in skip_docs:
            continue
        chunks_used += 1
        for m in g.find(r["text"] or "", r["chunk_id"], r["doc_id"]):
            # Unresolved mentions are QUARANTINED, not written. "Vault" means
            # both HashiCorp Vault and Azure Key Vault; committing a guess
            # would silently fuse two entities and every claim about either.
            bucket = ambiguous if m.ambiguous else agg
            rec = bucket.setdefault(m.canonical, {
                "class": m.entity_class, "owner": m.owner,
                "surfaces": Counter(), "docs": set(), "mentions": 0,
                "candidates": m.candidates,
            })
            rec["surfaces"][m.surface] += 1
            rec["docs"].add(m.doc_id)
            rec["mentions"] += 1

    now = _now()
    records = []
    for canonical, rec in sorted(agg.items()):
        # Observed surface forms become the alias set — a record of how the
        # corpus actually writes each term, and the query-expansion
        # dictionary at retrieval time (§8).
        surfaces = [s for s, _ in rec["surfaces"].most_common()
                    if s.lower() != canonical.lower()]
        records.append(repo.EntityRecord(
            id=repo.mint_iri("cloud_ai"),        # used ON CREATE only
            namespace="cloud_ai",
            source_key=canonical,
            pref_label=canonical,
            entity_class=rec["class"],
            owner=rec["owner"],
            aliases=surfaces,
            doc_ids=sorted(rec["docs"]),
            mention_count=rec["mentions"],
            extraction_route="gazetteer",
            ontology_version=kgx_config.get_settings().ontology_version,
            module_version=kgx_config.get_settings().module_versions["cloud_ai"],
            first_seen=now, last_seen=now, recorded_at=now,
        ))
    return records, ambiguous, chunks_used, len(rows)


def report(records, ambiguous, chunks_used, chunks_total) -> None:
    by_class = Counter(r.entity_class for r in records)
    docs = {d for r in records for d in r.doc_ids}
    total_mentions = sum(r.mention_count for r in records)

    print(f"\n=== ENTITY BUILD ===")
    print(f"chunks in scope        {chunks_total:>8,}")
    print(f"chunks used            {chunks_used:>8,}")
    print(f"documents represented  {len(docs):>8,}")
    print(f"verified mentions      {total_mentions:>8,}")
    print(f"entities to write      {len(records):>8,}")
    print(f"quarantined (ambiguous){len(ambiguous):>8,}  <-- NOT written")

    print("\n--- entities by class ---")
    for cls, n in by_class.most_common():
        print(f"  {cls:<14}{n:>5}")

    print("\n--- top 20 by document spread ---")
    for r in sorted(records, key=lambda r: -len(r.doc_ids))[:20]:
        al = f"  aliases={r.aliases[:2]}" if r.aliases else ""
        print(f"  {r.pref_label[:36]:<38}{r.entity_class:<13}"
              f"{len(r.doc_ids):>4} docs{al}")

    singles = [r for r in records if len(r.doc_ids) == 1]
    print(f"\nentities in exactly one document: {len(singles)}"
          f"  (curated gazetteer, so these are real, not artifacts)")

    if ambiguous:
        print("\n--- quarantined for disambiguation ---")
        for k, v in sorted(ambiguous.items(), key=lambda kv: -len(kv[1]["docs"])):
            print(f"  {k:<16}{len(v['docs']):>4} docs   candidates={v['candidates']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "entities_preview.tsv"
    with p.open("w", encoding="utf-8") as f:
        f.write("pref_label\tclass\towner\tdocs\tmentions\taliases\n")
        for r in sorted(records, key=lambda r: -len(r.doc_ids)):
            f.write(f"{r.pref_label}\t{r.entity_class}\t{r.owner}\t"
                    f"{len(r.doc_ids)}\t{r.mention_count}\t{'|'.join(r.aliases)}\n")
    print(f"\nwrote {p}")

    q = OUT_DIR / "entities_quarantined.tsv"
    with q.open("w", encoding="utf-8") as f:
        f.write("surface\tdocs\tmentions\tcandidates\n")
        for k, v in ambiguous.items():
            f.write(f"{k}\t{len(v['docs'])}\t{v['mentions']}\t{'|'.join(v['candidates'])}\n")
    print(f"wrote {q}")


def commit(records) -> None:
    if not repo.is_available():
        sys.exit("Neo4j unavailable — cannot commit.")
    repo.ensure_constraints()
    before = repo.stats("cloud_ai")
    created = merged = 0
    for i in range(0, len(records), BATCH):
        res = repo.upsert_entities_by_key("cloud_ai", records[i:i + BATCH])
        merged += res["merged"]
        created += res["created"]
        print(f"  batch {i // BATCH + 1}: merged={res['merged']} created={res['created']}")
    after = repo.stats("cloud_ai")

    print(f"\n=== COMMITTED ===")
    print(f"entities before  {before['entities']:>6,}")
    print(f"entities after   {after['entities']:>6,}")
    print(f"newly created    {created:>6,}")
    print(f"merged (total)   {merged:>6,}")
    print(f"claims           {after['claims']:>6,}  (relation pass not run yet)")
    print("\nRe-running is safe: MERGE is keyed on (namespace, source_key) and "
          "IRIs are set ON CREATE only, so existing nodes keep their identity.")


def main() -> None:
    p = argparse.ArgumentParser(description="Write gazetteer entities to the graph.")
    p.add_argument("--collection", default=None)
    p.add_argument("--commit", action="store_true",
                   help="actually write. Without this, nothing is modified.")
    p.add_argument("--dupes", default="kgx/out/duplicate_clusters.tsv")
    a = p.parse_args()

    records, ambiguous, used, total = build(a.collection, Path(a.dupes))
    report(records, ambiguous, used, total)

    if not a.commit:
        print("\nDRY RUN — nothing was written. Re-run with --commit to persist.")
        return
    commit(records)


if __name__ == "__main__":
    main()
