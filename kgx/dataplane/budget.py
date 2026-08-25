"""kgx/dataplane/budget.py — cost the relation pass BEFORE running it.

The naive trigger ("this chunk has >=2 gazetteer mentions, send it to the
LLM") selects 51% of the corpus. That number is not a fact about the domain;
it is a consequence of which classes we chose to put in the gazetteer. Vendor
mentions alone account for 30% of all matches, and a chunk whose only
entities are "AWS" and "Azure" contains no relation worth extracting — it
just costs four seconds of GPU to discover that.

So: enumerate candidate trigger rules, count what each selects, and pick with
numbers instead of intuition. Every rule below is cheap to evaluate because
the matcher is deterministic and free.

    python -m kgx.dataplane.budget
    python -m kgx.dataplane.budget --seconds-per-call 3.5 --concurrency 4
    python -m kgx.dataplane.budget --exclude-superseded kgx/out/duplicate_clusters.tsv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from kgx.dataplane.gazetteer_matcher import Gazetteer

# Classes that are too weak, alone, to justify an LLM call. A Vendor mention
# is usually a name-drop; a Capability is often a section heading. Neither
# implies a relation is present. They still get EXTRACTED — they just do not
# vote for spending money on the chunk.
WEAK_CLASSES = {"Vendor", "_ambiguous"}
WEAKER_CLASSES = WEAK_CLASSES | {"Capability"}


def _rules():
    """(name, predicate over a set of (canonical, class) pairs, description).

    Keyed on the ENTITY, not the class. Counting distinct classes would score
    "Observability + Evaluation" as one signal because both are Capability,
    which is exactly backwards: those are two entities and a relation between
    them is plausible.
    """
    def strong(pairs, weak):
        return sum(1 for _, cls in pairs if cls not in weak)

    RELATIONAL = {"Service", "Technology", "Protocol", "Pattern", "Model"}

    return [
        ("R1  >=2 mentions (current)",
         lambda pairs, total: total >= 2,
         "counts repeats; a chunk saying 'AWS' twice qualifies"),
        ("R2  >=2 distinct entities",
         lambda pairs, total: len(pairs) >= 2,
         "de-duplicates within the chunk"),
        ("R3  >=2 distinct, Vendor excluded",
         lambda pairs, total: strong(pairs, WEAK_CLASSES) >= 2,
         "RECOMMENDED - Vendor name-drops stop voting"),
        ("R4  >=3 distinct, Vendor excluded",
         lambda pairs, total: strong(pairs, WEAK_CLASSES) >= 3,
         "aggressive; expect recall loss on sparse chunks"),
        ("R5  >=2 distinct, Vendor+Capability excluded",
         lambda pairs, total: strong(pairs, WEAKER_CLASSES) >= 2,
         "tightest defensible rule"),
        ("R6  >=2 distinct AND >=1 relational class",
         lambda pairs, total: len(pairs) >= 2 and any(c in RELATIONAL for _, c in pairs),
         "requires something a predicate can actually attach to"),
    ]


def _superseded_ids(path: Path) -> set[str]:
    """doc_ids retired by the duplicate scan. Extracting from a superseded
    document costs money to produce claims that a newer document already
    supersedes — and inflates every downstream weight while doing it."""
    if not path.exists():
        print(f"  (no duplicate report at {path}; superseded docs not excluded)")
        return set()
    out = set()
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    hdr = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(hdr):
            continue
        row = dict(zip(hdr, parts))
        if row.get("role") in ("SUPERSEDED", "ABSORBED"):
            out.add(row.get("doc_id", ""))
    return out - {""}


def main() -> None:
    p = argparse.ArgumentParser(description="Cost the LLM relation pass.")
    p.add_argument("--collection", default=None)
    p.add_argument("--seconds-per-call", type=float, default=4.0)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--exclude-superseded", default="kgx/out/duplicate_clusters.tsv")
    a = p.parse_args()

    from docstore import store

    with store.conn() as c:
        if a.collection:
            row = c.execute("SELECT id,title FROM conversations WHERE id=? OR title=?",
                            (a.collection, a.collection)).fetchone()
            if not row:
                sys.exit(f"no collection {a.collection!r}")
            scopes = [(row["id"], row["title"])]
        else:
            scopes = [(r["id"], r["title"]) for r in c.execute(
                "SELECT id,title FROM conversations WHERE kind='collection' "
                "ORDER BY title").fetchall()]
        ids = [i for i, _ in scopes]
        ph = ",".join("?" * len(ids))
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    print("scanning: " + ", ".join(t for _, t in scopes))
    skip = _superseded_ids(Path(a.exclude_superseded))
    print(f"superseded/absorbed documents excluded: {len(skip)}")

    g = Gazetteer.load()
    rules = _rules()
    selected = Counter()
    eligible = 0
    zero_hit_docs: set[str] = set()
    hit_docs: set[str] = set()

    for r in rows:
        if r["doc_id"] in skip:
            continue
        eligible += 1
        mentions = g.find(r["text"] or "", r["chunk_id"], r["doc_id"])
        pairs = {(m.canonical, m.entity_class) for m in mentions}
        (hit_docs if mentions else zero_hit_docs).add(r["doc_id"])
        for name, pred, _ in rules:
            if pred(pairs, len(mentions)):
                selected[name] += 1

    print(f"\n=== RELATION-PASS BUDGET ===")
    print(f"chunks in scope        {len(rows):>8,}")
    print(f"after excluding superseded {eligible:>4,}")
    print(f"\n{'rule':<46}{'chunks':>9}{'% corpus':>10}{'hours':>8}")
    print("-" * 73)
    for name, _, note in rules:
        n = selected[name]
        hours = (n * a.seconds_per_call) / max(1, a.concurrency) / 3600
        print(f"{name:<46}{n:>9,}{n/max(1,eligible):>9.0%}{hours:>8.1f}")
        print(f"{'':<46}{note}")

    print(f"\nassumptions: {a.seconds_per_call}s per call, concurrency {a.concurrency}")
    print(f"documents with zero mentions: {len(zero_hit_docs - hit_docs):,}")
    print("\nNOTHING WAS MODIFIED. No LLM was called.")


if __name__ == "__main__":
    main()
