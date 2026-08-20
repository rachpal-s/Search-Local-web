"""jobs/graph_pipeline.py — v0 knowledge graph construction for one collection.

Reuses entities already sitting in doc_chunks from the existing I4 enrich
step — no re-extraction, no new LLM cost. Three stages:

    1. LOAD    read every chunk's entities/keywords for the collection from
               SQLite (fast; this is metadata already computed, not text
               processing).
    2. RESOLVE run docstore/entity_resolver.py's clustering — the CPU-bound
               part, parallelized across cores by splitting the collection's
               documents into shards and resolving each shard independently,
               then merging shard-level clusters in a fast final pass.
    3. WRITE   upsert resolved entities and co-occurrence edges into Neo4j.

Why sharded parallel resolution rather than one big single-process pass: the
resolver's blocking (by surname) already keeps any single comparison pass
close to linear, but at real scale — tens of thousands of mentions — even a
linear pass benefits from splitting across cores. Splitting by DOCUMENT
(never splitting a single document's chunks across shards) preserves the
same-document evidence signal the resolver depends on; only the FINAL
cross-shard merge needs to run single-process, and it is cheap because it
only compares already-deduplicated per-shard entities, not raw mentions.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from docstore.entity_resolver import Mention, ResolvedEntity, resolve_entities


# ------------------------------------------------------------------ worker side

def _resolve_shard(args: dict) -> list[dict]:
    """Runs in a pool process. Input/output are plain dicts — picklable,
    no shared state, no database connection crossing the process boundary."""
    threshold = args["threshold"]
    mentions = [Mention(**m) for m in args["mentions"]]
    resolved = resolve_entities(mentions, threshold=threshold)
    return [_serialize(r) for r in resolved]


def _serialize(r: ResolvedEntity) -> dict:
    return {
        "canonical_name": r.canonical_name, "label": r.label,
        "aliases": r.aliases, "doc_ids": sorted(r.doc_ids),
        "mention_count": r.mention_count,
        # Chunk-level co-occurrence needed for edge-building in the merge
        # step: which OTHER entity texts appeared alongside each mention.
        "chunk_cooccurrences": [
            {"chunk_id": m.chunk_id, "doc_id": m.doc_id,
             "cooccurring_entities": sorted(m.cooccurring_entities)}
            for m in r.mentions
        ],
    }


# ------------------------------------------------------------------ parent side

def load_mentions(collection_id: str) -> list[Mention]:
    """Pull every chunk's entities for this collection and build Mention
    records, using each chunk's OTHER entities/keywords as the "surrounding
    context" the resolver needs."""
    import json
    from docstore import store

    with store.conn() as c:
        rows = c.execute(
            "SELECT k.chunk_id, k.doc_id, k.entities, k.keywords, d.source_uri "
            "FROM doc_chunks k JOIN documents d ON d.id = k.doc_id "
            "WHERE k.conversation_id=?", (collection_id,)).fetchall()

    mentions: list[Mention] = []
    for row in rows:
        entities = json.loads(row["entities"] or "[]")
        keywords = frozenset(json.loads(row["keywords"] or "[]"))
        if not entities:
            continue
        cooccurring_here = frozenset(
            e["text"].strip().lower() for e in entities if e.get("text"))
        for e in entities:
            text = (e.get("text") or "").strip()
            label = e.get("label") or ""
            if not text or not label:
                continue
            # A mention's co-occurring set excludes itself.
            others = cooccurring_here - {text.strip().lower()}
            mentions.append(Mention(
                raw_text=text, label=label, doc_id=row["doc_id"],
                chunk_id=row["chunk_id"], source_uri=row["source_uri"] or "",
                cooccurring_entities=others, cooccurring_keywords=keywords,
            ))
    return mentions


def _shard_by_document(mentions: list[Mention], n_shards: int) -> list[list[Mention]]:
    """Split into shards, grouping by SOURCE FOLDER first, never splitting a
    single document across shards.

    This matters more than it looks: the folder-based evidence signal in
    context_score() is the primary way "Rachpal Singh" and "R Singh" merge
    when they are the same person — but that signal only fires between
    mentions IN THE SAME SHARD (cross-shard merging, below, is deliberately
    more conservative). Splitting round-robin by document size would
    routinely scatter one person's folder full of documents across several
    shards for no reason related to the data, silently weakening the exact
    signal this feature exists to use. Grouping by folder keeps a person's
    documents together whenever the corpus is organised that way — which,
    for exactly the CV-folder scenario this was built for, it is.
    """
    by_folder: dict[str, list[str]] = {}     # folder -> doc_ids, first-seen order
    by_doc: dict[str, list[Mention]] = {}
    for m in mentions:
        by_doc.setdefault(m.doc_id, []).append(m)
        folder = _folder_key(m.source_uri)
        docs = by_folder.setdefault(folder, [])
        if m.doc_id not in docs:
            docs.append(m.doc_id)

    # Bin-pack folders (by total mention count) across shards, largest first,
    # so shard sizes stay reasonably even without splitting a folder.
    folder_sizes = sorted(
        ((f, sum(len(by_doc[d]) for d in docs)) for f, docs in by_folder.items()),
        key=lambda kv: -kv[1])

    n = max(1, n_shards)
    shard_load = [0] * n
    shard_docs: list[list[str]] = [[] for _ in range(n)]
    for folder, size in folder_sizes:
        target = min(range(n), key=lambda i: shard_load[i])
        shard_load[target] += size
        shard_docs[target].extend(by_folder[folder])

    shards = [[m for d in docs for m in by_doc[d]] for docs in shard_docs]
    return [s for s in shards if s]


def _folder_key(source_uri: str) -> str:
    norm = (source_uri or "").replace("\\", "/")
    return norm.rsplit("/", 1)[0] if "/" in norm else norm


def _merge_shard_results(shard_results: list[list[dict]],
                         threshold: float) -> list[dict]:
    """Cross-shard merge: two shards might each have independently resolved
    an entity that should really be one (e.g. "Rachpal Singh" landed in one
    shard's documents, "R Singh" in another's, both from the same folder).
    Re-run resolution treating each shard's ALREADY-RESOLVED entity as a
    single synthetic mention carrying the union of its evidence — cheap,
    because this compares resolved entities (few), not raw mentions (many).
    """
    synthetic: list[Mention] = []
    entity_by_synthetic_id: dict[int, dict] = {}
    for entities in shard_results:
        for e in entities:
            idx = len(synthetic)
            all_cooccur: set[str] = set()
            for cc in e["chunk_cooccurrences"]:
                all_cooccur.update(cc["cooccurring_entities"])
            synthetic.append(Mention(
                raw_text=e["canonical_name"], label=e["label"],
                doc_id=e["doc_ids"][0] if e["doc_ids"] else f"synthetic_{idx}",
                chunk_id=f"synthetic_{idx}",
                source_uri="",   # cross-shard: folder signal already spent locally
                cooccurring_entities=frozenset(all_cooccur),
            ))
            entity_by_synthetic_id[idx] = e

    if not synthetic:
        return []

    merged_clusters = resolve_entities(synthetic, threshold=threshold)

    out: list[dict] = []
    for cluster in merged_clusters:
        member_indices = [synthetic.index(m) for m in cluster.mentions]
        originals = [entity_by_synthetic_id[i] for i in member_indices]
        all_aliases = sorted({originals[0]["canonical_name"]}
                             | {a for o in originals for a in o["aliases"]}
                             | {o["canonical_name"] for o in originals[1:]})
        canonical = cluster.canonical_name
        all_aliases = [a for a in all_aliases if a != canonical]
        doc_ids = sorted({d for o in originals for d in o["doc_ids"]})
        mention_count = sum(o["mention_count"] for o in originals)
        chunk_cooccurrences = [cc for o in originals for cc in o["chunk_cooccurrences"]]
        out.append({
            "canonical_name": canonical, "label": cluster.label,
            "aliases": all_aliases, "doc_ids": doc_ids,
            "mention_count": mention_count,
            "chunk_cooccurrences": chunk_cooccurrences,
        })
    return out


def _build_cooccurrence_edges(entities: list[dict]) -> list[dict]:
    """One edge per pair of canonical entities that appeared in the same
    chunk at least once, weighted by how many chunks co-mention them."""
    # Map: normalized entity text (any alias or canonical) -> canonical entity dict
    text_to_entity: dict[str, dict] = {}
    for e in entities:
        for name in [e["canonical_name"], *e["aliases"]]:
            text_to_entity[name.strip().lower()] = e

    edge_weight: dict[tuple[str, str, str, str], int] = {}
    edge_docs: dict[tuple[str, str, str, str], set[str]] = {}

    for e in entities:
        for cc in e["chunk_cooccurrences"]:
            for other_text in cc["cooccurring_entities"]:
                other = text_to_entity.get(other_text)
                if not other or other is e:
                    continue
                a, b = sorted([e["canonical_name"], other["canonical_name"]])
                la, lb = (e["label"], other["label"]) if a == e["canonical_name"] \
                    else (other["label"], e["label"])
                key = (a, b, la, lb)
                edge_weight[key] = edge_weight.get(key, 0) + 1
                edge_docs.setdefault(key, set()).add(cc["doc_id"])

    return [
        {"a": a, "b": b, "label_a": la, "label_b": lb,
         "weight": w, "doc_ids": sorted(edge_docs[(a, b, la, lb)])}
        for (a, b, la, lb), w in edge_weight.items()
    ]


def auto_workers(requested: int = 0) -> int:
    import os
    if requested and requested > 0:
        return requested
    cores = os.cpu_count() or 2
    return max(1, cores - max(1, int(cores * 0.25)))


@dataclass
class GraphBuildResult:
    entities: list[dict]
    edges: list[dict]
    mention_count: int
    elapsed_ms: int


def build_graph(collection_id: str, threshold: float,
                workers: int = 0) -> GraphBuildResult:
    """Synchronous, blocking — the runner calls this via asyncio.to_thread
    since ProcessPoolExecutor.submit needs a running loop context but the
    work itself is CPU-bound and short relative to a full ingestion job."""
    t0 = time.perf_counter()
    mentions = load_mentions(collection_id)
    if not mentions:
        return GraphBuildResult([], [], 0, int((time.perf_counter() - t0) * 1000))

    n_shards = auto_workers(workers)
    shards = _shard_by_document(mentions, n_shards)

    if len(shards) <= 1:
        shard_results = [[_serialize(r) for r in resolve_entities(shards[0], threshold)]] \
            if shards else []
    else:
        with ProcessPoolExecutor(max_workers=len(shards)) as pool:
            shard_results = list(pool.map(
                _resolve_shard,
                [{"mentions": [vars(m) for m in shard], "threshold": threshold}
                 for shard in shards],
            ))

    entities = _merge_shard_results(shard_results, threshold) if len(shards) > 1 \
        else shard_results[0]
    edges = _build_cooccurrence_edges(entities)

    return GraphBuildResult(
        entities=entities, edges=edges, mention_count=len(mentions),
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
