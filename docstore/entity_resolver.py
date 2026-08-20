"""docstore/entity_resolver.py — v0 entity resolution: no LLM, still smart.

The problem this exists to solve, stated precisely: spaCy NER gives you raw
mentions — "Rachpal Singh" in one chunk, "R Singh" in another, "R. Singh" in a
third. A knowledge graph is only useful if those collapse into ONE node when
they really are the same person, and stay SEPARATE nodes when they are not —
and a real corpus absolutely contains both cases. Two people named "R Singh"
in two unrelated documents are not the same person just because the strings
are compatible; two mentions of "Rachpal Singh" and "R Singh" inside the same
personal CV folder almost certainly are.

The approach: merging requires TWO independent things to agree, not one.

  1. Name compatibility — a cheap, syntactic check. "R Singh" is a plausible
     abbreviation of "Rachpal Singh" (shared surname, compatible given name).
     "Priya Singh" is NOT, even though it shares a surname too. This step
     only asks "could these be the same name", never "are they the same
     person" — that answer comes from evidence, not spelling.

  2. Contextual evidence — do the surrounding chunks actually corroborate it?
     Same document, same source folder, overlapping co-mentioned entities or
     keywords. A name-compatible pair with NO corroborating context stays
     separate; the same pair inside the same document or the same personal
     folder accumulates enough evidence to merge.

Nothing here uses an LLM. It is meant to be upgraded later (v1: LLM-assisted
resolution for ambiguous clusters this heuristic leaves unresolved) without
changing the shape of its output — a mapping from raw mention to canonical
entity, which is exactly what the graph-build job consumes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ data model

@dataclass
class Mention:
    """One entity mention, with just enough surrounding context to judge it."""
    raw_text: str
    label: str                  # spaCy label: PERSON, ORG, GPE, ...
    doc_id: str
    chunk_id: str
    source_uri: str = ""
    # Other entity texts (normalized) and keywords present in the SAME chunk —
    # this is the "surrounding chunk" evidence the resolver leans on.
    cooccurring_entities: frozenset[str] = field(default_factory=frozenset)
    cooccurring_keywords: frozenset[str] = field(default_factory=frozenset)


@dataclass
class ResolvedEntity:
    canonical_name: str
    label: str
    aliases: list[str]
    mentions: list[Mention]
    doc_ids: set[str]

    @property
    def mention_count(self) -> int:
        return len(self.mentions)


# ------------------------------------------------------------------ name compatibility

_TITLE_WORDS = {"mr", "mrs", "ms", "dr", "prof", "shri", "smt"}
_STRIP = re.compile(r"[.,]")


def normalize_name(text: str) -> list[str]:
    """Lowercase token list, punctuation and honorifics stripped."""
    cleaned = _STRIP.sub("", text.strip().lower())
    return [t for t in cleaned.split() if t and t not in _TITLE_WORDS]


def _token_compatible(a: str, b: str) -> bool:
    """Two name tokens are compatible if equal, or one is an initial of the
    other ("r" vs "rachpal"). A bare initial never counts as compatible with
    ANOTHER bare initial of a different letter — "r" and "s" are not the same
    person just because both are single letters."""
    if a == b:
        return True
    if len(a) == 1 and b.startswith(a):
        return True
    if len(b) == 1 and a.startswith(b):
        return True
    return False


def names_compatible(a: str, b: str) -> tuple[bool, float]:
    """(compatible?, base_name_score).

    Compatible means: same surname (last token, exact), and every remaining
    token in the SHORTER name has a compatible counterpart in the longer one,
    in order. Missing middle names are fine ("Rachpal Singh" vs "Rachpal
    Kumar Singh"); a genuinely different surname is an immediate reject
    regardless of anything else — no amount of contextual evidence should
    merge "Rachpal Singh" with "Rachpal Kapoor".

    base_name_score: 1.0 for an exact normalized match (strong evidence on
    its own), 0.35 for a compatible-but-abbreviated match (weak alone — many
    people share "R Singh" — needs contextual corroboration to merge).
    """
    ta, tb = normalize_name(a), normalize_name(b)
    if not ta or not tb:
        return False, 0.0
    if ta == tb:
        return True, 1.0
    if ta[-1] != tb[-1]:          # surname mismatch: hard reject
        return False, 0.0

    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # Every token in the shorter name must find a compatible, order-preserving
    # match in the longer name (surname already confirmed equal above).
    li = 0
    for st in short[:-1]:                 # all but the surname, already matched
        found = False
        while li < len(long_) - 1:
            if _token_compatible(st, long_[li]):
                found = True
                li += 1
                break
            li += 1
        if not found:
            return False, 0.0
    return True, 0.35


# ------------------------------------------------------------------ contextual evidence

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _folder_of(source_uri: str) -> str:
    """Parent directory, forward-slash-normalized, for a coarse 'same personal
    folder' signal without needing exact path equality."""
    norm = (source_uri or "").replace("\\", "/")
    return norm.rsplit("/", 1)[0] if "/" in norm else norm


def context_score(m1: Mention, m2: Mention) -> float:
    """0..1 evidence that two mentions refer to the same real-world entity,
    from everything EXCEPT the name string itself."""
    score = 0.0

    if m1.doc_id == m2.doc_id:
        # Two mentions in the same document, already name-compatible, are
        # about as strong a signal as this heuristic can offer without an
        # LLM: a single document very rarely refers to two different people
        # by compatible-but-different forms of the same name.
        score += 0.40
    elif _folder_of(m1.source_uri) and _folder_of(m1.source_uri) == _folder_of(m2.source_uri):
        # Different documents, same source folder — exactly the "2.0
        # Rachpal.Singh/Accounts/CV" scenario: many files about one person,
        # not one file that happens to share a folder with unrelated others.
        score += 0.30

    score += 0.30 * _jaccard(m1.cooccurring_entities, m2.cooccurring_entities)
    score += 0.15 * _jaccard(m1.cooccurring_keywords, m2.cooccurring_keywords)
    return min(score, 1.0)


# ------------------------------------------------------------------ clustering

class _UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {i: i for i in items}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _blocking_key(name: str) -> str:
    """Coarse grouping key so comparison is O(n) within a surname bucket, not
    O(n^2) across an entire collection's mentions."""
    tokens = normalize_name(name)
    return tokens[-1] if tokens else ""


def resolve_entities(mentions: list[Mention],
                     threshold: float = 0.62) -> list[ResolvedEntity]:
    """Cluster mentions into canonical entities.

    Only mentions sharing a label (PERSON with PERSON, never PERSON with ORG)
    and a blocking key (same final name token) are ever compared — this
    keeps the whole pass fast even at tens of thousands of mentions, since
    the vast majority of pairs are never candidates for merging in the first
    place.
    """
    if not mentions:
        return []

    by_bucket: dict[tuple[str, str], list[int]] = {}
    for i, m in enumerate(mentions):
        key = (m.label, _blocking_key(m.raw_text))
        if key[1]:
            by_bucket.setdefault(key, []).append(i)

    uf = _UnionFind(list(range(len(mentions))))

    for idxs in by_bucket.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                m1, m2 = mentions[i], mentions[j]
                compatible, name_score = names_compatible(m1.raw_text, m2.raw_text)
                if not compatible:
                    continue
                if name_score >= 1.0:
                    # Exact normalized match. Still same real-world entity
                    # with overwhelming likelihood; merge without requiring
                    # further corroboration.
                    uf.union(i, j)
                    continue
                total = name_score + context_score(m1, m2)
                if total >= threshold:
                    uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(mentions)):
        clusters.setdefault(uf.find(i), []).append(i)

    out: list[ResolvedEntity] = []
    for member_idxs in clusters.values():
        members = [mentions[i] for i in member_idxs]
        # Canonical name: the longest (most complete) surface form seen,
        # ties broken by which appeared first — "Rachpal Singh" over
        # "R Singh" when both are present.
        seen_order = list(dict.fromkeys(m.raw_text for m in members))
        canonical = max(seen_order, key=lambda t: (len(t), -seen_order.index(t)))
        aliases = [t for t in seen_order if t != canonical]
        out.append(ResolvedEntity(
            canonical_name=canonical, label=members[0].label,
            aliases=aliases, mentions=members,
            doc_ids={m.doc_id for m in members},
        ))
    return out
