"""kgx/dataplane/gazetteer_matcher.py — deterministic extraction, route 1.

Framework §7: route, don't default. This module handles the ~75-85% of this
domain that is ENUMERABLE — vendors, services, models, protocols, patterns.
No LLM, no inference, perfect auditability, and every mention carries a
verified character span.

WHY THIS RUNS BEFORE ANY LLM
----------------------------
Two reasons, both about money and trust:

  1. It costs nothing. You can run it over 26,018 chunks in a couple of
     minutes and see entity recall per document BEFORE committing three
     hours of GPU to the relation pass. A bad gazetteer entry found here is
     free; found mid-backfill it is not.

  2. It supplies the entity list that the LLM relation pass is CONSTRAINED
     to (§7 non-negotiable #4, two-pass extraction). Pass 2 receives a
     resolved entity list and extracts only relations among those entities.
     That is what stops the model inventing subjects.

SPAN GROUNDING IS NOT OPTIONAL
------------------------------
Every Mention carries (start, end) into the chunk text, and verify() asserts
text[start:end] normalizes to the matched surface form. A mention that fails
verification is discarded, not repaired. Downstream, Claim.evidence_span_*
inherits these offsets.

    python -m kgx.dataplane.gazetteer_matcher selftest      # no DB needed
    python -m kgx.dataplane.gazetteer_matcher report
    python -m kgx.dataplane.gazetteer_matcher report --collection "Library (BE)"
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

GAZETTEER_PATH = Path("kgx/semantic/gazetteer/cloud_ai.yaml")
OUT_DIR = Path("kgx/out")

# Separators treated as interchangeable INSIDE a term. Zero-or-more, so
# "Retrieval-Augmented Generation", "Retrieval Augmented Generation" and
# "RetrievalAugmentedGeneration" all resolve to one entity. The v0.1 flat
# table used a strict boundary and missed 37 documents of a single term.
_SEP = r"[-_\s/]*"

# Guards. A term must not be preceded or followed by a word character, so
# "RAG" does not fire inside "RAGE" and "REST" does not fire inside "RESTORE".
_PRE = r"(?<![\w-])"
_POST = r"(?![\w-])"

# Surface forms shorter than this are dropped: "EA", "DR", "KG" collide with
# ordinary prose (kilogram, Doctor, and so on) far more often than they hit.
MIN_FORM_CHARS = 3


@dataclass
class Mention:
    """One verified occurrence of a gazetteer term in one chunk."""
    canonical: str
    entity_class: str
    owner: str
    surface: str
    start: int
    end: int
    chunk_id: str
    doc_id: str
    ambiguous: bool = False
    candidates: list[str] = field(default_factory=list)
    route: str = "gazetteer"

    def verify(self, text: str) -> bool:
        """Programmatic span check (§7 rule 3). The single highest-value
        guard in the extraction pipeline: it makes a fabricated offset
        impossible rather than unlikely."""
        if not (0 <= self.start < self.end <= len(text)):
            return False
        got = re.sub(r"[-_\s/]+", "", text[self.start:self.end]).lower()
        want = re.sub(r"[-_\s/]+", "", self.surface).lower()
        return got == want


@dataclass
class Gazetteer:
    forms: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    hub_exclusions: set[str] = field(default_factory=set)
    _ci: re.Pattern | None = None
    _cs: re.Pattern | None = None
    dropped: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = GAZETTEER_PATH) -> "Gazetteer":
        import yaml
        if not path.exists():
            sys.exit(f"gazetteer not found at {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        g = cls()
        g.hub_exclusions = {h.lower() for h in data.get("hub_exclusions", [])}
        g.ambiguous = {k: list(v or []) for k, v in (data.get("ambiguous") or {}).items()}

        for entity_class, owners in (data.get("terms") or {}).items():
            for owner, items in (owners or {}).items():
                owner = "" if owner == "_" else owner
                for canonical, aliases in (items or {}).items():
                    for surface in [canonical, *(aliases or [])]:
                        if len(surface) < MIN_FORM_CHARS:
                            g.dropped.append(surface)
                            continue
                        if surface.lower() in g.hub_exclusions:
                            g.dropped.append(surface)
                            continue
                        # First writer wins: gazetteer order is the priority
                        # order, so an earlier class keeps an ambiguous form.
                        g.forms.setdefault(surface, (canonical, entity_class, owner))

        # Ambiguous bare forms must be MATCHABLE, or the signal disappears.
        # "Vault" spans 102 documents meaning both HashiCorp Vault and Azure
        # Key Vault. Leaving it unregistered silently drops all 102 — which
        # is worse than guessing, because a guess is at least visible. These
        # are emitted as unresolved mentions carrying their candidate list,
        # for type/context disambiguation at resolution time (§8: never let
        # Apple the company merge with Apple the fruit).
        for bare in g.ambiguous:
            if len(bare) >= MIN_FORM_CHARS and bare.lower() not in g.hub_exclusions:
                g.forms.setdefault(bare, (bare, "_ambiguous", ""))
        g._compile()
        return g

    # -- pattern construction ------------------------------------------------

    @staticmethod
    def _to_pattern(surface: str) -> str:
        """Tokenise on separators and rejoin with the flexible class."""
        tokens = [t for t in re.split(r"[-_\s/]+", surface) if t]
        return _SEP.join(re.escape(t) for t in tokens)

    def _is_acronym(self, surface: str) -> bool:
        """All-caps forms must match case-sensitively or they collide with
        prose: REST/rest, ACID/acid, IAM/I am, DORA/Dora."""
        core = re.sub(r"[-_\s/0-9.]", "", surface)
        return bool(core) and core.isupper() and len(core) <= 6

    def _compile(self) -> None:
        # Longest first so "Azure OpenAI Service" wins over "Azure" even
        # before overlap resolution runs.
        ci, cs = [], []
        for surface in sorted(self.forms, key=len, reverse=True):
            (cs if self._is_acronym(surface) else ci).append(self._to_pattern(surface))
        self._ci = re.compile(_PRE + "(" + "|".join(ci) + ")" + _POST,
                              re.I) if ci else None
        self._cs = re.compile(_PRE + "(" + "|".join(cs) + ")" + _POST) if cs else None

    # -- matching ------------------------------------------------------------

    def _resolve(self, surface: str) -> tuple[str, str, str] | None:
        """Map a matched string back to its entry, separator-insensitively."""
        if surface in self.forms:
            return self.forms[surface]
        key = re.sub(r"[-_\s/]+", "", surface).lower()
        for form, meta in self.forms.items():
            if re.sub(r"[-_\s/]+", "", form).lower() == key:
                return meta
        return None

    def find(self, text: str, chunk_id: str = "", doc_id: str = "") -> list[Mention]:
        """All non-overlapping verified mentions, longest-match-wins."""
        raw: list[tuple[int, int, str]] = []
        for pat in (self._cs, self._ci):
            if pat:
                raw.extend((m.start(), m.end(), m.group(1)) for m in pat.finditer(text))

        # Overlap resolution: longest span wins; ties broken by earlier start.
        raw.sort(key=lambda t: (t[0], -(t[1] - t[0])))
        chosen: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, surface in sorted(raw, key=lambda t: (-(t[1] - t[0]), t[0])):
            if any(not (end <= s or start >= e) for s, e, _ in chosen):
                continue
            chosen.append((start, end, surface))
        chosen.sort()

        out: list[Mention] = []
        for start, end, surface in chosen:
            meta = self._resolve(surface)
            if not meta:
                continue
            canonical, entity_class, owner = meta
            is_amb = entity_class == "_ambiguous" or canonical in self.ambiguous
            m = Mention(canonical=canonical, entity_class=entity_class, owner=owner,
                        surface=surface, start=start, end=end,
                        chunk_id=chunk_id, doc_id=doc_id, ambiguous=is_amb,
                        candidates=self.ambiguous.get(canonical, []))
            if m.verify(text):          # discard, never repair
                out.append(m)
        return out


# --------------------------------------------------------------------- report

def report(collection: str | None, sample: int) -> None:
    from docstore import store

    with store.conn() as c:
        if collection:
            row = c.execute("SELECT id,title FROM conversations WHERE id=? OR title=?",
                            (collection, collection)).fetchone()
            if not row:
                sys.exit(f"no collection {collection!r}")
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
    g = Gazetteer.load()
    print(f"gazetteer: {len(g.forms):,} surface forms "
          f"({len(g.dropped)} dropped as too short or hub-excluded)")

    per_class: Counter = Counter()
    per_term_docs: dict[str, set[str]] = defaultdict(set)
    per_doc: Counter = Counter()
    hist: Counter = Counter()
    ambiguous_hits = 0
    total = 0
    samples: list[Mention] = []

    for r in rows:
        text = r["text"] or ""
        mentions = g.find(text, r["chunk_id"], r["doc_id"])
        hist[min(len(mentions), 6)] += 1
        for m in mentions:
            per_class[m.entity_class] += 1
            per_term_docs[m.canonical].add(m.doc_id)
            per_doc[m.doc_id] += 1
            ambiguous_hits += m.ambiguous
            total += 1
            if len(samples) < sample:
                samples.append(m)

    n_chunks = len(rows)
    n_docs = len({r["doc_id"] for r in rows})
    ge1 = sum(v for k, v in hist.items() if k >= 1)
    ge2 = sum(v for k, v in hist.items() if k >= 2)

    print(f"\n=== GAZETTEER MATCH REPORT ===")
    print(f"documents               {n_docs:>8,}")
    print(f"chunks                  {n_chunks:>8,}")
    print(f"verified mentions       {total:>8,}")
    print(f"chunks with >=1 mention {ge1:>8,} ({ge1/max(1,n_chunks):.0%})")
    print(f"chunks with >=2 mentions{ge2:>8,} ({ge2/max(1,n_chunks):.0%})"
          f"   <-- LLM relation-pass workload")
    print(f"distinct entities seen  {len(per_term_docs):>8,}")
    print(f"ambiguous mentions      {ambiguous_hits:>8,} "
          f"({ambiguous_hits/max(1,total):.1%})  <-- held for disambiguation")
    print(f"documents with 0 hits   {n_docs - len(per_doc):>8,}"
          f"   <-- likely the personal collection, expected")

    print("\n--- mentions by class ---")
    for cls, n in per_class.most_common():
        print(f"  {cls:<14}{n:>8,}")

    print("\n--- top 25 entities by document spread ---")
    for term, docs in sorted(per_term_docs.items(), key=lambda kv: -len(kv[1]))[:25]:
        print(f"  {term[:44]:<46}{len(docs):>5} docs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mention_sample.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("canonical\tclass\tsurface\tstart\tend\tambiguous\tchunk_id\n")
        for m in samples:
            f.write(f"{m.canonical}\t{m.entity_class}\t{m.surface}\t"
                    f"{m.start}\t{m.end}\t{m.ambiguous}\t{m.chunk_id}\n")
    print(f"\nwrote {out} ({len(samples)} verified spans for eyeballing)")
    print("NOTHING WAS MODIFIED.")


# ------------------------------------------------------------------- selftest

CASES = [
    ("We used Retrieval-Augmented Generation here.",       ["Retrieval Augmented Generation"]),
    ("RAG pipelines need reranking.",                      ["Retrieval Augmented Generation", "Reranking"]),
    ("Azure OpenAI Service is now Azure AI Foundry.",       ["Azure OpenAI Service", "Azure AI Foundry"]),
    ("Deployed on Amazon EKS behind AWS PrivateLink.",      ["Amazon EKS", "AWS PrivateLink"]),
    ("human-in-the-loop review, i.e. HITL.",                ["Human in the Loop", "Human in the Loop"]),
    ("PCI-DSS and PCI DSS are the same thing.",             ["PCI DSS", "PCI DSS"]),
    ("mTLS, or Mutual TLS, between services.",             ["mTLS", "mTLS"]),
    ("The rest of the document is boilerplate.",            []),   # 'rest' != REST
    ("She had a raging headache.",                          []),   # 'raging' != RAG
    ("Acidic soil needs lime.",                             []),   # 'Acidic' != ACID
    ("Store secrets in Vault.",                             ["Vault"]),            # unresolved
    ("Store secrets in HashiCorp Vault.",                   ["HashiCorp Vault"]),  # longest wins
    ("An SLO of 99.9% and an RTO of 4 hours.",             ["Service Level Objective", "Recovery Time Objective"]),
]


def selftest() -> int:
    g = Gazetteer.load()
    print(f"loaded {len(g.forms):,} surface forms\n")
    failures = 0
    for text, expected in CASES:
        got = g.find(text)
        names = sorted({m.canonical for m in got})
        want = sorted(set(expected))
        ok = names == want
        span_ok = all(m.verify(text) for m in got)
        flag = "PASS" if ok and span_ok else "FAIL"
        failures += flag == "FAIL"
        print(f"  [{flag}] {text[:46]:<48} -> {names}")
        if not ok:
            print(f"         expected {want}")
        for m in got:
            if m.ambiguous:
                print(f"         (ambiguous: {m.surface!r} held for disambiguation)")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic gazetteer extraction.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="run built-in cases; no database needed")
    r = sub.add_parser("report", help="match the whole corpus, report recall")
    r.add_argument("--collection", default=None)
    r.add_argument("--sample", type=int, default=300)
    a = p.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    report(a.collection, a.sample)


if __name__ == "__main__":
    main()
