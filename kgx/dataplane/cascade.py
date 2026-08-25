"""kgx/dataplane/cascade.py — route, don't default (framework §7).

The pipeline currently sends every triggered chunk to an LLM. That is 100%
LLM for relations, against a target mix of roughly 60% deterministic / 25%
classical / 15% LLM. This module builds the cheaper tiers and, more
importantly, MEASURES them against the 95 claims the LLM already produced,
so the decision to keep or drop each tier rests on a number.

    T0  ontology       already known; the LLM should never be asked
    T1  trigger filter no relational verb -> not a relation-bearing chunk
    T2  patterns       lexico-syntactic over TYPED spans
    T3  LLM            only the residue

    python -m kgx.dataplane.cascade selftest
    python -m kgx.dataplane.cascade analyse
    python -m kgx.dataplane.cascade replay kgx/out/claims_preview.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from kgx.dataplane import shapes
from kgx.dataplane.gazetteer_matcher import Gazetteer

OUT_DIR = Path("kgx/out")

# ---------------------------------------------------------------------------
# T0 — PREDICATES THE SEMANTIC PLANE ALREADY ANSWERS
#
# The gazetteer records `Amazon Bedrock -> owner: AWS`, and entity_writer
# already persists it as a property on the node. So offeredBy is reference
# data, not a claim the corpus makes, and asking a model to rediscover it
# spends money to learn what is already stored. In the 200-chunk pilot this
# was 9 of 95 claims.
#
# The saving is not just those calls: dropping the predicate also drops
# Vendor entities from most prompts, which shortens every remaining call.
# ---------------------------------------------------------------------------
ONTOLOGY_ANSWERED = {"offeredBy"}


def prompt_predicates(all_predicates: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in all_predicates.items() if k not in ONTOLOGY_ANSWERED}


# ---------------------------------------------------------------------------
# T1 — TRIGGER LEXICON (negative filter)
#
# A chunk holding entity names but no relational verb is a heading, a bullet
# list of product names, a table row or a contents page. Your corpus is full
# of them, and they are what produced "Evaluation appliesTo Chunking" — the
# model was handed two nouns and no proposition, so it manufactured one.
#
# Used ONLY to exclude. A chunk that passes still goes to T2/T3; nothing is
# asserted on the strength of a verb being present.
# ---------------------------------------------------------------------------
TRIGGERS = re.compile(r"""\b(
    runs?\ on|running\ on|deployed?\ (on|to|in)|hosted\ (on|by|in)|executes?\ on|
    implements?|implementing|provides?|delivers?|offers?|enables?|
    supports?|exposes?|speaks?|serves?|publishes?|emits?|
    depends?\ on|requires?|needs?|relies\ on|built\ on|based\ on|backed\ by|
    integrates?\ with|works\ with|connects?\ to|talks?\ to|
    replaces?|supersedes?|deprecat(es|ed)|renamed|rebranded|migrat(es|ed)\ (to|from)|
    alternatives?\ to|instead\ of|rather\ than|versus|vs\.?|compared\ (to|with)|
    mitigat(es|ed)|prevents?|protects?\ against|addresses|reduces?|guards?\ against|
    complies?\ with|governed\ by|subject\ to|mandated\ by|regulated\ by|
    measured\ by|tracked\ by|monitored\ by|observed\ by|
    part\ of|component\ of|consists?\ of|comprises?|includes?|contains?|
    is\ a|are\ a|acts?\ as|serves?\ as|used\ (for|as|by|to)
)\b""", re.I | re.X)


def has_trigger(text: str) -> bool:
    return bool(TRIGGERS.search(text or ""))


# ---------------------------------------------------------------------------
# T2 — LEXICO-SYNTACTIC PATTERNS over the gap between two TYPED spans.
#
# Matching runs on the text BETWEEN two verified entity mentions, not on raw
# prose, so the endpoints are grounded before a pattern is even tried. Class
# validity is not encoded here — shapes.check() enforces it, which keeps one
# source of truth for domain/range.
# ---------------------------------------------------------------------------
# A gap like " does not run on " must still MATCH, or a negated statement
# produces silence instead of an escalation — the worst possible outcome,
# because it looks like the cascade found nothing to say.
_AUX = r"(?:does|do|did|is|are|was|were|can|could|may|might|will|would|should|must)\s+"
_NEG = r"(?:not|never|no\s+longer)\s+"
_PREFIX = rf"^\W*(?:{_AUX})?(?:{_NEG})?"

_CORES: list[tuple[str, str]] = [
    ("runsOn",         r"(runs?|running|executes?|deployed?|hosted)\b.{0,20}?\b(on|to|in|by)"),
    ("implements",     r"(implements?|implementing|provides?|delivers?|enables?|supports?)"),
    ("exposes",        r"(exposes?|speaks?|serves?\ over|publishes?|emits?)"),
    ("dependsOn",      r"(depends?\ on|relies\ on|requires?|needs?|built\ on|based\ on|backed\ by)"),
    ("integratesWith", r"(integrates?\ with|works\ with|connects?\ to|talks?\ to)"),
    ("alternativeTo",  r"(vs\.?|versus|instead\ of|rather\ than|as\ an\ alternative\ to|compared\ (to|with))"),
    ("supersedes",     r"(replaces?|supersedes?|has\ replaced)"),
    ("rebrandedAs",    r"(renamed\ (to|as)|rebranded\ (to|as)|now\ (called|known\ as))"),
    ("deprecatedBy",   r"(deprecated\ (by|in\ favour\ of|in\ favor\ of))"),
    ("addressesRisk",  r"(mitigat(es|e)|prevents?|protects?\ against|addresses|guards?\ against|reduces?)"),
    ("governedBy",     r"(complies?\ with|governed\ by|subject\ to|mandated\ by|regulated\ by|comply\ with)"),
    ("measuredBy",     r"(measured\ by|tracked\ by|monitored\ (by|via)|expressed\ as)"),
    ("partOf",         r"(part\ of|component\ of|module\ of|belongs\ to)"),
    ("hasCapability",  r"(provides?|offers?|has|includes?)"),
]
PATTERNS: list[tuple[str, re.Pattern]] = [
    (pred, re.compile(_PREFIX + core + r"\W*$", re.I | re.X)) for pred, core in _CORES
]

# Escalate rather than assert. A pattern matches adjacent spans; it cannot
# scope a conditional. "X runs on Y only when Z" is not "X runs on Y", and a
# regex has no way to know the difference.
QUALIFIER = re.compile(r"\b(if|when|unless|only|except|provided|assuming|"
                       r"in\ some\ cases|typically|usually|may|might|can\ be)\b", re.I)
NEGATION = re.compile(r"\b(not|never|cannot|can't|won't|doesn't|does\ not|"
                      r"no\ longer|isn't|is\ not|without|rather\ than|instead\ of)\b", re.I)
SENT_END = re.compile(r"[.!?]\s")

MAX_GAP = 60          # chars between two mentions for a pattern to apply


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence containing a span. Qualifiers usually sit AFTER the
    object — "Istio runs on Kubernetes only when the mesh is hardened" — so
    inspecting the inter-entity gap alone would miss every one of them."""
    left = max((text.rfind(c, 0, start) for c in ".!?\n"), default=-1)
    right = min((r for r in (text.find(c, end) for c in ".!?\n") if r != -1),
                default=len(text))
    return text[left + 1:right]


def pattern_relations(text: str, mentions: list) -> tuple[list[dict], list[dict]]:
    """Return (asserted, escalate). `escalate` are pairs a pattern touched but
    declined to assert — they must still reach the LLM, or the cascade
    silently caps recall on exactly the hard cases."""
    ms = sorted(mentions, key=lambda m: m.start)
    asserted, escalate = [], []

    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            if b.start - a.end > MAX_GAP:
                break
            if a.canonical == b.canonical or a.ambiguous or b.ambiguous:
                continue
            gap = text[a.end:b.start]
            if SENT_END.search(gap):          # different sentences
                continue

            hit = next((p for p, rx in PATTERNS if rx.search(gap)), None)
            if not hit:
                continue

            # A qualified or negated statement carries scope a pattern cannot
            # represent. Hand it up rather than flatten it.
            sent = _sentence(text, a.start, b.end)
            reason = ("negated" if NEGATION.search(gap) else
                      "qualified" if QUALIFIER.search(sent) else None)
            if reason:
                escalate.append({"subject": a.canonical, "object": b.canonical,
                                 "predicate": hit, "reason": reason,
                                 "span": (a.start, b.end)})
                continue

            v = shapes.check(hit, a.entity_class, b.entity_class)
            if not v.ok:
                # Never auto-invert. If the inverse would be valid, the
                # PATTERN is wrong about direction — fix the pattern.
                escalate.append({"subject": a.canonical, "object": b.canonical,
                                 "predicate": hit,
                                 "reason": "type_violation", "span": (a.start, b.end)})
                continue

            asserted.append({
                "subject": a.canonical, "subject_class": a.entity_class,
                "predicate": hit,
                "object": b.canonical, "object_class": b.entity_class,
                "span": (a.start, b.end), "evidence": text[a.start:b.end],
                "negated": False, "route": "pattern", "confidence": 0.85,
            })
    return asserted, escalate


# ---------------------------------------------------------------------------
# ESCALATION — what may NOT be settled by a cheap tier or a cheap model.
# ---------------------------------------------------------------------------
HIGH_STAKES = {"supersedes", "rebrandedAs", "deprecatedBy", "contradicts"}


def must_escalate(predicate: str, type_ok: bool, tiers_agree: bool,
                  routed_other: bool) -> tuple[bool, str]:
    """Escalation is decided by INDEPENDENT signals, never by a model's
    self-reported confidence. The pilot showed self-report pinned at 1.0
    including on a backwards claim and a fabricated quote."""
    if predicate in HIGH_STAKES:
        return True, "high_stakes_predicate"      # the temporal backbone
    if not type_ok:
        return True, "type_violation"
    if routed_other:
        return True, "unknown_predicate"
    if not tiers_agree:
        return True, "tier_disagreement"
    return False, ""


# ------------------------------------------------------------------- analyse

def analyse(collection: str | None, dupes: Path, limit: int | None) -> None:
    from kgx import config as kgx_config
    from kgx.dataplane.entity_writer import _scopes, _superseded
    from docstore import store

    scopes = [s for s in _scopes(collection) if s[2] == "cloud_ai"]
    skip = _superseded(dupes)
    ids = [i for i, _, _ in scopes]
    ph = ",".join("?" * len(ids))
    with store.conn() as c:
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    g = Gazetteer.load()
    weak = set(kgx_config.get_settings().trigger_weak_classes)
    funnel = Counter()
    pattern_hits, escalations = Counter(), Counter()
    n = 0

    for r in rows:
        if r["doc_id"] in skip:
            continue
        text = r["text"] or ""
        ms = [m for m in g.find(text) if not m.ambiguous]
        strong = {m.canonical for m in ms if m.entity_class not in weak}
        if len(strong) < 2:
            continue
        n += 1
        funnel["R3 triggered (today's LLM workload)"] += 1
        if not has_trigger(text):
            funnel["T1 dropped: no relational verb"] += 1
            continue
        funnel["T1 passed to patterns"] += 1
        asserted, esc = pattern_relations(text, ms)
        for a in asserted:
            pattern_hits[a["predicate"]] += 1
        for e in esc:
            escalations[e["reason"]] += 1
        if asserted:
            funnel["T2 asserted >=1 relation"] += 1
        if esc or not asserted:
            funnel["T3 still needs the LLM"] += 1
        if limit and n >= limit:
            break

    base = funnel["R3 triggered (today's LLM workload)"] or 1
    print(f"\n=== CASCADE FUNNEL ===")
    for k in ["R3 triggered (today's LLM workload)", "T1 dropped: no relational verb",
              "T1 passed to patterns", "T2 asserted >=1 relation",
              "T3 still needs the LLM"]:
        print(f"  {k:<38}{funnel[k]:>7,}  ({funnel[k]/base:.0%})")

    saved = funnel["T1 dropped: no relational verb"]
    print(f"\nLLM calls avoided by T1 alone: {saved:,} ({saved/base:.0%})")
    print(f"remaining LLM workload:        {funnel['T3 still needs the LLM']:,}")

    print("\n--- T2 pattern assertions by predicate ---")
    for p, c in pattern_hits.most_common():
        print(f"  {p:<20}{c:>7,}")
    print(f"  {'TOTAL':<20}{sum(pattern_hits.values()):>7,}")

    print("\n--- escalated by patterns (NOT asserted) ---")
    for r_, c in escalations.most_common():
        print(f"  {r_:<20}{c:>7,}")


# -------------------------------------------------------------------- replay

def replay(path: Path, collection: str | None, dupes: Path) -> None:
    """Score the cheap tiers against claims the LLM already produced.

    Free evaluation: the LLM pass is paid for, its output is on disk, and the
    only question is how much of it the cheap tiers reproduce.
    """
    if not path.exists():
        sys.exit(f"no claims file at {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    hdr = lines[0].split("\t")
    llm = set()
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) == len(hdr):
            row = dict(zip(hdr, p))
            if row.get("type_ok", "True") == "False":
                continue                      # quarantined; not ground truth
            llm.add((row["subject"], row["predicate"], row["object"]))
    print(f"LLM claims (type-valid): {len(llm):,}")

    from kgx.dataplane.entity_writer import _scopes, _superseded
    from docstore import store
    scopes = [s for s in _scopes(collection) if s[2] == "cloud_ai"]
    skip = _superseded(dupes)
    ids = [i for i, _, _ in scopes]
    ph = ",".join("?" * len(ids))
    with store.conn() as c:
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    g = Gazetteer.load()
    pat = set()
    no_verb_chunks = 0
    for r in rows:
        if r["doc_id"] in skip:
            continue
        text = r["text"] or ""
        if not has_trigger(text):
            no_verb_chunks += 1
            continue
        ms = [m for m in g.find(text) if not m.ambiguous]
        asserted, _ = pattern_relations(text, ms)
        for a in asserted:
            pat.add((a["subject"], a["predicate"], a["object"]))

    both = llm & pat
    only_llm = llm - pat
    only_pat = pat - llm
    print(f"pattern claims:          {len(pat):,}")
    print(f"\n  reproduced by patterns {len(both):>6,}  "
          f"({len(both)/max(1,len(llm)):.0%} of LLM output)")
    print(f"  LLM only               {len(only_llm):>6,}  <-- what the LLM is FOR")
    print(f"  patterns only          {len(only_pat):>6,}  <-- recall the LLM missed")

    if only_pat:
        print("\n  sample of pattern-only claims:")
        for s, p, o in sorted(only_pat)[:10]:
            print(f"    {s[:26]:<28}{p:<16}{o[:26]}")
    if only_llm:
        print("\n  sample of LLM-only claims (the residue that justifies T3):")
        for s, p, o in sorted(only_llm)[:10]:
            print(f"    {s[:26]:<28}{p:<16}{o[:26]}")


# ------------------------------------------------------------------ selftest

def selftest() -> int:
    g = Gazetteer.load()
    cases = [
        ("OpenTelemetry provides distributed tracing.",              1, 0, "clean pattern"),
        ("Terraform, Pulumi, Helm, Argo CD.",                        0, 0, "list of nouns, no verb"),
        ("Observability. Evaluation. Chunking.",                     0, 0, "headings"),
        ("Istio runs on Kubernetes.",                                1, 0, "runsOn"),
        ("LangGraph does not run on Kubernetes.",                    0, 1, "negated -> escalate"),
        ("Istio runs on Kubernetes only when the mesh is hardened.", 0, 1, "qualified -> escalate"),
        ("Azure AI Foundry replaces Azure OpenAI Service.",          0, 0, "high stakes: never cheap"),
    ]
    fails = 0
    for text, want_a, want_e, note in cases:
        verb = has_trigger(text)
        ms = [m for m in g.find(text) if not m.ambiguous]
        a, e = pattern_relations(text, ms) if verb else ([], [])
        # High-stakes predicates are never settled by a cheap tier, however
        # confidently the pattern matched.
        kept = [x for x in a if not must_escalate(x["predicate"], True, True, False)[0]]
        e = e + [x for x in a if x not in kept]
        a = kept
        ok = (len(a) == want_a) and (len(e) >= want_e)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] verb={str(verb):<5} "
              f"asserted={len(a)} escalated={len(e)}  {note}")
    print(f"\n{len(cases) - fails}/{len(cases)} passed")
    return 1 if fails else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Cheap extraction tiers before the LLM.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("analyse")
    a.add_argument("--collection", default=None)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--dupes", default="kgx/out/duplicate_clusters.tsv")
    r = sub.add_parser("replay")
    r.add_argument("claims", nargs="?", default="kgx/out/claims_preview.tsv")
    r.add_argument("--collection", default=None)
    r.add_argument("--dupes", default="kgx/out/duplicate_clusters.tsv")
    ns = p.parse_args()
    if ns.cmd == "selftest":
        sys.exit(selftest())
    if ns.cmd == "analyse":
        analyse(ns.collection, Path(ns.dupes), ns.limit)
    else:
        replay(Path(ns.claims), ns.collection, Path(ns.dupes))


if __name__ == "__main__":
    main()
