"""kgx/dataplane/parse_tier.py — T2: relations from grammar, not meaning.

WHY THIS TIER, AFTER TWO FAILURES
---------------------------------
Three general-purpose NLP techniques have now been measured on this corpus:

    spaCy NER          knows entity TYPES      failed: 200,541 PROPN, 0 acronyms
    NLI entailment     knows what entities ARE failed: contradicts true claims
                                                       about Entra ID at p=1.00
    gazetteer matching knows STRING STRUCTURE  worked: 320/330 terms observed

The pattern is not about model size. Techniques that depend on world
knowledge fail here because the corpus is full of products released after any
general model's training data. Techniques that depend on structure work.

Dependency parsing is structural. "Keycloak implements OpenID Connect" has
nsubj=Keycloak, dobj=OpenID Connect because of English word order and
morphology — the parser needs no representation of Keycloak. That is exactly
the judgement NLI got backwards, and it is why direction is the thing this
tier is FOR.

WHAT IT ADDS OVER THE REGEX TIER
--------------------------------
The gap-regex tier asserted 61 relations from 6,516 chunks (0.9%) because it
required the inter-entity text to be a bare verb phrase. Real prose is not.

    "OpenTelemetry, the CNCF standard, provides distributed tracing"
      regex: gap is ", the CNCF standard, provides " -> no match
      parse: nsubj=OpenTelemetry, dobj=tracing, appositive skipped structurally

Four things the parser gets that a regex cannot:
    subject/object roles   -> direction, which kills the type violations
    passive voice          -> "tracing is provided by OTel" inverts correctly
    conjunction expansion  -> "Istio and Linkerd run on K8s" yields two
    negation scope         -> the `neg` dependency, not "not" being nearby

    pip install spacy
    python -m spacy download en_core_web_sm
    python -m kgx.dataplane.parse_tier selftest
    python -m kgx.dataplane.parse_tier analyse --limit 2000
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from kgx.dataplane import shapes
from kgx.dataplane.gazetteer_matcher import Gazetteer

OUT_DIR = Path("kgx/out")

# ---------------------------------------------------------------------------
# VERB (+ preposition) -> CANDIDATE PREDICATES, tried in order.
#
# Candidates rather than a single mapping, because the right predicate often
# depends on the OBJECT'S CLASS, which shapes.check already knows: "provides"
# is hasCapability when the object is a Capability and implements when it is
# a Pattern. Encoding that here would duplicate the ontology; letting the
# type gate arbitrate keeps one source of truth.
# ---------------------------------------------------------------------------
VERB_MAP: dict[str, list[str]] = {
    "run":        ["runsOn"],
    "execute":    ["runsOn"],
    "deploy":     ["runsOn"],
    "host":       ["runsOn"],
    "implement":  ["implements"],
    "provide":    ["hasCapability", "implements", "exposes"],
    "offer":      ["hasCapability", "implements"],
    "deliver":    ["hasCapability", "implements"],
    "enable":     ["hasCapability", "implements"],
    "support":    ["implements", "integratesWith", "exposes"],
    "expose":     ["exposes"],
    "speak":      ["exposes"],
    "publish":    ["exposes"],
    "emit":       ["exposes"],
    "depend":     ["dependsOn"],
    "rely":       ["dependsOn"],
    "require":    ["requires", "dependsOn"],
    "need":       ["requires", "dependsOn"],
    "use":        ["dependsOn"],
    "build":      ["dependsOn"],
    "base":       ["dependsOn"],
    "back":       ["dependsOn"],
    "integrate":  ["integratesWith"],
    "connect":    ["integratesWith"],
    "work":       ["integratesWith"],
    "replace":    ["supersedes"],
    "supersede":  ["supersedes"],
    "rename":     ["rebrandedAs"],
    "rebrand":    ["rebrandedAs"],
    "deprecate":  ["deprecatedBy"],
    "mitigate":   ["addressesRisk"],
    "prevent":    ["addressesRisk"],
    "address":    ["addressesRisk"],
    "reduce":     ["addressesRisk"],
    "protect":    ["addressesRisk"],
    "comply":     ["governedBy"],
    "govern":     ["governedBy"],
    "measure":    ["measuredBy"],
    "track":      ["measuredBy"],
    "monitor":    ["measuredBy", "hasCapability"],
    "belong":     ["partOf"],
    "compete":    ["competesWith"],
}

# "X includes Y" asserts Y partOf X — subject and object swap. Kept explicit
# so the inversion is a declared property, never inferred at runtime.
INVERTED: dict[str, list[str]] = {
    "include":  ["partOf"],
    "contain":  ["partOf"],
    "comprise": ["partOf"],
}

SUBJ_DEPS = {"nsubj", "nsubjpass"}
OBJ_DEPS = {"dobj", "attr", "oprd", "pobj", "dative"}
QUALIFIER_LEMMAS = {"if", "when", "unless", "except", "provided", "assuming",
                    "only", "may", "might", "typically", "usually", "sometimes"}


def _load_nlp():
    """Parser + lemmatizer only. NER is DISABLED deliberately: its label set
    was measured as unusable on this corpus, and the gazetteer already owns
    entity recognition. We want this model's grammar, not its opinions."""
    try:
        import spacy
    except ImportError:
        print("[kgx.parse] spacy not installed; tier disabled.")
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        print("[kgx.parse] en_core_web_sm missing. Run: "
              "python -m spacy download en_core_web_sm")
        return None


def _governing_verb(tok):
    """Climb to the verb governing a token, remembering any preposition
    crossed on the way ("depends ON kafka" needs the preposition to
    disambiguate the verb sense)."""
    prep = None
    seen = 0
    cur = tok
    while cur.head is not cur and seen < 6:
        if cur.dep_ == "prep":
            prep = cur.lemma_.lower()
        if cur.head.pos_ in ("VERB", "AUX"):
            return cur.head, prep, cur.dep_
        cur = cur.head
        seen += 1
    return None, prep, None


def _role(tok):
    """Grammatical role of an entity's head token, following conjunctions so
    that "Istio and Linkerd run on Kubernetes" yields both subjects."""
    cur, hops = tok, 0
    while hops < 4:
        if cur.dep_ in SUBJ_DEPS:
            return ("subject", cur.dep_ == "nsubjpass")
        if cur.dep_ in OBJ_DEPS:
            return ("object", False)
        if cur.dep_ == "conj":
            cur = cur.head          # inherit the coordinated sibling's role
            hops += 1
            continue
        return (None, False)
    return (None, False)


def extract(nlp, text: str, mentions: list) -> tuple[list[dict], list[dict]]:
    """Return (asserted, escalate). Escalated items are NOT dropped — they
    still reach the LLM, or the cascade silently caps recall on exactly the
    conditional and negated statements the corpus is most interesting for."""
    if nlp is None or not mentions:
        return [], []
    doc = nlp(text)

    # Align gazetteer character spans to parse tokens. char_span with
    # alignment_mode="expand" survives tokenizer disagreements such as
    # "Pub/Sub" or "GPT-4".
    anchors = []
    for m in mentions:
        if m.ambiguous:
            continue
        span = doc.char_span(m.start, m.end, alignment_mode="expand")
        if span is not None:
            anchors.append((span.root, m))
    if len(anchors) < 2:
        return [], []

    asserted, escalate = [], []
    by_verb: dict[int, dict] = {}

    for root, m in anchors:
        verb, prep, _ = _governing_verb(root)
        if verb is None:
            continue
        role, passive = _role(root)
        if role is None:
            continue
        slot = by_verb.setdefault(verb.i, {"verb": verb, "subject": [], "object": []})
        slot[role].append((m, prep, passive))

    for slot in by_verb.values():
        verb = slot["verb"]
        lemma = verb.lemma_.lower()
        inverted = lemma in INVERTED
        candidates = INVERTED.get(lemma) or VERB_MAP.get(lemma)
        if not candidates:
            continue

        negated = any(c.dep_ == "neg" for c in verb.children)
        sent = verb.sent
        qualified = any(t.lemma_.lower() in QUALIFIER_LEMMAS
                        and t.dep_ in ("mark", "advmod", "aux") for t in sent)

        for subj, _sp, passive in slot["subject"]:
            for obj, oprep, _ in slot["object"]:
                if subj.canonical == obj.canonical:
                    continue
                s, o = (obj, subj) if (passive or inverted) else (subj, obj)

                pred = next((p for p in candidates
                             if shapes.check(p, s.entity_class, o.entity_class).ok), None)
                span = (min(s.start, o.start), max(s.end, o.end))
                base = {"subject": s.canonical, "subject_class": s.entity_class,
                        "object": o.canonical, "object_class": o.entity_class,
                        "span": span, "evidence": text[span[0]:span[1]],
                        "verb": lemma, "prep": oprep}

                if pred is None:
                    escalate.append({**base, "predicate": candidates[0],
                                     "reason": "no_type_valid_predicate"})
                    continue
                if negated or qualified:
                    escalate.append({**base, "predicate": pred,
                                     "reason": "negated" if negated else "qualified"})
                    continue
                from kgx.dataplane.cascade import must_escalate
                stop, why = must_escalate(pred, True, True, False)
                if stop:
                    escalate.append({**base, "predicate": pred, "reason": why})
                    continue
                asserted.append({**base, "predicate": pred, "negated": False,
                                 "route": "parse", "confidence": 0.88})
    return asserted, escalate


# ------------------------------------------------------------------ selftest

CASES = [
    ("Keycloak implements OpenID Connect.", 1, "the case NLI reversed"),
    ("OpenTelemetry, the CNCF standard, provides distributed tracing.", 1,
     "appositive that defeated the regex tier"),
    ("Istio and Linkerd both run on Kubernetes.", 2, "conjunction expansion"),
    ("Distributed tracing is provided by OpenTelemetry.", 1, "passive inverts"),
    ("LangGraph does not run on Kubernetes.", 0, "negation -> escalate"),
    ("Istio runs on Kubernetes only when the mesh is hardened.", 0,
     "qualifier -> escalate"),
    ("Azure AI Foundry replaces Azure OpenAI Service.", 0,
     "high stakes -> never cheap"),
    ("Terraform, Pulumi, Helm.", 0, "no verb, no relation"),
]


def selftest() -> int:
    nlp = _load_nlp()
    if nlp is None:
        return 2
    g = Gazetteer.load()
    fails = 0
    for text, want, note in CASES:
        ms = g.find(text)
        a, e = extract(nlp, text, ms)
        ok = len(a) == want
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] asserted={len(a)} escalated={len(e)}"
              f"   {note}")
        for x in a:
            print(f"           {x['subject']} -{x['predicate']}-> {x['object']}")
        for x in e:
            print(f"           (escalated: {x['subject']} -{x['predicate']}-> "
                  f"{x['object']}, {x['reason']})")
    print(f"\n{len(CASES) - fails}/{len(CASES)} passed")
    return 1 if fails else 0


# ------------------------------------------------------------------- analyse

def analyse(collection: str | None, dupes: Path, limit: int | None) -> None:
    from kgx import config as kgx_config
    from kgx.dataplane.cascade import has_trigger
    from kgx.dataplane.entity_writer import _scopes, _superseded
    from docstore import store

    nlp = _load_nlp()
    if nlp is None:
        sys.exit(1)

    scopes = [s for s in _scopes(collection) if s[2] == "cloud_ai"]
    skip = _superseded(dupes)
    ids = [i for i, _, _ in scopes]
    ph = ",".join("?" * len(ids))
    with store.conn() as c:
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    g = Gazetteer.load()
    weak = set(kgx_config.get_settings().trigger_weak_classes)
    preds, reasons = Counter(), Counter()
    claims: list[dict] = []
    n = covered = 0

    for r in rows:
        if r["doc_id"] in skip:
            continue
        text = r["text"] or ""
        ms = [m for m in g.find(text) if not m.ambiguous]
        if len({m.canonical for m in ms if m.entity_class not in weak}) < 2:
            continue
        n += 1
        if not has_trigger(text):
            continue
        a, e = extract(nlp, text, ms)
        for x in a:
            preds[x["predicate"]] += 1
            claims.append(x)
        for x in e:
            reasons[x["reason"]] += 1
        if a:
            covered += 1
        if limit and n >= limit:
            break

    print(f"\n=== PARSE TIER ===")
    print(f"chunks in scope        {n:,}")
    print(f"chunks with >=1 claim  {covered:,}  ({covered/max(1,n):.0%})")
    print(f"claims asserted        {len(claims):,}")
    print(f"claims per chunk       {len(claims)/max(1,n):.2f}"
          f"   (LLM baseline was 0.48)")

    print("\n--- by predicate ---")
    for p, c in preds.most_common():
        print(f"  {p:<20}{c:>7,}")
    print("\n--- escalated to the LLM ---")
    for r_, c in reasons.most_common():
        print(f"  {r_:<24}{c:>7,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "parse_claims.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("subject\tsubj_class\tpredicate\tobject\tobj_class\tverb\tevidence\n")
        for x in claims:
            ev = x["evidence"].replace("\t", " ").replace("\n", " ")[:160]
            f.write(f"{x['subject']}\t{x['subject_class']}\t{x['predicate']}\t"
                    f"{x['object']}\t{x['object_class']}\t{x['verb']}\t{ev}\n")
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Dependency-parse relation tier.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("analyse")
    a.add_argument("--collection", default=None)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--dupes", default="kgx/out/duplicate_clusters.tsv")
    ns = p.parse_args()
    if ns.cmd == "selftest":
        sys.exit(selftest())
    analyse(ns.collection, Path(ns.dupes), ns.limit)


if __name__ == "__main__":
    main()
