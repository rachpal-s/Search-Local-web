"""kgx/dataplane/nli_verifier.py — verify candidates, don't generate them.

An NLI model scores whether a premise ENTAILS, CONTRADICTS, or is NEUTRAL
toward a hypothesis. Template a hypothesis from any candidate triple and you
get an independent judgement on it:

    premise:    "LangGraph does not run on Kubernetes by default."
    hypothesis: "LangGraph runs on Kubernetes."
    -> contradiction 0.94

WHY THIS IS THE PIECE THAT WAS MISSING
--------------------------------------
Every confidence number in the pipeline so far is either deterministic
(span located: yes/no) or the model grading its own homework. The 200-chunk
pilot showed what the latter is worth: 1.0 on a backwards claim, 0.95 on a
fabricated quote. Self-report was a constant.

NLI is different in three ways that matter:

  1. It VERIFIES rather than generates, so one comparable score applies to
     candidates from every tier — pattern, parse, cheap LLM, frontier LLM.
     The cascade finally has a common currency.

  2. It is trained specifically on entailment, so the score is calibrated in
     a way an LLM's introspective guess is not.

  3. CONTRADICTION is a distinct output class. That is how negation gets
     handled properly, instead of by looking for "not" near a verb.

DIRECTION TESTING
-----------------
The largest error class in the pilot was reversed relations. Scoring the
FLIPPED hypothesis too costs one extra forward pass and catches them: if
"OpenTelemetry implements Distributed Tracing" entails more strongly than
the claim as written, the claim is backwards. Reported, never auto-applied
— a systematic flip is a prompt or pattern defect, not a thousand data edits.

    pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
    python -m kgx.dataplane.nli_verifier selftest
    python -m kgx.dataplane.nli_verifier calibrate kgx/out/claims_preview.tsv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

OUT_DIR = Path("kgx/out")

# xsmall is ~70MB and CPU-fast; base is ~400MB and more accurate. Start with
# xsmall — if it cannot separate the pilot's known-bad claims from the good
# ones, a bigger model probably will not rescue the approach either.
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

ENTAIL_MIN = 0.65        # accept without escalation
CONTRA_MIN = 0.65        # reject, or confirm an asserted negation
FLIP_MARGIN = 0.20       # reversed must beat forward by this to call it backwards

# ---------------------------------------------------------------------------
# Hypotheses are written as PLAIN ENGLISH, not ontology jargon. The NLI model
# was trained on natural sentences; "X implements Y" scores well, "X
# hasCapability Y" does not parse as English and scores as noise.
# ---------------------------------------------------------------------------
HYPOTHESIS: dict[str, str] = {
    "offeredBy":          "{s} is a product offered by {o}.",
    "runsOn":             "{s} runs on {o}.",
    "implements":         "{s} implements {o}.",
    "integratesWith":     "{s} integrates with {o}.",
    "dependsOn":          "{s} depends on {o}.",
    "alternativeTo":      "{s} is an alternative to {o}.",
    "competesWith":       "{s} competes with {o}.",
    "partOf":             "{s} is part of {o}.",
    "hasCapability":      "{s} provides {o}.",
    "addressesRisk":      "{s} mitigates the risk of {o}.",
    "mitigatedBy":        "{s} is mitigated by {o}.",
    "governedBy":         "{s} is governed by {o}.",
    "requires":           "{s} requires {o}.",
    "exposes":            "{s} exposes {o}.",
    "supersedes":         "{s} replaces {o}.",
    "rebrandedAs":        "{s} was renamed to {o}.",
    "deprecatedBy":       "{s} is deprecated by {o}.",
    "benchmarkedAgainst": "{s} is benchmarked against {o}.",
    "measuredBy":         "{s} is measured by {o}.",
    "appliesTo":          "{s} applies to {o}.",
}


def _symmetric() -> set[str]:
    """Read symmetry from the ontology, not a second hardcoded list.

    Flip-testing a symmetric predicate is meaningless: "RDS is an alternative
    to Cloud SQL" and its reverse assert the same thing, so a higher reverse
    score is an artefact of word order, not evidence of a backwards claim.
    Two of the seven direction flags in the first calibration were this bug.
    """
    try:
        d = yaml.safe_load(
            Path("kgx/semantic/modules/cloud_ai.yaml").read_text(encoding="utf-8"))
        return {k for k, v in (d.get("predicate_constraints") or {}).items()
                if (v or {}).get("symmetric")}
    except Exception:
        return {"alternativeTo", "competesWith", "integratesWith"}


SYMMETRIC = _symmetric()


def hypothesis(predicate: str, subject: str, obj: str) -> str | None:
    t = HYPOTHESIS.get(predicate)
    return t.format(s=subject, o=obj) if t else None


@dataclass
class Verdict:
    entail: float = 0.0
    neutral: float = 0.0
    contradict: float = 0.0
    flipped_entail: float = 0.0
    decision: str = "unavailable"   # verified | contradicted | escalate | unavailable
    direction_suspect: bool = False
    note: str = ""

    @property
    def component(self) -> float:
        """Contribution to the composite confidence score."""
        return {"verified": 1.0, "escalate": 0.4,
                "contradicted": 0.0, "unavailable": 0.5}[self.decision]


class NLIVerifier:
    """Lazy-loading, fail-soft. If transformers is absent the verifier returns
    `unavailable` for everything and the cascade continues — the same
    contract graph_repo uses for a missing Neo4j."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16,
                 max_length: int = 256):
        self.model_name, self.batch_size, self.max_length = model_name, batch_size, max_length
        self._tok = self._model = None
        self._idx: dict[str, int] = {}
        self._unavailable: str | None = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._unavailable:
            return False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            self._unavailable = f"{e}"
            print(f"[kgx.nli] transformers/torch not installed ({e}); "
                  f"verification disabled.")
            return False
        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._torch = torch
            # Label order differs between checkpoints. Read it from the model
            # rather than assuming; a silently transposed mapping would turn
            # every entailment into a contradiction and look like bad data.
            self._idx = {v.lower(): k for k, v in self._model.config.id2label.items()}
            missing = {"entailment", "neutral", "contradiction"} - set(self._idx)
            if missing:
                raise ValueError(f"model labels {self._model.config.id2label} "
                                 f"missing {missing}")
        except Exception as e:                                   # noqa: BLE001
            self._unavailable = f"{type(e).__name__}: {e}"
            print(f"[kgx.nli] could not load {self.model_name}: {e}")
            return False
        return True

    def _scores(self, pairs: list[tuple[str, str]]) -> list[dict]:
        if not self._load() or not pairs:
            return [{} for _ in pairs]
        out: list[dict] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            enc = self._tok([p for p, _ in batch], [h for _, h in batch],
                            truncation=True, padding=True,
                            max_length=self.max_length, return_tensors="pt")
            with self._torch.no_grad():
                probs = self._torch.softmax(self._model(**enc).logits, dim=-1)
            for row in probs:
                out.append({k: float(row[i_]) for k, i_ in self._idx.items()})
        return out

    def verify(self, premise: str, subject: str, predicate: str, obj: str,
               negated: bool = False) -> Verdict:
        return self.verify_batch([(premise, subject, predicate, obj, negated)])[0]

    def verify_batch(self, items: list[tuple]) -> list[Verdict]:
        pairs, index = [], []
        verdicts = [Verdict() for _ in items]

        for n, (premise, s, pred, o, _neg) in enumerate(items):
            h = hypothesis(pred, s, o)
            if not h:
                verdicts[n].note = "no hypothesis template"
                continue
            flipped = None if pred in SYMMETRIC else hypothesis(pred, o, s)
            index.append((n, len(pairs), flipped is not None))
            pairs.append((premise, h))
            pairs.append((premise, flipped or h))

        scored = self._scores(pairs)
        if not scored or not scored[0]:
            for v in verdicts:
                v.decision = "unavailable"
            return verdicts

        for n, base, flip_tested in index:
            fwd, rev = scored[base], (scored[base + 1] if flip_tested else {})
            v = verdicts[n]
            v.entail = round(fwd.get("entailment", 0.0), 3)
            v.neutral = round(fwd.get("neutral", 0.0), 3)
            v.contradict = round(fwd.get("contradiction", 0.0), 3)
            v.flipped_entail = round(rev.get("entailment", 0.0), 3)
            _decide(v, negated=items[n][4], flip_tested=flip_tested)
        return verdicts


def _decide(v: Verdict, negated: bool, flip_tested: bool = True) -> None:
    """Separated from the model so the policy is testable without a download."""
    # A claim asserted as negated SHOULD contradict its positive hypothesis.
    # Contradiction is confirmation here, not failure.
    if negated:
        v.decision = "verified" if v.contradict >= CONTRA_MIN else "escalate"
        v.note = "negation confirmed" if v.decision == "verified" else "negation unconfirmed"
        return

    if (flip_tested and v.flipped_entail - v.entail >= FLIP_MARGIN
            and v.flipped_entail >= ENTAIL_MIN):
        v.direction_suspect = True
        v.decision = "escalate"
        v.note = f"reversed reads stronger ({v.flipped_entail} vs {v.entail})"
        return

    if v.contradict >= CONTRA_MIN:
        v.decision = "contradicted"
        v.note = "source denies this relation"
    elif v.entail >= ENTAIL_MIN:
        v.decision = "verified"
    else:
        v.decision = "escalate"
        v.note = "neither entailed nor contradicted"


# ----------------------------------------------------------------- calibrate

def _chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    try:
        from docstore import store
        with store.conn() as c:
            ph = ",".join("?" * len(chunk_ids))
            return {r["chunk_id"]: r["text"] for r in c.execute(
                f"SELECT chunk_id, text FROM doc_chunks WHERE chunk_id IN ({ph})",
                list(chunk_ids))}
    except Exception as e:                                       # noqa: BLE001
        print(f"  (could not load full chunks: {e})")
        return {}


def calibrate(path: Path, model_name: str, limit: int | None,
              full_chunk: bool = False) -> None:
    """Score claims already on disk. Free: the LLM pass is paid for, and the
    only question is whether NLI separates its good output from its bad."""
    if not path.exists():
        sys.exit(f"no claims file at {path}")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    hdr = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) == len(hdr):
            rows.append(dict(zip(hdr, p)))
    if limit:
        rows = rows[:limit]
    print(f"claims to score: {len(rows):,}   model: {model_name}")

    premises = {}
    if full_chunk:
        premises = _chunk_texts({r["chunk_id"] for r in rows if r.get("chunk_id")})
        print(f"using full chunk text as premise for {len(premises):,} claims")

    v = NLIVerifier(model_name, max_length=512 if full_chunk else 256)
    items = [(premises.get(r.get("chunk_id", ""), r.get("evidence", "")),
              r["subject"], r["predicate"], r["object"],
              str(r.get("negated", "False")).lower() == "true") for r in rows]
    verdicts = v.verify_batch(items)
    if verdicts and verdicts[0].decision == "unavailable":
        print("NLI unavailable — install transformers and torch, then re-run.")
        return

    from collections import Counter
    dec = Counter(x.decision for x in verdicts)
    print("\n=== DECISIONS ===")
    for k, n in dec.most_common():
        print(f"  {k:<16}{n:>6,}  ({n/max(1,len(verdicts)):.0%})")

    # THE KEY QUESTION: does NLI agree with the deterministic type gate?
    # If claims the ontology already rejected score LOWER on entailment, the
    # signal is real and can be trusted on claims the gate cannot judge.
    good = [x for x, r in zip(verdicts, rows) if r.get("type_ok", "True") == "True"]
    bad = [x for x, r in zip(verdicts, rows) if r.get("type_ok", "True") == "False"]
    if good and bad:
        gm = sum(x.entail for x in good) / len(good)
        bm = sum(x.entail for x in bad) / len(bad)
        print(f"\n=== SEPARATION (vs the type gate) ===")
        print(f"  mean entailment, type-valid    {gm:.3f}  (n={len(good)})")
        print(f"  mean entailment, type-invalid  {bm:.3f}  (n={len(bad)})")
        print(f"  gap                            {gm - bm:+.3f}")
        print("  A clear positive gap means NLI independently agrees with the")
        print("  ontology, and can be trusted where the ontology is silent.")

    flips = [(r, x) for r, x in zip(rows, verdicts) if x.direction_suspect]
    contra = [(r, x) for r, x in zip(rows, verdicts) if x.decision == "contradicted"]
    print(f"\ndirection-suspect: {len(flips):,}   contradicted: {len(contra):,}")
    for r, x in flips[:8]:
        print(f"  BACKWARDS?  {r['subject'][:22]:<24}{r['predicate']:<16}"
              f"{r['object'][:22]:<24}fwd={x.entail} rev={x.flipped_entail}")
    for r, x in contra[:8]:
        print(f"  DENIED      {r['subject'][:22]:<24}{r['predicate']:<16}"
              f"{r['object'][:22]:<24}contra={x.contradict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "nli_scores.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("subject\tpredicate\tobject\ttype_ok\tdecision\tentail\t"
                "contradict\tflipped_entail\tdirection_suspect\tnote\n")
        for r, x in zip(rows, verdicts):
            f.write(f"{r['subject']}\t{r['predicate']}\t{r['object']}\t"
                    f"{r.get('type_ok','')}\t{x.decision}\t{x.entail}\t"
                    f"{x.contradict}\t{x.flipped_entail}\t{x.direction_suspect}\t{x.note}\n")
    print(f"\nwrote {out}")


# ------------------------------------------------------------------ selftest

def selftest() -> int:
    """Exercises templating and the decision policy with SYNTHETIC scores, so
    it runs without downloading a model. The policy is where the bugs hide;
    the model is a black box either way."""
    print("--- hypothesis templating ---")
    for pred, s, o in [("implements", "OpenTelemetry", "Distributed Tracing"),
                       ("runsOn", "Istio", "Kubernetes"),
                       ("addressesRisk", "Guardrails", "Prompt Injection"),
                       ("nonsense", "A", "B")]:
        print(f"  {pred:<16}-> {hypothesis(pred, s, o)}")

    print("\n--- decision policy ---")
    cases = [
        ("clean entailment",      dict(entail=.92, contradict=.02, flipped_entail=.30), False, "verified"),
        ("source denies it",      dict(entail=.05, contradict=.91, flipped_entail=.06), False, "contradicted"),
        ("negation confirmed",    dict(entail=.04, contradict=.88, flipped_entail=.05), True,  "verified"),
        ("negation unconfirmed",  dict(entail=.55, contradict=.20, flipped_entail=.40), True,  "escalate"),
        ("backwards relation",    dict(entail=.41, contradict=.05, flipped_entail=.89), False, "escalate"),
        ("weak, neither way",     dict(entail=.44, contradict=.18, flipped_entail=.39), False, "escalate"),
        ("both high, fwd wins",   dict(entail=.88, contradict=.03, flipped_entail=.75), False, "verified"),
    ]
    fails = 0
    for note, sc, neg, want in cases:
        v = Verdict(**sc)
        _decide(v, negated=neg)
        ok = v.decision == want
        fails += not ok
        flag = "  <-- direction_suspect" if v.direction_suspect else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {note:<22}-> {v.decision:<14}"
              f"component={v.component}{flag}")
        if not ok:
            print(f"         expected {want}")
    print(f"\n{len(cases) - fails}/{len(cases)} passed")
    return 1 if fails else 0


def main() -> None:
    p = argparse.ArgumentParser(description="NLI verification for extracted claims.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("calibrate")
    c.add_argument("claims", nargs="?", default="kgx/out/claims_preview.tsv")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--full-chunk", action="store_true",
                   help="use the whole chunk as premise instead of the "
                        "truncated evidence span (a fairer test)")
    ns = p.parse_args()
    if ns.cmd == "selftest":
        sys.exit(selftest())
    calibrate(Path(ns.claims), ns.model, ns.limit, ns.full_chunk)


if __name__ == "__main__":
    main()
