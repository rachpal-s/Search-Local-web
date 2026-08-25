"""kgx/dataplane/relation_extractor.py — pass 2, relations only.

Pass 1 (entity_writer) already resolved entities deterministically. This pass
does ONE thing: given a chunk and the entities the gazetteer verified inside
it, decide which of a CLOSED set of predicates hold between them.

THE FOUR CONSTRAINTS THAT MAKE THIS TRUSTWORTHY
-----------------------------------------------
1. Subjects and objects are chosen from a supplied list by INDEX. The model
   cannot name an entity that is not already in the graph, so it cannot
   invent one. Free-text endpoints are the main source of graph junk (§7).

2. Predicates come from the module's enum, loaded from the same YAML the
   ontology uses. Extractor and validator cannot disagree because they read
   one file. Anything else must be emitted as OTHER + proposed_label, which
   goes to the suggestion queue and NOT to the graph (anti-pattern #23).

3. Every relation must quote supporting text. That quote is located in the
   chunk and converted to character offsets. If it cannot be located, the
   relation is REJECTED, not repaired — a claim whose evidence cannot be
   found is a hallucination regardless of how plausible it reads.

4. Negation is explicit. "X MUST NOT run on Y" extracted without the flag
   asserts the opposite of the source.

COST WARNING
------------
The repo's configured inference model is a *-cloud model. This is a batch of
thousands of calls, not one interactive turn. Run --limit 200 first, inspect
kgx/out/claims_preview.tsv, and only then commit the full pass. Point
--model at a local model if you would rather spend GPU than credit.

    python -m kgx.dataplane.relation_extractor --limit 50           # dry run
    python -m kgx.dataplane.relation_extractor --limit 200 --commit
    python -m kgx.dataplane.relation_extractor --commit             # full pass
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from kgx import config as kgx_config
from kgx.dataplane.gazetteer_matcher import Gazetteer
from kgx.dataplane import shapes
from kgx.repositories import graph_repo as repo

OUT_DIR = Path("kgx/out")
MODULE_YAML = Path("kgx/semantic/modules/cloud_ai.yaml")
CHECKPOINT = OUT_DIR / "extract_checkpoint.jsonl"
PROMPT_VERSION = "v1"
MAX_ENTITIES_PER_CHUNK = 12       # keeps the prompt bounded and the task legible


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_predicates() -> dict[str, str]:
    """Single source of truth: the same enum the ontology declares."""
    data = yaml.safe_load(MODULE_YAML.read_text(encoding="utf-8"))
    vals = data["enums"]["CloudAiPredicateEnum"]["permissible_values"]
    return {k: (v or {}).get("description", "") for k, v in vals.items()}


# ------------------------------------------------------------------- prompt

SYSTEM = (
    "You extract relations between named technical entities. "
    "You never invent entities. You never invent predicates. "
    "You quote the exact supporting text. You output only JSON."
)


def build_prompt(text: str, entities: list[tuple[str, str]],
                 predicates: dict[str, str]) -> str:
    # Showing the CLASS is not decoration. "[0] Amazon Bedrock (Service)"
    # tells the model which predicates are structurally available, and the
    # pilot's errors were overwhelmingly class errors: a Capability cannot
    # implement a Technology.
    ent_block = "\n".join(f"  [{i}] {n} ({c})" for i, (n, c) in enumerate(entities))
    pred_block = "\n".join(f"  {k}: {v}" for k, v in predicates.items() if k != "OTHER")
    return f"""Extract relations that the PASSAGE explicitly states between the ENTITIES.

ENTITIES (refer to these by index only):
{ent_block}

ALLOWED PREDICATES:
{pred_block}

RULES
- subject and object MUST be indices from the ENTITIES list above. Never a name, never a new entity.
- Respect the CLASS shown in brackets. A Capability cannot implement anything; a Technology implements a Pattern or Capability, not the reverse. Get the DIRECTION right: "OpenTelemetry implements Distributed Tracing", never the other way round.
- predicate MUST be one from ALLOWED PREDICATES. If the passage states a relation that fits none of them, use "OTHER" and put your suggested name in "proposed_label".
- "evidence" MUST be a VERBATIM span copied from the PASSAGE that states the relation. Do not paraphrase. Do not summarise. If you cannot copy a supporting span, do not emit the relation.
- Set "negated": true when the passage DENIES the relation ("does not support", "must not run on", "is not a substitute for").
- Extract only what the passage STATES. Do not use your own knowledge about these technologies.
- If the passage merely mentions entities near each other without stating a relation, return an empty list. This is common and correct.

PASSAGE:
\"\"\"{text}\"\"\"

Return ONLY this JSON:
{{"relations": [{{"subject": 0, "predicate": "runsOn", "object": 1, "evidence": "verbatim span", "negated": false, "confidence": 0.0, "proposed_label": ""}}]}}"""


def _extract_json(raw: str) -> dict | None:
    """Parse JSON that a model may have wrapped in prose or code fences.

    Not laxity for its own sake: format="json" is honoured inconsistently
    across models and endpoints, and a run that dies on a fence teaches you
    nothing about extraction quality.
    """
    txt = (raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    start, end = txt.find("{"), txt.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except Exception:
            return None
    return None


# ------------------------------------------------------------ span location

_WS = re.compile(r"\s+")


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quoted span in the chunk and return character offsets.

    Three attempts, loosening only on whitespace and case — never on wording.
    A model that paraphrased instead of quoting will fail all three, which is
    the intended outcome: that relation gets rejected.
    """
    quote = (quote or "").strip().strip('"').strip()
    if len(quote) < 8:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    i = text.lower().find(quote.lower())
    if i >= 0:
        return i, i + len(quote)
    tokens = [re.escape(t) for t in _WS.split(quote) if t]
    if not tokens:
        return None
    m = re.search(r"\s+".join(tokens), text, re.I)
    return (m.start(), m.end()) if m else None


# ------------------------------------------------------------------- client

class OllamaClient:
    def __init__(self, url: str, model: str, api_key: str = "",
                 num_ctx: int = 16384, keep_alive: str = "30m",
                 timeout: float = 180.0, num_predict: int = 8192):
        self.url = url.rstrip("/") + "/api/chat"
        self.model, self.num_ctx, self.keep_alive = model, num_ctx, keep_alive
        # num_predict must cover a reasoning model's hidden trace AND the
        # visible JSON. Capping it low is a false economy: the run completes
        # fast and produces nothing.
        self.num_predict = num_predict
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def complete(self, system: str, prompt: str) -> str:
        r = await self._client.post(self.url, json={
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        })
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(f"ollama error: {body['error']}")
        return (body.get("message") or {}).get("content", "")

    async def aclose(self):
        await self._client.aclose()


# ---------------------------------------------------------------- validation

def validate(raw: str, text: str, entities: list[tuple[str, str]],
             predicates: dict[str, str], chunk_id: str, doc_id: str,
             stats: Counter) -> tuple[list[dict], list[dict]]:
    """Turn a model response into accepted relations and rejects-with-reasons.

    Rejection reasons are counted rather than discarded: the distribution is
    how you tell a prompt problem from a model problem. A high
    evidence_not_found rate means the model is paraphrasing; a high
    bad_index rate means the entity list is too long to track.
    """
    accepted, rejected = [], []

    def fail(reason: str, detail: str = ""):
        stats[reason] += 1
        return [], [{"reason": reason, "chunk_id": chunk_id,
                     "detail": detail[:400], "extra": ""}]

    if not (raw or "").strip():
        # A reasoning model that spent its whole num_predict budget on hidden
        # chain-of-thought returns an empty message. The repo's critic.py hit
        # exactly this at num_predict=1024. Distinguish it from malformed
        # JSON: the remedies are different (raise the budget vs fix the prompt).
        return fail("empty_response", "model returned no content")

    data = _extract_json(raw)
    if data is None:
        return fail("unparseable_json", raw)

    for rel in (data.get("relations") or [])[:20]:
        def rej(reason, extra=""):
            stats[reason] += 1
            rejected.append({"reason": reason, "chunk_id": chunk_id,
                             "detail": str(rel)[:180], "extra": extra})

        s_i, o_i = rel.get("subject"), rel.get("object")
        if not isinstance(s_i, int) or not isinstance(o_i, int):
            rej("bad_index_type"); continue
        if not (0 <= s_i < len(entities)) or not (0 <= o_i < len(entities)):
            rej("index_out_of_range"); continue
        if s_i == o_i:
            rej("self_relation"); continue

        pred = (rel.get("predicate") or "").strip()
        proposed = ""
        if pred not in predicates:
            # Not discarded — routed. An unknown predicate is a signal about
            # the ontology, and the suggestion queue is where it gets one
            # month of consideration instead of a silent new edge type.
            proposed = pred or (rel.get("proposed_label") or "")
            pred = "OTHER"
            stats["routed_to_OTHER"] += 1
        elif pred == "OTHER":
            proposed = (rel.get("proposed_label") or "").strip()
            stats["routed_to_OTHER"] += 1

        span = locate(text, rel.get("evidence") or "")
        if span is None:
            rej("evidence_not_found", (rel.get("evidence") or "")[:60]); continue

        # THE VALIDATION GATE. Structural possibility, checked against the
        # same YAML the ontology declares — extractor and validator cannot
        # drift apart because they read one file.
        s_cls, o_cls = entities[s_i][1], entities[o_i][1]
        verdict = shapes.check(pred, s_cls, o_cls)
        if not verdict.ok:
            stats["type_violation"] += 1
            if verdict.inversion_valid:
                stats["type_violation_inverted"] += 1

        self_conf = rel.get("confidence")
        self_conf = float(self_conf) if isinstance(self_conf, (int, float)) else 0.5
        components = {
            "self_report": round(min(max(self_conf, 0.0), 1.0), 3),
            "span_verified": 1.0,          # guaranteed: we located it above
            "gazetteer_hit": 1.0,          # both endpoints came from pass 1
            "predicate_prior": 0.5 if pred == "OTHER" else 1.0,
            # The only INDEPENDENT signal in the set. The pilot showed the
            # model reports 1.0 on everything, which pinned three of the four
            # components and collapsed confidence to a constant.
            "type_match": shapes.type_component(verdict),
        }
        # Deliberately NOT the model's own number. Raw self-report is poorly
        # calibrated (§7 rule 5), so it contributes one weighted term among
        # several and the components are stored for later recalibration.
        confidence = round(
            0.20 * components["self_report"] + 0.20 * components["span_verified"]
            + 0.15 * components["gazetteer_hit"] + 0.10 * components["predicate_prior"]
            + 0.35 * components["type_match"], 3)

        accepted.append({
            "subject_key": entities[s_i][0], "object_key": entities[o_i][0],
            "predicate": pred, "proposed_label": proposed,
            "evidence_text": text[span[0]:span[1]],
            "span": span, "negated": bool(rel.get("negated")),
            "confidence": confidence, "components": components,
            "chunk_id": chunk_id, "doc_id": doc_id,
            "subject_class": s_cls, "object_class": o_cls,
            "type_ok": verdict.ok, "type_reason": verdict.detail,
            "inversion_valid": verdict.inversion_valid,
        })
        stats["accepted"] += 1
    return accepted, rejected


# ------------------------------------------------------------------ pipeline

def _load_checkpoint() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    done = set()
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["chunk_id"])
        except Exception:
            continue
    return done


def _checkpoint(chunk_id: str, n: int) -> None:
    """Append-only. Resumability without a database: interrupt the run at any
    point and the next invocation skips exactly what completed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": chunk_id, "claims": n}) + "\n")


def select_chunks(collection: str | None, dupes: Path, limit: int | None):
    """R3: >=2 distinct non-Vendor entities. Chosen from measurement, not
    intuition — see kgx.dataplane.budget."""
    from docstore import store
    from kgx.dataplane.entity_writer import _superseded, _scopes

    scopes = [s for s in _scopes(collection) if s[2] == "cloud_ai"]
    if not scopes:
        sys.exit("no cloud_ai collections in scope")
    skip = _superseded(dupes) if kgx_config.get_settings().skip_superseded else set()
    ids = [i for i, _, _ in scopes]
    ph = ",".join("?" * len(ids))
    with store.conn() as c:
        rows = c.execute(f"SELECT chunk_id, doc_id, text FROM doc_chunks "
                         f"WHERE conversation_id IN ({ph})", ids).fetchall()

    g = Gazetteer.load()
    weak = set(kgx_config.get_settings().trigger_weak_classes)
    done = _load_checkpoint()
    out = []
    for r in rows:
        if r["doc_id"] in skip or r["chunk_id"] in done:
            continue
        text = r["text"] or ""
        pairs = {(m.canonical, m.entity_class) for m in g.find(text)
                 if not m.ambiguous}
        strong = {k for k, cls in pairs if cls not in weak}
        if len(strong) < 2:
            continue
        # Vendors still go INTO the prompt (offeredBy needs them); they just
        # do not vote for whether the chunk is worth a call. Classes travel
        # with the names so the gate can check them afterwards.
        ents = sorted(pairs, key=lambda p: p[0] not in strong)
        out.append((r["chunk_id"], r["doc_id"], text, ents[:MAX_ENTITIES_PER_CHUNK]))
        if limit and len(out) >= limit:
            break
    print(f"chunks selected: {len(out):,}  (already done: {len(done):,})")
    return out


async def run(chunks, client: OllamaClient, concurrency: int, commit: bool):
    predicates = load_predicates()
    sem = asyncio.Semaphore(concurrency)
    stats: Counter = Counter()
    all_accepted, all_rejected = [], []
    raw_samples: list[tuple[str, str]] = []
    done = 0

    async def one(chunk_id, doc_id, text, entities):
        nonlocal done
        async with sem:
            try:
                raw = await client.complete(SYSTEM, build_prompt(text, entities, predicates))
            except Exception as e:                                  # noqa: BLE001
                stats["llm_error"] += 1
                return [], [{"reason": "llm_error", "chunk_id": chunk_id,
                             "detail": f"{type(e).__name__}: {e}"[:400], "extra": ""}]
            if len(raw_samples) < 10:
                raw_samples.append((chunk_id, raw))
            acc, rej = validate(raw, text, entities, predicates, chunk_id, doc_id, stats)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(chunks)} chunks  accepted={stats['accepted']} "
                      f"rejected={sum(v for k, v in stats.items() if k not in ('accepted','routed_to_OTHER'))}")
            if commit:
                _checkpoint(chunk_id, len(acc))
            return acc, rej

    results = await asyncio.gather(*(one(*c) for c in chunks))
    for acc, rej in results:
        all_accepted.extend(acc)
        all_rejected.extend(rej)

    # Always dump raw responses. When nothing is accepted, this file is the
    # only thing that distinguishes "prompt is wrong" from "model returned
    # nothing" from "endpoint rejected the request".
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump = OUT_DIR / "raw_responses.txt"
    with dump.open("w", encoding="utf-8") as f:
        for cid, raw in raw_samples:
            f.write(f"===== {cid} =====\n{raw!r}\n\n")
    print(f"  raw model responses -> {dump}")
    return all_accepted, all_rejected, stats


def persist(accepted: list[dict]) -> None:
    """Map gazetteer keys to entity IRIs, then write reified claims."""
    keys = sorted({a["subject_key"] for a in accepted} | {a["object_key"] for a in accepted})
    id_by_key = repo.entities_by_key("cloud_ai", keys)
    missing = [k for k in keys if k not in id_by_key]
    if missing:
        print(f"  WARNING: {len(missing)} entities not in graph "
              f"(run entity_writer --commit first): {missing[:5]}")

    cfg = kgx_config.get_settings()
    now = _now()
    records, skipped = [], 0
    for i, a in enumerate(accepted):
        s, o = id_by_key.get(a["subject_key"]), id_by_key.get(a["object_key"])
        if not s or not o:
            skipped += 1
            continue
        # Quarantine, never delete: the reason is retained so a constraint
        # later judged too tight can be loosened and these replayed without
        # paying for re-extraction.
        if not a.get("type_ok", True):
            status = "quarantined"
        elif a["confidence"] < cfg.min_confidence:
            status = "quarantined"
        else:
            status = "valid"
        records.append(repo.ClaimRecord(
            id=repo.mint_iri("cloud_ai", "claim"),
            namespace="cloud_ai", subject_iri=s, predicate=a["predicate"], object_iri=o,
            predicate_module="cloud_ai", proposed_label=a["proposed_label"],
            evidence_chunk_id=a["chunk_id"], evidence_span_start=a["span"][0],
            evidence_span_end=a["span"][1], source_doc_id=a["doc_id"],
            source_tier="self_authored", confidence=a["confidence"],
            negated=a["negated"], extraction_route="llm",
            extractor_model=cfg.extract_model, extractor_version=cfg.extractor_version,
            prompt_version=PROMPT_VERSION, recorded_at=now, validation_status=status)
        )
    # OTHER never reaches the served graph. It is a proposal, not a fact.
    servable = [r for r in records
                if r.predicate != "OTHER" and r.validation_status == "valid"]
    quarantined = len(records) - len(servable)
    written = 0
    for i in range(0, len(servable), 500):
        written += repo.upsert_claims("cloud_ai", servable[i:i + 500])
    print(f"\nclaims written {written:,}   quarantined/held: {quarantined:,}   "
          f"unresolved endpoints: {skipped:,}")


def write_reports(accepted, rejected, stats) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "claims_preview.tsv"
    with p.open("w", encoding="utf-8") as f:
        f.write("subject\tsubj_class\tpredicate\tobject\tobj_class\ttype_ok\t"
                "inversion_valid\tnegated\tconfidence\tevidence\tchunk_id\n")
        for a in sorted(accepted, key=lambda a: -a["confidence"]):
            ev = a["evidence_text"].replace("\t", " ").replace("\n", " ")[:160]
            f.write(f"{a['subject_key']}\t{a.get('subject_class','')}\t{a['predicate']}\t"
                    f"{a['object_key']}\t{a.get('object_class','')}\t{a.get('type_ok')}\t"
                    f"{a.get('inversion_valid')}\t{a['negated']}\t{a['confidence']}\t"
                    f"{ev}\t{a['chunk_id']}\n")
    q = OUT_DIR / "claims_rejected.tsv"
    with q.open("w", encoding="utf-8") as f:
        f.write("reason\tchunk_id\tdetail\textra\n")
        for r in rejected:
            detail = str(r.get("detail", "")).replace("\t", " ").replace("\n", " ")
            f.write(f"{r.get('reason','?')}\t{r.get('chunk_id','')}\t"
                    f"{detail}\t{r.get('extra','')}\n")
    s = OUT_DIR / "predicate_suggestions.tsv"
    props = Counter(a["proposed_label"] for a in accepted
                    if a["predicate"] == "OTHER" and a["proposed_label"])
    with s.open("w", encoding="utf-8") as f:
        f.write("proposed_label\tcount\n")
        for label, n in props.most_common():
            f.write(f"{label}\t{n}\n")
    print(f"wrote {p}, {q}, {s}")

    if not accepted:
        top = stats.most_common(1)[0][0] if stats else "unknown"
        print("\n  NOTHING WAS ACCEPTED. Dominant reason: " + top)
        print("  Inspect kgx/out/raw_responses.txt — it holds the model's actual output.")
        hint = {
            "empty_response": "raise --num-predict (a reasoning model's hidden "
                              "trace consumed the budget; critic.py hit this at 1024)",
            "unparseable_json": "the model ignored format=json; try --model with a "
                                "non-reasoning model, or raise --num-predict",
            "llm_error": "endpoint/auth/timeout — see the detail column in claims_rejected.tsv",
            "evidence_not_found": "the model is paraphrasing rather than quoting",
        }.get(top)
        if hint:
            print(f"  Likely fix: {hint}")

    print(f"\n=== EXTRACTION SUMMARY ===")
    for k, v in stats.most_common():
        print(f"  {k:<22}{v:>7,}")
    if accepted:
        print("\n--- predicate distribution ---")
        for k, v in Counter(a["predicate"] for a in accepted).most_common():
            print(f"  {k:<22}{v:>7,}")
        bad = [a for a in accepted if not a.get("type_ok", True)]
        inv = [a for a in bad if a.get("inversion_valid")]
        print(f"\n--- validation gate ---")
        print(f"  type-valid            {len(accepted) - len(bad):>7,}")
        print(f"  quarantined           {len(bad):>7,}"
              f"  ({len(bad)/max(1,len(accepted)):.0%})")
        print(f"  of which invertible   {len(inv):>7,}"
              f"  <-- direction bias; fix the PROMPT, not the data")
        if bad:
            print("\n  worst offenders:")
            for a in sorted(bad, key=lambda a: -a["confidence"])[:8]:
                arrow = "<-INVERT" if a.get("inversion_valid") else ""
                print(f"    {a['subject_key'][:20]:<22}({a.get('subject_class','')[:11]:<12})"
                      f"-{a['predicate']:<16}-> {a['object_key'][:20]:<22}"
                      f"({a.get('object_class','')[:11]}) {arrow}")

        print("\n--- 15 highest-confidence claims ---")
        for a in sorted(accepted, key=lambda a: -a["confidence"])[:15]:
            neg = "NOT " if a["negated"] else ""
            print(f"  {a['subject_key'][:24]:<26}{neg}{a['predicate']:<18}"
                  f"{a['object_key'][:24]:<26}{a['confidence']}")


async def amain(a) -> None:
    try:
        from config import get_settings as repo_settings
        rs = repo_settings()
        url, key = rs.ollama_inference_url, rs.ollama_inference_api_key
        default_model, keep = rs.ollama_inference_model, rs.ollama_keep_alive
    except Exception:
        url, key, default_model, keep = "http://localhost:11434", "", "qwen2.5:14b", "30m"

    model = a.model or default_model
    chunks = select_chunks(a.collection, Path(a.dupes), a.limit)
    if not chunks:
        print("nothing to do.")
        return
    if "-cloud" in model and not a.limit:
        print(f"\n  WARNING: {model} is a CLOUD model and this is a "
              f"{len(chunks):,}-call batch.")
        print("  Run with --limit 200 first, or pass --model <local model>.")
        if input("  type 'yes' to continue: ").strip().lower() != "yes":
            return

    print(f"model={model}  concurrency={a.concurrency}  commit={a.commit}")
    client = OllamaClient(url, model, key, num_ctx=a.num_ctx, keep_alive=keep,
                          num_predict=a.num_predict)
    try:
        accepted, rejected, stats = await run(chunks, client, a.concurrency, a.commit)
    finally:
        await client.aclose()

    write_reports(accepted, rejected, stats)
    if not a.commit:
        print("\nDRY RUN — nothing written to the graph, no checkpoint recorded.")
        return
    persist(accepted)
    print(f"\ngraph now: {repo.stats('cloud_ai')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract relations into reified claims.")
    p.add_argument("--collection", default=None)
    p.add_argument("--limit", type=int, default=None, help="pilot on N chunks")
    p.add_argument("--model", default=None, help="override the repo's inference model")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--num-ctx", type=int, default=16384)
    p.add_argument("--num-predict", type=int, default=8192,
                   help="must cover a reasoning model's hidden trace AND the JSON")
    p.add_argument("--commit", action="store_true")
    p.add_argument("--dupes", default="kgx/out/duplicate_clusters.tsv")
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
