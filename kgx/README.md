# kgx — ontology plane

A plug-and-play sidecar for the SEARCH> app. Builds an ontology-driven
knowledge graph alongside the existing one, so both can run and be compared
by flipping a config flag.

**The rule that makes it removable:** `kgx/` imports the repo. The repo never
imports `kgx/`.

---

## 1. Integration — exactly two lines

Dropping the folder in does **nothing**. Python executes nothing on its own.
The plane activates only when `switch.install()` is called.

In `main.py`, at the end of `_startup()`:

```python
@app.on_event("startup")
async def _startup() -> None:
    store.init_db()
    ...
    from kgx import switch          # <-- ADD
    switch.install()                # <-- ADD
```

That is the **only** edit anywhere in the repo. `retrieve.py`,
`graph_hydrate.py`, `ingest.py`, `store.py`, `chunker.py`, the SQLite schema
and the existing `:Entity` graph are all untouched.

Remove the two lines and the app is byte-for-byte what it was.

### How the bind works

`retrieve.context_for_query` imports hydrate *inside* the function:

```python
from docstore.graph_hydrate import hydrate as graph_hydrate   # line 328
```

Because that runs on every call, `install()` replaces the attribute on the
`docstore.graph_hydrate` module and `retrieve.py` picks up the new function
without being edited.

**The fragility:** if that import is ever hoisted to module level,
`retrieve.py` captures its own reference at import time and the bind silently
stops working. `switch.verify()` checks for this and prints loudly. For
anything beyond a trial, use the explicit patch in `switch.EXPLICIT_PATCH` —
uglier, but it cannot fail quietly.

### What is NOT wired

Extraction does not run automatically on ingest. Entity writing and relation
extraction are manual CLI commands. Auto-extraction would need a phase added
to `jobs/pipeline.py`, which *is* a repo edit — deliberately deferred.

---

## 2. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KGX_MODE` | `legacy` | `legacy` \| `shadow` \| `ontology` \| `compare` |
| `KGX_IDENTIFIER_SALT` | *(unset)* | **Required** for any non-legacy mode. HMAC salt for personal-namespace identifier hashing |
| `KGX_TRIGGER_RULE` | `R3` | Relation-pass trigger |
| `KGX_EXTRACT_MODEL` | repo's inference model | Override for extraction |
| `KGX_EXTRACT_CONCURRENCY` | `4` | Parallel extraction calls |
| `KGX_MIN_CONFIDENCE` | `0.35` | Below this a claim is quarantined |

### Modes

| Mode | Behaviour |
|---|---|
| `legacy` | No bind at all. Plane inert. **Default.** |
| `shadow` | kgx hydration runs, its trace is logged, its blocks are **discarded**. No answer is affected. The standing mode. |
| `ontology` | kgx hydration replaces the legacy path. |
| `compare` | Both run; both returned, labelled. ~2× hydration cost — admin views and evals, not production traffic. |

Startup refuses any non-legacy mode while `KGX_IDENTIFIER_SALT` is unset: the
placeholder default would make every personal-namespace hash reproducible
from source.

---

## 3. Onboarding a NEW collection

The steps below are the whole method, in order. Skipping the measurement
steps is how you end up with the graph this project replaced.

### Step 0 — Ingest normally

Use the existing app pipeline. kgx reads `doc_chunks` and `documents`; it
never ingests anything itself.

### Step 1 — Map the collection to a namespace

`kgx/config.py`:

```python
COLLECTION_NAMESPACE = {
    "library":            "cloud_ai",
    "identities & certs": "personal",
    "my new collection":  "cloud_ai",     # <-- add
}
```

`namespace_for()` **fails closed**. An unmapped collection raises rather than
defaulting, because a wrong default routes personal records into the Cloud/AI
graph the first time someone renames something.

### Step 2 — Decide: existing module, or a new domain?

Existing module if the collection is about the same *kinds of things*.
A new module if the classes genuinely differ (Cloud/AI vs identity documents).

**If a new domain is needed, write the competency questions FIRST.** A class
no question needs does not get built. `semantic/competency_questions/*.yaml`
is the scope gate, not documentation.

Then `semantic/modules/<domain>.yaml` importing `upper.yaml`, containing only
what is domain-specific: classes, a closed predicate enum, and
`predicate_constraints` (machine-readable domain/range — every predicate
needs one, `shapes.load_constraints()` raises if any is missing).

### Step 3 — Mine the vocabulary from the corpus, don't invent it

```bash
python -m kgx.mine_corpus terms
```

Produces `out/gazetteer_hits.tsv` (what the seed already covers) and
`out/candidate_terms.tsv` (the growth queue, ranked by document spread).

Triage the candidates into `semantic/gazetteer/<domain>.yaml`. Rules:

- **Hub exclusions** — anything appearing in >30% of documents is barred
  unless a CQ needs it. A node connected to half the corpus discriminates
  nothing. (`API` hit 46% of documents; excluded.)
- **Aliases matter more than canonicals.** Matching is separator-insensitive,
  so `Retrieval-Augmented Generation`, `Retrieval Augmented Generation` and
  `RAG` are one entity. Missing aliases cost real recall — one omission lost
  37 documents.
- **Ambiguous terms get listed, not guessed.** `Vault` means HashiCorp Vault
  *and* Azure Key Vault. They are emitted as unresolved mentions carrying a
  candidate list, never silently fused.
- **Compound product names.** `Grafana Tempo` must be its own entry or
  longest-match splits it into `Grafana` + `Tempo` and every claim attaches
  to the wrong node.

### Step 4 — Deduplicate before extracting

```bash
python -m kgx.mine_corpus dupes
```

Not optional and not cosmetic — see the bug in §5. Retired documents are
marked `SUPERSEDED`, never deleted; old citations must keep resolving.

Check: largest cluster should be small (single digits). A cluster containing
tens of documents means the linkage rule is too loose — raise `--jaccard`.

### Step 5 — Verify entity coverage (free, no LLM)

```bash
python -m kgx.dataplane.gazetteer_matcher selftest     # ~1s, no DB
python -m kgx.dataplane.gazetteer_matcher report
```

What to read:

- `chunks with >=2 mentions` — the relation-pass workload
- `documents with 0 hits` — should match collections in *other* namespaces.
  Unexpected in-domain zeroes mean a gazetteer hole; patch it now, not later
- `ambiguous mentions` — above ~5% and the ambiguity list needs splitting
- eyeball `out/mention_sample.tsv`; a wrong offset here becomes a wrong
  `evidence_span` in every claim built on it

### Step 6 — Cost the LLM pass before spending

```bash
python -m kgx.dataplane.budget --exclude-superseded out/duplicate_clusters.tsv
```

Six candidate trigger rules with chunk counts and estimated hours. Pick with
numbers. If the spread between rules is small, choose for recall, not cost.

### Step 7 — Prove isolation, then write entities

```bash
python -m kgx.repositories.graph_repo selftest     # MUST pass first
python -m kgx.dataplane.entity_writer              # dry run
python -m kgx.dataplane.entity_writer --commit
```

Never run extraction before the isolation test passes. There must be no
window in which personal-namespace entities exist while the boundary is
merely intended.

Re-running is safe: MERGE is keyed on `(namespace, source_key)` and IRIs are
set ON CREATE only.

> **Renaming a gazetteer canonical is a migration, not an edit.** Change
> `Azure OpenAI Service` to `Azure AI Foundry` and the writer creates a NEW
> node. The rename needs the old name added as an alias plus a `MERGED_INTO`
> redirect, so existing citations keep resolving.

### Step 8 — Extract relations, pilot first

```bash
python -m kgx.dataplane.relation_extractor --limit 50            # dry run
python -m kgx.dataplane.relation_extractor --limit 200 --commit  # pilot
python -m kgx.dataplane.relation_extractor --commit              # full pass
```

Resumable via append-only checkpoint; dry runs write no checkpoint.

Diagnostics to read after the pilot:

| Signal | Meaning |
|---|---|
| `evidence_not_found` >15% | The model is paraphrasing instead of quoting. Harden the prompt |
| `empty_response` | A reasoning model spent its budget on hidden trace. Raise `--num-predict` |
| `type_violation` share | Structural errors. High + `invertible` = direction bias; fix the **prompt**, not the data |
| `predicate_suggestions.tsv` | What the ontology is missing. First input to the next version |

### Step 9 — Flip the switch

```bash
KGX_MODE=shadow      # watch the logs; nothing reaches an answer
KGX_MODE=compare     # both graphs side by side
KGX_MODE=ontology    # serve from the new graph
```

---

## 4. Changing the semantic plane

The semantic plane has its own release train. It is not edited casually.

| Change | Class | Consequence |
|---|---|---|
| Add a class or predicate | MINOR | Forward-only; backfill optional |
| Add `predicate_constraints` | MINOR | Re-validate existing claims |
| Rename a canonical | **MIGRATION** | Alias + `MERGED_INTO` redirect required |
| Tighten domain/range | **MAJOR** | Quarantines existing claims; they can be replayed |
| Remove a predicate | **MAJOR** | Orphans every claim using it |

Bump `version:` in the module, update the changelog block at the top, and
stamp `module_version` on new records. Records carry the version that wrote
them, which is what makes selective backfill possible.

New predicates should come from `out/predicate_suggestions.tsv` — the
`OTHER`-routed proposals — not from intuition.

---

## 5. Bugs found in the existing knowledge graph

These are all in the *original* pipeline, unrelated to kgx. Recorded because
they explain why the old graph behaved as it did.

### 5.1 Duplicate documents silently skew what the model sees ← *the subtle one*

The whole chain:

1. `documents` contains near-duplicates — ~11% of the corpus. Same content
   as `.docx` and `.md`, `v1`/`v2` pairs, `Combined_Document` snapshots.
2. `graph_pipeline` weights each `CO_OCCURS_WITH` edge by **how many chunks
   co-mention the pair**.
3. So N copies of a document multiply every edge weight in its neighbourhood
   by roughly N.
4. `graph_store.py:201` — hydration selects facts with
   `ORDER BY r.weight DESC LIMIT $limit`.

**The facts injected into the prompt are therefore the ones most inflated by
duplication.** Nothing errors, nothing logs, no test fails. The graph simply
answers with whatever you happened to save twice — a popularity contest that
duplicates win. This is why dedup runs *before* extraction in kgx, not after.

### 5.2 Transitive over-merge in entity resolution

`entity_resolver._UnionFind` (line 172) clusters by connected components.
A~B and B~C merges A with C even when A≁C. Chains collapse silently; every
aggregate over the merged entity is then wrong. kgx uses complete-linkage
with a size cap instead.

### 5.3 Name-based node identity

`graph_store.py:108` merges on
`(collection_id, label, canonical_name)`. Identity *is* the name, so a rename
breaks every existing reference and there is no `MERGED_INTO` redirect and no
unmerge path. kgx uses opaque ULIDs with a separate `source_key` for lookup.

### 5.4 spaCy never ran — the entity layer is a Title-Case regex

All 200,541 mentions carry the label `PROPN`, which only `_enrich_fallback`
emits. `enrich_chunks` catches the missing `en_core_web_sm` and degrades
silently — fail-open working as designed, hiding a total capability loss.

Worse, `_PROPER` requires lowercase letters after the capital, so **acronyms
are structurally unrepresentable**: zero all-caps forms across 36,574
distinct entities. No `AWS`, `API`, `RAG`, `MCP`, `IAM`, `S3`. Top entities
are `The` (476 docs), `This`, `Text`, `Image`, `Tables`.

*Repo-side fix worth making regardless:* `enrich_chunks` should log loudly
when it degrades. Fail-open is right; fail-open silently, forever, is how you
get here.

### 5.5 Legacy hydration is broken at both ends

`graph_hydrate._entity_names_from_hits` seeds from `h["entities"]` — the same
`PROPN` column — then looks those names up in a graph built from that column.
It searches for `The` in a graph of `The`.

### 5.6 Sensitivity is computed but never enforced

`ingest.classify()` detects Aadhaar, PAN, phone and card patterns and
escalates to `confidential`. But `allowed_principals` is hardcoded `["*"]`.
The label is computed and nothing acts on it. The corpus contains scanned
identity documents — one with an Aadhaar number in the filename.

### 5.7 Silent extraction failures

Documents with `chunk_count = 0` sit in the index with a success status.
Nothing surfaces them.

---

## 6. What was tried and rejected

Four techniques were measured on this corpus. Recorded so they are not
retried on a hunch.

| Technique | Result | Why |
|---|---|---|
| **Gazetteer matching** | ✅ **kept** | 320/330 seed terms observed, 109,657 verified mentions, 0.9% ambiguity |
| **Type constraints** | ✅ **kept** | Caught every error class in the pilot, including reversed relations |
| **Trigger lexicon (T1)** | ✅ **kept** | 21% of chunks dropped for containing no relational verb. Free |
| **Gap regex patterns** | ❌ rejected | 61 claims from 6,516 chunks (0.9%). Required the inter-entity text to be a bare verb phrase; real prose is not |
| **NLI entailment** | ❌ rejected | Separation gap +0.09. Contradicted *true* claims about `Azure Entra ID` at p=1.00 — a post-cutoff rename it has no representation of |
| **Dependency parsing** | ❌ rejected | 0.01 claims/chunk. 79% of chunks contain a relational verb but only 0.5% contain parseable subject-verb-object — the corpus is largely bullet lists and tables, which have no clause to parse |

**The rule these establish:**

> Cheap tiers that depend on **structure** work here.
> Cheap tiers that depend on **world knowledge** fail.

The corpus is full of products released after any general model's training
data. Anything that needs to know what Debezium *is* has to be the LLM —
that is the one component that actually knows. Anything that only needs to
know what a string looks like, or what English grammar does, or what the
ontology permits, can be done for free.

The rejected modules are left in the tree rather than deleted. With their
numbers attached they are a record of what was tried, which is worth more
than a clean directory.

---

## 7. Layout

```
kgx/
├── README.md
├── config.py                       mode switch, collection→namespace routing
├── switch.py                       the single integration point
├── mine_corpus.py                  read-only: terms + dupes
├── dataplane/
│   ├── gazetteer_matcher.py        deterministic extraction  ✅
│   ├── shapes.py                   type constraint gate      ✅
│   ├── budget.py                   cost the LLM pass         ✅
│   ├── entity_writer.py            mentions → :KgxEntity     ✅
│   ├── relation_extractor.py       LLM → :KgxClaim           ✅
│   ├── cascade.py                  T0/T1 kept, T2 rejected   ◐
│   ├── parse_tier.py               rejected, retained        ✗
│   └── nli_verifier.py             rejected, retained        ✗
├── repositories/
│   └── graph_repo.py               namespaced Neo4j, 7/7 isolation
├── retrievalplane/
│   └── hydrate.py                  the read path
├── semantic/                       own release train
│   ├── upper.yaml                  shared: identity, provenance, Claim
│   ├── modules/{cloud_ai,personal}.yaml
│   ├── gazetteer/cloud_ai.yaml
│   └── competency_questions/{cloud_ai,personal}.yaml
└── out/                            generated reports — gitignore this
```

## 8. Self-tests

Every one runs without touching production data.

```bash
python -m kgx.dataplane.gazetteer_matcher selftest   # 13/13, no DB
python -m kgx.dataplane.shapes                       # 12/12, no DB
python -m kgx.dataplane.cascade selftest             # 7/7,  no DB
python -m kgx.repositories.graph_repo selftest       # 7/7,  needs Neo4j
python -m kgx.switch                                 # prints status JSON
```

## 9. Known state / not done

- Graph currently holds **436 entities** and **87 claims written before the
  type gate existed** — roughly a third are type-invalid. Clear the namespace
  and re-run both writers before trusting `compare`.
- `claims_preview.tsv` accumulates across runs; clear it between comparisons.
- `offeredBy` should be dropped from prompts (it is already stored as
  `owner`) and T1 wired into `select_chunks`.
- Personal namespace has an ontology and CQs but **no extractor**. It needs
  dates, typed identifiers and issuers — no gazetteer, no LLM.
- No eval harness yet. The competency questions are written; nothing scores
  against them.
- Auto-extraction on ingest is not wired.
