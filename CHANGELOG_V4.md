# V4 — Batch ingestion jobs, collections, code & archive support

Integrated into the V3 codebase. Everything that worked before still works the
same way; the changes below are additive except where noted.

---

## The design problem, and how it was solved

Every document, chunk and vector in V3 is keyed by `conversation_id`, and
`docstore/retrieve.py` was explicit about why:

> Scope is always a single conversation. Cross-thread retrieval is deliberately
> not offered: a user who uploads a contract in one thread has not consented to
> it surfacing in another.

That is the right rule for uploads and the wrong one for a batch job ingesting
ten thousand documents from a shared drive — those exist so that *many* chats
can search them, and there is no conversation they belong to.

**Collections** resolve this. A collection is a scope that holds documents but
no messages. It reuses the `conversations` table with `kind='collection'`, which
is a pragmatic choice worth stating plainly: `documents`, `doc_chunks` and
`chunk_vectors` already carry a foreign key to `conversations(id)` with
`ON DELETE CASCADE`, and that machinery works unchanged for either kind. The
alternative — a separate table plus a nullable `collection_id` on three child
tables — would have touched nearly every query in `store.py` to gain a naming
distinction.

Consent survives. A collection is searchable from a chat only if a row exists in
`conversation_collections`. Nothing attaches implicitly, and a chat upload is
never promoted into a collection. The rule becomes *"this conversation plus what
it has explicitly been given"* — still a boundary, just one the operator can
widen deliberately.

---

## New files

| File | Purpose |
|---|---|
| `docstore/filetypes.py` | The one registry of ingestible types. Upload and batch both read it. |
| `docstore/archive.py` | Nested ZIP expansion with bomb/traversal/symlink guards. |
| `docstore/code_extractors.py` | `.py/.js/.css/...` extraction + a code-specific quality scorer. |
| `docstore/code_chunker.py` | Symbol-aware chunking so functions aren't cut mid-body. |
| `docstore/collections.py` | Collection CRUD, attach/detach, scope resolution. |
| `jobs/models.py` | `Job`, `FolderSpec`, `JobOptions`, `Phase`. |
| `jobs/store.py` | Queue persistence and the claim protocol. |
| `jobs/pipeline.py` | The multicore per-document pipeline. |
| `jobs/runner.py` | Phase orchestration for one job. |
| `jobs/worker.py` | The daemon. Runs as a **separate process** from uvicorn. |
| `routers/jobs.py` | `/jobs` page, `/api/jobs*`, `/api/collections*`, `/api/filetypes`. |
| `templates/jobs.html`, `static/js/jobs.js`, `static/css/jobs.css` | Dashboard. |
| `static/css/collections.css` | Collections sheet in the chat UI. |

## Modified files

| File | Change |
|---|---|
| `docstore/store.py` | `conversations.kind` + `description`; `conversation_collections`; `documents.source_uri/parent_doc_id/job_id`; `_migrate()` for in-place upgrades; scoped query variants. |
| `docstore/retrieve.py` | `scope_ids: list[str]` throughout. `context_for_query()` resolves scopes itself, so no caller changes. |
| `docstore/extractors.py` | `CodeExtractor` and `HtmlModeRouter` registered ahead of `HtmlExtractor`. |
| `docstore/ingest.py` | `MEDIA_TYPES` now derived from the registry; `_chunk()` routes code; new `register_file()` for in-place batch registration. |
| `routers/uploads.py` | ZIP fan-out; per-type size caps; human-readable rejection reasons. |
| `main.py` | Mounts the jobs router; initialises job tables; creates ingest roots. |
| `templates/chat.html` | Jobs link + collections sheet in the rail. |
| `static/js/chat.js` | Server-driven `accept`; `expanding` state; child counts; collections UI. |
| `config.py` | `ingest_allowed_roots`, job concurrency knobs, code-chunk sizes. |

**No files were deleted** except the two `- Copy (Before Guardrails).py` backups.

---

## Three things worth knowing

### 1. Code would have been silently quarantined

`score_text()` in `extractors.py` penalises a low alphanumeric ratio by 0.35 and
a short file by another 0.20. Source code is roughly a third punctuation, so
nearly every `.py`/`.js`/`.css` file would have scored under the floor and gone
to `quarantined / low_extraction_quality`. The corpus would have reported itself
healthy and contained no code at all.

`score_code()` inverts the intuition — punctuation density is evidence the
extraction *worked* — and instead catches minification, binary content
mislabelled as text, and vendored `node_modules`.

Verified: `app.py` → quality **1.0**, `style.css` → **1.0**.

### 2. Multicore is real parallelism, not threads

`docstore/ingest.py` parks CPU work on `asyncio.to_thread`, which is right for
one interactive upload and wrong for ten thousand documents — PDF parsing,
chunking and spaCy are CPU-bound Python, so N threads give you one core.

`jobs/pipeline.py` splits by where work actually blocks:

- **ProcessPoolExecutor** — extract → classify → chunk → enrich. Real cores.
  Tasks return plain dicts, so nothing unpicklable crosses the boundary.
- **Parent event loop** — database writes and embedding. Embedding is HTTP, so
  it wants concurrency, not cores; writes stay in one process because SQLite
  with many writers is a lock-contention problem nobody needs.

Stage *logic* is unchanged and still imported from `docstore.*`.

### 3. The worker must be a separate process

Not tidiness. A pool sized to the machine's cores inside uvicorn means ingestion
competes with chat for CPU, and forking from a process holding SQLite handles and
an event loop hangs in ways that only appear under load.

---

## Running it

```bash
# terminal 1 — API
uvicorn main:app --host 0.0.0.0 --port 1976

# terminal 2 — ingestion worker
python -m jobs.worker
```

`INGEST_ALLOWED_ROOTS` (default `data/incoming`) confines the folder textbox on
the job form. Without it that input is an arbitrary-file-read endpoint — "/etc"
is a valid path, and the result would be indexed into a collection any chat can
attach. Multiple roots are `os.pathsep`-separated.

---

## Verified in this build

| Check | Result |
|---|---|
| All 54 Python files parse | ✅ |
| `chat.js` / `jobs.js` syntax | ✅ |
| Both templates render under Jinja | ✅ |
| Schema + additive migration on an existing DB | ✅ |
| Atomic job claim (second worker gets `None`) | ✅ |
| Nested ZIP (`outer.zip` → `inner.zip` → files) | ✅ 2 archives, 2 files |
| Code quality scoring | ✅ 1.0, not quarantined |
| Symbol chunking | ✅ `['app.py', 'load_config', 'PaymentGateway']` |
| Full job run | ✅ `succeeded`, 5 discovered, 4 documents |
| Content dedup | ✅ identical `notes.md`/`readme.md` collapsed to one doc |
| Re-run resume | ✅ 5 discovered, 5 skipped, 0 reprocessed |
| PII → classification escalation | ✅ `notes.md` → `confidential` |
| Retrieval consent boundary | ✅ 0 hits before attach, 3 after |
| Graceful embed failure | ✅ `degraded`, verify phase warned |

The `degraded` status in the smoke test is correct behaviour, not a defect —
there is no Ollama in the build sandbox, so embedding failed and the pipeline
kept the chunks (lexical search still works) rather than discarding the document.

---

## Not done, and deliberately so

- **Spreadsheets** remain `deferred_tabular`. They need a row-oriented chunker
  with the header carried into every chunk; running them through the prose
  chunker produces confidently-cited nonsense.
- **Vision extraction inside the pool** blocks a process rather than a core —
  wasteful but bounded. `skip_images` exists for corpora where those files
  aren't worth the wall-clock.
- **Brute-force cosine** is still the dense arm. Past roughly 50k chunks per
  scope, `retrieve.vector_search` is the single function to swap.
- **One job at a time.** Two concurrent jobs would double the pool and
  oversubscribe the box, making both slower than running them in sequence.

---

## The thing to watch

There are now two ingestion paths feeding one retrieval layer: the per-document
pipeline behind `/api/uploads`, and the batch pipeline behind the job runner.
They currently share the extractor registry, the classifier, the chunker configs
and the embed model — `jobs/pipeline.py` calls into `docstore.*` rather than
reimplementing it, specifically to keep that true.

If they ever drift — a different chunk size, a different classification default,
a different embedding model — the symptom is inconsistent answers depending on
how a document happened to arrive, which is close to undiagnosable from the query
side because the query looks fine. Worth a test that ingests the same file both
ways and asserts the chunks match.
