"""docstore/store.py — SQLite persistence for conversations, messages and the
uploaded-document corpus.

Why SQLite and not Qdrant/Postgres
---------------------------------
The reference pipeline wrote vectors to Qdrant and lexical rows to SQLite FTS5,
then swapped an alias at I7. That architecture exists to keep a *live* index
serving while a *build* collection is populated — it is the right answer for a
shared enterprise corpus rebuilt nightly.

Per-conversation uploads are a different shape entirely: tens to low-hundreds of
chunks, scoped to one thread, written once and read immediately. There is no
live index to protect and nothing to swap, so the alias machinery would be pure
ceremony. Vectors live here as float32 BLOBs and are scored in-process. FTS5 is
kept, because that part of I6 transfers directly.

If a corpus ever outgrows this (>~50k chunks), the swap-in point is
`docstore.retrieve.vector_search` — nothing else needs to know.

Schema is created on demand and is additive-safe; `init_db()` is idempotent.
Foreign keys cascade so deleting a conversation reclaims its documents,
chunks and vectors in one statement.
"""
from __future__ import annotations

import array
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import get_settings

cfg = get_settings()

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- `kind` turns this table into a scope table. 'chat' rows are conversations and
-- own messages; 'collection' rows are named corpora built by batch jobs and own
-- no messages. Both own documents, which is why they share a table: documents,
-- doc_chunks and chunk_vectors all carry a foreign key to conversations(id) with
-- ON DELETE CASCADE, and that machinery works unchanged for either kind.
CREATE TABLE IF NOT EXISTS conversations (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT 'New chat',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0,
    kind            TEXT NOT NULL DEFAULT 'chat',
    description     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_conv_updated ON conversations(archived, updated_at DESC);
-- ix_conv_kind is created AFTER _migrate() in init_db(), not here — 'kind' is
-- an additive column on a database that predates it, and an index on a
-- not-yet-existent column would fail before the migration ever runs. See the
-- ordering note on init_db().

-- Which collections a chat is allowed to search. Retrieval scope is the
-- conversation plus its rows here — never implicit, so attaching a corpus stays
-- a deliberate act rather than something that happens to a user.
CREATE TABLE IF NOT EXISTS conversation_collections (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    collection_id   TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    attached_at     TEXT NOT NULL,
    PRIMARY KEY (conversation_id, collection_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content         TEXT NOT NULL DEFAULT '',
    attachments     TEXT NOT NULL DEFAULT '[]',   -- JSON: [{doc_id,file_name,status}]
    context         TEXT NOT NULL DEFAULT '[]',   -- JSON: worker payloads
    action_logs     TEXT NOT NULL DEFAULT '[]',   -- JSON: execution trace
    feedback        TEXT,                          -- critic evaluation
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS documents (
    id                   TEXT PRIMARY KEY,          -- content-addressed doc_id
    conversation_id      TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    file_name            TEXT NOT NULL,
    media_type           TEXT NOT NULL DEFAULT '',
    size_bytes           INTEGER NOT NULL DEFAULT 0,
    stored_path          TEXT,
    status               TEXT NOT NULL DEFAULT 'pending',
    reason_code          TEXT,
    extractor            TEXT,
    quality              REAL NOT NULL DEFAULT 1.0,
    page_count           INTEGER,
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    data_classification  TEXT NOT NULL DEFAULT 'internal',
    allowed_principals   TEXT NOT NULL DEFAULT '["*"]',
    pii_tags             TEXT NOT NULL DEFAULT '[]',
    warnings             TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_doc_conv ON documents(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS doc_chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    conversation_id     TEXT NOT NULL,
    ordinal             INTEGER NOT NULL,
    text                TEXT NOT NULL,
    char_start          INTEGER NOT NULL DEFAULT 0,
    char_end            INTEGER NOT NULL DEFAULT 0,
    page                INTEGER,
    section_path        TEXT NOT NULL DEFAULT '[]',
    keywords            TEXT NOT NULL DEFAULT '[]',
    entities            TEXT NOT NULL DEFAULT '[]',
    summary             TEXT,
    data_classification TEXT NOT NULL DEFAULT 'internal',
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunk_conv ON doc_chunks(conversation_id);
CREATE INDEX IF NOT EXISTS ix_chunk_doc  ON doc_chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS chunk_vectors (
    chunk_id        TEXT PRIMARY KEY REFERENCES doc_chunks(chunk_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    model           TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    vec             BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_vec_conv ON chunk_vectors(conversation_id);
"""

# FTS5 is optional — some Python builds ship without it. Lexical search
# degrades to LIKE rather than failing the whole ingest.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    conversation_id UNINDEXED,
    text,
    keywords,
    tokenize = 'porter unicode61'
);
"""

_fts_available: bool | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def _db_path() -> Path:
    p = Path(getattr(cfg, "chat_db_path", "data/chat.db"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(_db_path(), timeout=30.0)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _migrate(c: sqlite3.Connection) -> None:
    """Additive column migrations for databases created before a column existed.

    `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so new columns
    in _SCHEMA never reach a database that already has the old shape. Anyone
    upgrading in place would get `no such column: kind` on the first query.
    ALTER TABLE ADD COLUMN is cheap and safe; the duplicate-column error is the
    expected outcome on a fresh database and is swallowed.
    """
    for table, column, ddl in (
        ("conversations", "kind", "TEXT NOT NULL DEFAULT 'chat'"),
        ("conversations", "description", "TEXT NOT NULL DEFAULT ''"),
        ("documents", "source_uri", "TEXT"),
        ("documents", "parent_doc_id", "TEXT"),
        ("documents", "job_id", "TEXT"),
        # The file's own last-modified time on disk — NOT when we ingested it.
        # For a driving licence, a CV, a contract: the file's mtime is a much
        # more honest "which copy is current" signal than ingestion date,
        # since a batch job can ingest a five-year-old scan and a same-day
        # renewal in the same run, seconds apart, with no relationship
        # between when we happened to run the job and which document is
        # actually more recent. NULL for chat uploads, where the browser
        # doesn't reliably expose the original file's mtime.
        ("documents", "source_modified_at", "TEXT"),
        # Per-collection knowledge-graph state. 'none' until someone opts in
        # and builds one — a collection's default behaviour is completely
        # unaffected by any of this until graph_status leaves 'none'.
        #   graph_status              none | building | ready | stale | failed
        #   graph_built_at            when the graph last finished building
        #   graph_chunk_count_at_build the collection's chunk count at that
        #                              moment — compared against the current
        #                              count to detect staleness cheaply,
        #                              without needing a content hash.
        #   graph_stats_json          {"nodes": N, "edges": M} for display
        ("conversations", "graph_status", "TEXT NOT NULL DEFAULT 'none'"),
        ("conversations", "graph_built_at", "TEXT"),
        ("conversations", "graph_chunk_count_at_build", "INTEGER"),
        ("conversations", "graph_stats_json", "TEXT NOT NULL DEFAULT '{}'"),
    ):
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            print(f"[store] migrated: {table}.{column} added")
        except sqlite3.OperationalError:
            pass   # already present


def init_db() -> None:
    """Create schema. Safe to call on every boot.

    Order matters here in a way it easy to get wrong: CREATE TABLE IF NOT
    EXISTS is a no-op on a database that already has the table, so on an
    existing production database, _SCHEMA only ever adds what's genuinely
    missing. _migrate() must run AFTER that (it needs the table to exist to
    ALTER it) but BEFORE anything that indexes a migrated column — an index
    on a column that isn't there yet fails outright, which used to happen
    here for `kind` on any database created before collections existed.
    """
    global _fts_available
    with conn() as c:
        c.executescript(_SCHEMA)
        _migrate(c)
        # Safe only now that source_modified_at / kind / etc. are guaranteed
        # to exist, on a brand new database and a decades-old one alike.
        c.execute("CREATE INDEX IF NOT EXISTS ix_conv_kind ON conversations(kind, title)")
        try:
            c.executescript(_FTS_SCHEMA)
            _fts_available = True
        except sqlite3.OperationalError as e:
            _fts_available = False
            print(f"[store] ⚠️ FTS5 unavailable ({e}); lexical search will use LIKE.")


def fts_enabled() -> bool:
    if _fts_available is None:
        init_db()
    return bool(_fts_available)


# ------------------------------------------------------------------ vectors

def pack_vector(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def unpack_vector(blob: bytes) -> array.array:
    a = array.array("f")
    a.frombytes(blob)
    return a


# ------------------------------------------------------------------ conversations

def create_conversation(title: str = "New chat") -> dict[str, Any]:
    cid, ts = new_id(), _now()
    with conn() as c:
        c.execute(
            "INSERT INTO conversations (id,title,created_at,updated_at) VALUES (?,?,?,?)",
            (cid, title[:500] or "New chat", ts, ts),
        )
    return {"id": cid, "title": title, "created_at": ts, "updated_at": ts,
            "message_count": 0}


def list_conversations(limit: int = 100, q: str | None = None) -> list[dict]:
    # kind='chat' keeps collections out of the rail. They hold documents but no
    # messages, so showing them as openable threads would be a dead end.
    sql = ("SELECT c.*, (SELECT COUNT(*) FROM documents d "
           "WHERE d.conversation_id=c.id AND d.status IN ('indexed','degraded')) AS doc_count "
           "FROM conversations c WHERE c.archived=0 AND c.kind='chat'")
    params: list[Any] = []
    if q:
        sql += " AND c.title LIKE ?"
        params.append(f"%{q}%")
    sql += " ORDER BY c.updated_at DESC LIMIT ?"
    params.append(limit)
    with conn() as c:
        return [dict(r) for r in c.execute(sql, params)]


def get_conversation(conversation_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM conversations WHERE id=?",
                      (conversation_id,)).fetchone()
        return dict(r) if r else None


def rename_conversation(conversation_id: str, title: str) -> None:
    with conn() as c:
        c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                  (title[:500], _now(), conversation_id))


def delete_conversation(conversation_id: str) -> None:
    """Hard delete. Cascades to messages, documents, chunks and vectors."""
    with conn() as c:
        ids = [r["chunk_id"] for r in c.execute(
            "SELECT chunk_id FROM doc_chunks WHERE conversation_id=?",
            (conversation_id,))]
        if ids and fts_enabled():
            c.executemany("DELETE FROM chunks_fts WHERE chunk_id=?",
                          [(i,) for i in ids])
        c.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))


def touch_conversation(conversation_id: str) -> None:
    with conn() as c:
        c.execute("UPDATE conversations SET updated_at=? WHERE id=?",
                  (_now(), conversation_id))


def autotitle_if_default(conversation_id: str, first_prompt: str) -> str | None:
    """Name a thread from its first prompt. Returns the new title, or None.

    Titles are derived, never model-generated: a summarisation call here would
    add latency to the critical path of the first message for a string the user
    can rename in one click.
    """
    with conn() as c:
        r = c.execute("SELECT title FROM conversations WHERE id=?",
                      (conversation_id,)).fetchone()
        if not r or (r["title"] or "").strip() not in ("", "New chat"):
            return None
        title = " ".join((first_prompt or "").split())[:60].rstrip(" ,.;:-")
        title = title or "New chat"
        c.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                  (title, _now(), conversation_id))
        return title


# ------------------------------------------------------------------ messages

def add_message(conversation_id: str, role: str, content: str,
                attachments: list | None = None, context: list | None = None,
                action_logs: list | None = None,
                feedback: str | None = None) -> str:
    mid, ts = new_id(), _now()
    with conn() as c:
        c.execute(
            "INSERT INTO messages (id,conversation_id,role,content,attachments,"
            "context,action_logs,feedback,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, conversation_id, role, content or "",
             json.dumps(attachments or []), json.dumps(context or [])[:2_000_000],
             json.dumps(action_logs or []), feedback, ts),
        )
        c.execute("UPDATE conversations SET updated_at=?, "
                  "message_count=(SELECT COUNT(*) FROM messages WHERE conversation_id=?) "
                  "WHERE id=?", (ts, conversation_id, conversation_id))
    return mid


def get_messages(conversation_id: str, limit: int = 500) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE conversation_id=? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (conversation_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("attachments", "context", "action_logs"):
            try:
                d[k] = json.loads(d[k] or "[]")
            except json.JSONDecodeError:
                d[k] = []
        out.append(d)
    return out


def recent_turns(conversation_id: str, turns: int = 6) -> list[dict]:
    """Last N messages, oldest-first — the window handed to the supervisor.

    Trimmed hard: the supervisor prompt already carries agent capabilities and
    scraped context, and history is the cheapest thing to sacrifice when the
    context budget gets tight.
    """
    with conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (conversation_id, turns)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ------------------------------------------------------------------ documents

def upsert_document(doc: dict) -> None:
    """Insert or update a document row by content-addressed id."""
    ts = _now()
    with conn() as c:
        c.execute("""
            INSERT INTO documents (id,conversation_id,file_name,media_type,size_bytes,
                stored_path,status,reason_code,extractor,quality,page_count,chunk_count,
                data_classification,allowed_principals,pii_tags,warnings,created_at,updated_at,
                source_uri,parent_doc_id,job_id,source_modified_at)
            VALUES (:id,:conversation_id,:file_name,:media_type,:size_bytes,
                :stored_path,:status,:reason_code,:extractor,:quality,:page_count,:chunk_count,
                :data_classification,:allowed_principals,:pii_tags,:warnings,:created_at,:updated_at,
                :source_uri,:parent_doc_id,:job_id,:source_modified_at)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, reason_code=excluded.reason_code,
                extractor=excluded.extractor, quality=excluded.quality,
                page_count=excluded.page_count, chunk_count=excluded.chunk_count,
                data_classification=excluded.data_classification,
                allowed_principals=excluded.allowed_principals,
                pii_tags=excluded.pii_tags, warnings=excluded.warnings,
                stored_path=excluded.stored_path, updated_at=excluded.updated_at,
                source_uri=excluded.source_uri, parent_doc_id=excluded.parent_doc_id,
                job_id=excluded.job_id,
                source_modified_at=COALESCE(excluded.source_modified_at, documents.source_modified_at)
        """, {
            "id": doc["id"], "conversation_id": doc["conversation_id"],
            "file_name": doc.get("file_name", ""),
            "media_type": doc.get("media_type", ""),
            "size_bytes": doc.get("size_bytes", 0),
            "stored_path": doc.get("stored_path"),
            "status": doc.get("status", "pending"),
            "reason_code": doc.get("reason_code"),
            "extractor": doc.get("extractor"),
            "quality": doc.get("quality", 1.0),
            "page_count": doc.get("page_count"),
            "chunk_count": doc.get("chunk_count", 0),
            "data_classification": doc.get("data_classification", "internal"),
            "allowed_principals": json.dumps(doc.get("allowed_principals", ["*"])),
            "pii_tags": json.dumps(doc.get("pii_tags", [])),
            "warnings": json.dumps(doc.get("warnings", [])),
            "created_at": doc.get("created_at", ts), "updated_at": ts,
            # Batch/archive provenance. NULL for a plain chat upload.
            #   source_uri    "/data/in/specs.zip!/design/api.md" for archive children
            #   parent_doc_id the archive row a child came out of
            #   job_id        the ingestion job that produced this document
            "source_uri": doc.get("source_uri"),
            "parent_doc_id": doc.get("parent_doc_id"),
            "job_id": doc.get("job_id"),
            "source_modified_at": doc.get("source_modified_at"),
        })


def set_document_status(doc_id: str, status: str, reason_code: str | None = None,
                        **fields: Any) -> None:
    sets = ["status=?", "reason_code=?", "updated_at=?"]
    vals: list[Any] = [status, reason_code, _now()]
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if isinstance(v, (list, dict)) else v)
    vals.append(doc_id)
    with conn() as c:
        c.execute(f"UPDATE documents SET {','.join(sets)} WHERE id=?", vals)


def list_documents(conversation_id: str, include_paths: bool = False) -> list[dict]:
    """List a conversation's documents, oldest first.

    `stored_path` is OFF by default and must be asked for. Two callers return
    this straight to the browser (routers/uploads.py, routers/conversations.py)
    and stored_path is an absolute server filesystem path. Server-side consumers
    that need to READ the file — docstore/artifacts.py for whole-file injection
    and agents/code_editor.py for editing — pass include_paths=True.

    Leaving the column out of the projection entirely was the earlier bug: both
    of those modules filter on `d.get("stored_path")`, which matched nothing, so
    whole-file mode silently never engaged and the code editor reported "no
    uploaded files in this conversation" for a conversation full of them.
    """
    cols = ("id,file_name,media_type,size_bytes,status,reason_code,"
            "chunk_count,quality,page_count,data_classification,warnings,created_at,"
            "source_uri,parent_doc_id,job_id")
    if include_paths:
        cols += ",stored_path"
    with conn() as c:
        rows = c.execute(
            f"SELECT {cols} FROM documents WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["warnings"] = json.loads(d["warnings"] or "[]")
        except json.JSONDecodeError:
            d["warnings"] = []
        out.append(d)
    return out


def get_document(doc_id: str) -> dict | None:
    with conn() as c:
        r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(r) if r else None


def delete_document(doc_id: str) -> None:
    with conn() as c:
        if fts_enabled():
            ids = [r["chunk_id"] for r in c.execute(
                "SELECT chunk_id FROM doc_chunks WHERE doc_id=?", (doc_id,))]
            c.executemany("DELETE FROM chunks_fts WHERE chunk_id=?",
                          [(i,) for i in ids])
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def document_exists_indexed(doc_id: str) -> bool:
    """True when this exact content is already ingested — the idempotency gate.

    'indexed' AND 'degraded' both count. A degraded document went through
    extraction, chunking, and classification successfully — only the
    embedding step failed, usually a transient issue (an Ollama timeout under
    load during a large bulk run, not a problem with the document itself).
    Its chunks are already saved and already searchable lexically.

    Treating only 'indexed' as done meant every re-run reprocessed every
    degraded document from scratch — full re-extraction, re-chunking,
    re-enrichment — for something that only needed its embedding retried.
    On a large corpus with any embedding flakiness at all, that compounds:
    each run reprocesses everything degraded in the last one, degrades a
    fresh subset under the same load, and the "already indexed" set never
    actually grows the way an operator watching docs_ok expects it to.
    """
    with conn() as c:
        r = c.execute("SELECT status FROM documents WHERE id=?", (doc_id,)).fetchone()
    return bool(r and r["status"] in ("indexed", "degraded"))


# ------------------------------------------------------------------ chunks

def save_chunks(conversation_id: str, records: list[Any]) -> int:
    """Persist ChunkRecords + FTS rows. Vectors are written separately by I5."""
    ts = _now()
    rows, fts_rows = [], []
    for r in records:
        d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        rows.append((
            d["chunk_id"], d["doc_id"], conversation_id, d.get("ordinal", 0),
            d.get("text", ""), d.get("char_start", 0), d.get("char_end", 0),
            d.get("page"), json.dumps(d.get("section_path", [])),
            json.dumps(d.get("keywords", [])), json.dumps(d.get("entities", [])),
            d.get("summary"), d.get("data_classification", "internal"), ts,
        ))
        fts_rows.append((d["chunk_id"], conversation_id, d.get("text", ""),
                         " ".join(d.get("keywords", []))))
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO doc_chunks (chunk_id,doc_id,conversation_id,"
            "ordinal,text,char_start,char_end,page,section_path,keywords,entities,"
            "summary,data_classification,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        if fts_enabled():
            c.executemany("DELETE FROM chunks_fts WHERE chunk_id=?",
                          [(r[0],) for r in fts_rows])
            c.executemany("INSERT INTO chunks_fts (chunk_id,conversation_id,text,"
                          "keywords) VALUES (?,?,?,?)", fts_rows)
    return len(rows)


def save_vectors(conversation_id: str, model: str,
                 pairs: list[tuple[str, list[float]]]) -> int:
    rows = [(cid, conversation_id, model, len(v), pack_vector(v))
            for cid, v in pairs if v]
    with conn() as c:
        c.executemany(
            "INSERT OR REPLACE INTO chunk_vectors (chunk_id,conversation_id,"
            "model,dim,vec) VALUES (?,?,?,?,?)", rows)
    return len(rows)


def chunks_missing_vectors(doc_id: str) -> list[tuple[str, str]]:
    """(chunk_id, text) for chunks not yet embedded — makes I5 resumable."""
    with conn() as c:
        return [(r["chunk_id"], r["text"]) for r in c.execute(
            "SELECT k.chunk_id, k.text FROM doc_chunks k "
            "LEFT JOIN chunk_vectors v ON v.chunk_id=k.chunk_id "
            "WHERE k.doc_id=? AND v.chunk_id IS NULL ORDER BY k.ordinal",
            (doc_id,))]


def iter_vectors(conversation_id: str) -> list[sqlite3.Row]:
    with conn() as c:
        return c.execute(
            "SELECT v.chunk_id, v.vec, v.dim FROM chunk_vectors v "
            "WHERE v.conversation_id=?", (conversation_id,)).fetchall()


def iter_vectors_scoped(scope_ids: list[str]) -> list[sqlite3.Row]:
    """Vectors across a conversation and any collections attached to it.

    Kept as a separate function rather than widening iter_vectors' parameter,
    because the single-scope form is on the hot path for a thread with no
    collections and the IN-clause version costs a query plan it does not need.
    """
    if not scope_ids:
        return []
    marks = ",".join("?" * len(scope_ids))
    with conn() as c:
        return c.execute(
            f"SELECT v.chunk_id, v.vec, v.dim FROM chunk_vectors v "
            f"WHERE v.conversation_id IN ({marks})", scope_ids).fetchall()


def corpus_stats_scoped(scope_ids: list[str]) -> dict:
    """Combined stats across every scope a query may retrieve from."""
    if not scope_ids:
        return {"documents": 0, "chunks": 0, "vectors": 0}
    marks = ",".join("?" * len(scope_ids))
    with conn() as c:
        docs = c.execute(
            f"SELECT COUNT(*) n FROM documents WHERE conversation_id IN ({marks}) "
            f"AND status IN ('indexed','degraded')", scope_ids).fetchone()["n"]
        chunks = c.execute(
            f"SELECT COUNT(*) n FROM doc_chunks WHERE conversation_id IN ({marks})",
            scope_ids).fetchone()["n"]
        vecs = c.execute(
            f"SELECT COUNT(*) n FROM chunk_vectors WHERE conversation_id IN ({marks})",
            scope_ids).fetchone()["n"]
    return {"documents": docs, "chunks": chunks, "vectors": vecs}


def count_documents(conversation_id: str, statuses: tuple[str, ...] | None = None) -> int:
    sql = "SELECT COUNT(*) n FROM documents WHERE conversation_id=?"
    params: list[Any] = [conversation_id]
    if statuses:
        sql += f" AND status IN ({','.join('?' * len(statuses))})"
        params += list(statuses)
    with conn() as c:
        return c.execute(sql, params).fetchone()["n"]


def documents_by_job(job_id: str) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT id,file_name,status,reason_code,chunk_count,quality,"
            "data_classification,created_at FROM documents WHERE job_id=? "
            "ORDER BY created_at", (job_id,)).fetchall()
    return [dict(r) for r in rows]


def job_document_tally(job_id: str) -> dict[str, int]:
    """Status histogram for one job — the dashboard's per-status counters."""
    with conn() as c:
        rows = c.execute(
            "SELECT status, COUNT(*) n FROM documents WHERE job_id=? GROUP BY status",
            (job_id,)).fetchall()
    return {r["status"]: r["n"] for r in rows}


def indexed_doc_ids(conversation_id: str) -> set[str]:
    """doc_ids already fully ingested in this scope — the batch resume gate."""
    with conn() as c:
        rows = c.execute(
            "SELECT id FROM documents WHERE conversation_id=? AND status='indexed'",
            (conversation_id,)).fetchall()
    return {r["id"] for r in rows}


def get_chunks_by_ids(chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    marks = ",".join("?" * len(chunk_ids))
    with conn() as c:
        rows = c.execute(
            f"SELECT k.*, d.file_name, d.source_modified_at, d.created_at AS doc_created_at "
            f"FROM doc_chunks k JOIN documents d ON d.id=k.doc_id "
            f"WHERE k.chunk_id IN ({marks})",
            chunk_ids).fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        try:
            d["section_path"] = json.loads(d["section_path"] or "[]")
        except json.JSONDecodeError:
            d["section_path"] = []
        out[d["chunk_id"]] = d
    return out


def corpus_stats(conversation_id: str) -> dict:
    with conn() as c:
        # `degraded` counts: embedding failed but chunks exist, so the document
        # is still retrievable through the lexical arm. Excluding it would show
        # "0 docs" on a thread the user can demonstrably search.
        docs = c.execute("SELECT COUNT(*) n FROM documents WHERE conversation_id=? "
                         "AND status IN ('indexed','degraded')",
                         (conversation_id,)).fetchone()["n"]
        chunks = c.execute("SELECT COUNT(*) n FROM doc_chunks WHERE conversation_id=?",
                           (conversation_id,)).fetchone()["n"]
        vecs = c.execute("SELECT COUNT(*) n FROM chunk_vectors WHERE conversation_id=?",
                         (conversation_id,)).fetchone()["n"]
    return {"documents": docs, "chunks": chunks, "vectors": vecs}


# ------------------------------------------------------------------ knowledge graph

def set_graph_status(collection_id: str, status: str, *,
                     built_at: str | None = None,
                     chunk_count_at_build: int | None = None,
                     stats: dict | None = None) -> None:
    """Update a collection's graph state. Only touches fields explicitly passed."""
    sets, params = ["graph_status=?"], [status]
    if built_at is not None:
        sets.append("graph_built_at=?")
        params.append(built_at)
    if chunk_count_at_build is not None:
        sets.append("graph_chunk_count_at_build=?")
        params.append(chunk_count_at_build)
    if stats is not None:
        sets.append("graph_stats_json=?")
        params.append(json.dumps(stats))
    params.append(collection_id)
    with conn() as c:
        c.execute(f"UPDATE conversations SET {','.join(sets)} WHERE id=?", params)


def get_graph_state(collection_id: str) -> dict:
    """Graph status plus a cheap, honest staleness check.

    Staleness is chunk-count comparison, not a content hash — cheap, and good
    enough: if the collection has grown or shrunk since the graph was built,
    the graph is out of date. It won't catch a document that was replaced
    with another of identical chunk count, but that is a rare enough edge
    case not to justify a hash walk on every collection-list render.
    """
    with conn() as c:
        row = c.execute(
            "SELECT graph_status, graph_built_at, graph_chunk_count_at_build, "
            "graph_stats_json FROM conversations WHERE id=?",
            (collection_id,)).fetchone()
    if not row:
        return {"status": "none", "built_at": None, "stale": False, "stats": {}}

    status = row["graph_status"] or "none"
    stats = json.loads(row["graph_stats_json"] or "{}")
    stale = False
    if status == "ready":
        current_chunks = corpus_stats(collection_id)["chunks"]
        built_chunks = row["graph_chunk_count_at_build"]
        if built_chunks is not None and current_chunks != built_chunks:
            stale = True
            status = "stale"

    return {"status": status, "built_at": row["graph_built_at"],
           "stale": stale, "stats": stats}
