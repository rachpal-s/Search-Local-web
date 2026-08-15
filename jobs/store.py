"""jobs/store.py — job queue persistence.

Lives in the same SQLite file as everything else (cfg.chat_db_path), reusing
docstore.store.conn(). One database means a job row and the documents it
produced are consistent with each other without a distributed transaction, and
WAL mode already lets the API read while the worker writes.

Claiming a job is a single conditional UPDATE, not SELECT-then-UPDATE, so two
workers racing cannot both win — the loser's UPDATE matches zero rows and it
moves on. That is the whole concurrency design, and it is sufficient because
job starts are rare events, not a hot path.

A worker killed with SIGKILL leaves a job stuck in `running` forever, so
`reclaim_stale()` returns jobs whose heartbeat has gone quiet back to `queued`.
Without it an operator has to unstick the queue by editing the database.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from docstore.store import _now, conn, new_id
from jobs.models import (
    PHASES,
    FolderSet,
    FolderSpec,
    Job,
    JobEvent,
    JobOptions,
    JobStatus,
    Phase,
    PhaseRun,
)

LEASE_SECONDS = 180

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_folder_sets (
    set_id        TEXT PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL DEFAULT '',
    folders_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id           TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'queued',
    collection_id    TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    set_id           TEXT,
    folders_json     TEXT NOT NULL,
    options_json     TEXT NOT NULL,
    created_by       TEXT NOT NULL DEFAULT 'system',
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    worker_id        TEXT,
    heartbeat_at     TEXT,
    current_phase    TEXT,
    progress_pct     REAL NOT NULL DEFAULT 0.0,
    files_discovered INTEGER NOT NULL DEFAULT 0,
    docs_ok          INTEGER NOT NULL DEFAULT 0,
    docs_failed      INTEGER NOT NULL DEFAULT 0,
    docs_quarantined INTEGER NOT NULL DEFAULT 0,
    docs_skipped     INTEGER NOT NULL DEFAULT 0,
    chunks_total     INTEGER NOT NULL DEFAULT 0,
    error            TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_status  ON jobs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS job_phases (
    job_id      TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    phase       TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    started_at  TEXT,
    finished_at TEXT,
    items_done  INTEGER NOT NULL DEFAULT 0,
    items_total INTEGER NOT NULL DEFAULT 0,
    tally_json  TEXT NOT NULL DEFAULT '{}',
    error       TEXT,
    PRIMARY KEY (job_id, phase)
);

CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    ts       TEXT NOT NULL,
    level    TEXT NOT NULL DEFAULT 'info',
    phase    TEXT,
    message  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_job_events ON job_events(job_id, event_id);

-- A worker only writes to `jobs` while it holds one, so an IDLE worker was
-- previously invisible: the dashboard had no way to distinguish "queue is
-- empty" from "nothing is watching the queue at all". This table exists
-- purely so the worker daemon can say "I'm alive" independent of whether it
-- currently has work, via a heartbeat separate from any job.
CREATE TABLE IF NOT EXISTS workers (
    worker_id  TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    pid        INTEGER,
    hostname   TEXT
);
"""


def init_db() -> None:
    """Create the job tables. Idempotent; called from the app startup hook."""
    with conn() as c:
        c.executescript(_SCHEMA)


# ------------------------------------------------------------------ folder sets

def save_folder_set(fs: FolderSet) -> FolderSet:
    fs.updated_at = _now()
    fs.set_id = fs.set_id or f"set_{uuid.uuid4().hex[:10]}"
    with conn() as c:
        c.execute(
            "INSERT INTO job_folder_sets (set_id,name,description,folders_json,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(set_id) DO UPDATE SET name=excluded.name, "
            "description=excluded.description, folders_json=excluded.folders_json, "
            "updated_at=excluded.updated_at",
            (fs.set_id, fs.name, fs.description,
             json.dumps([f.to_dict() for f in fs.folders]),
             fs.created_at, fs.updated_at))
    return fs


def list_folder_sets() -> list[FolderSet]:
    with conn() as c:
        rows = c.execute("SELECT * FROM job_folder_sets ORDER BY name").fetchall()
    return [_row_to_set(r) for r in rows]


def get_folder_set(set_id: str) -> FolderSet | None:
    with conn() as c:
        r = c.execute("SELECT * FROM job_folder_sets WHERE set_id=?", (set_id,)).fetchone()
    return _row_to_set(r) if r else None


def delete_folder_set(set_id: str) -> bool:
    with conn() as c:
        return c.execute("DELETE FROM job_folder_sets WHERE set_id=?",
                         (set_id,)).rowcount > 0


def _row_to_set(row) -> FolderSet:
    return FolderSet(
        set_id=row["set_id"], name=row["name"], description=row["description"],
        folders=[FolderSpec(**f) for f in json.loads(row["folders_json"])],
        created_at=row["created_at"], updated_at=row["updated_at"])


# ------------------------------------------------------------------ jobs

def create_job(name: str, collection_id: str, folders: list[FolderSpec],
               options: JobOptions | None = None, set_id: str | None = None,
               created_by: str = "ui") -> Job:
    if not folders:
        raise ValueError("A job needs at least one folder.")
    job = Job(
        job_id=f"job_{uuid.uuid4().hex[:12]}", name=name,
        collection_id=collection_id, set_id=set_id,
        folders=folders, options=options or JobOptions(), created_by=created_by,
    )
    job.phases = [PhaseRun(phase=p, seq=i) for i, p in enumerate(PHASES)]

    with conn() as c:
        c.execute(
            "INSERT INTO jobs (job_id,name,status,collection_id,set_id,folders_json,"
            "options_json,created_by,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (job.job_id, job.name, job.status.value, job.collection_id, job.set_id,
             json.dumps([f.to_dict() for f in job.folders]),
             json.dumps(job.options.to_dict()), job.created_by, job.created_at))
        c.executemany(
            "INSERT INTO job_phases (job_id,phase,seq,status) VALUES (?,?,?,?)",
            [(job.job_id, p.phase.value, p.seq, p.status) for p in job.phases])

    add_event(job.job_id, f"Queued with {len(folders)} folder(s).")
    return job


def get_job(job_id: str) -> Job | None:
    with conn() as c:
        r = c.execute(
            "SELECT j.*, c.title AS collection_name FROM jobs j "
            "LEFT JOIN conversations c ON c.id = j.collection_id WHERE j.job_id=?",
            (job_id,)).fetchone()
        if not r:
            return None
        phases = c.execute("SELECT * FROM job_phases WHERE job_id=? ORDER BY seq",
                           (job_id,)).fetchall()
    return _row_to_job(r, phases)


def list_jobs(limit: int = 25, offset: int = 0, status: str | None = None) -> list[Job]:
    sql = ("SELECT j.*, c.title AS collection_name FROM jobs j "
           "LEFT JOIN conversations c ON c.id = j.collection_id")
    params: list[Any] = []
    if status:
        sql += " WHERE j.status=?"
        params.append(status)
    sql += " ORDER BY j.created_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]

    with conn() as c:
        rows = c.execute(sql, params).fetchall()
        ids = [r["job_id"] for r in rows]
        by_job: dict[str, list] = {i: [] for i in ids}
        if ids:
            marks = ",".join("?" * len(ids))
            for p in c.execute(
                f"SELECT * FROM job_phases WHERE job_id IN ({marks}) ORDER BY seq", ids):
                by_job[p["job_id"]].append(p)
    return [_row_to_job(r, by_job[r["job_id"]]) for r in rows]


def count_jobs(status: str | None = None) -> int:
    with conn() as c:
        if status:
            return c.execute("SELECT COUNT(*) n FROM jobs WHERE status=?",
                             (status,)).fetchone()["n"]
        return c.execute("SELECT COUNT(*) n FROM jobs").fetchone()["n"]


def stats() -> dict:
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        agg = c.execute(
            "SELECT COALESCE(SUM(docs_ok),0) d, COALESCE(SUM(chunks_total),0) k "
            "FROM jobs WHERE status='succeeded'").fetchone()
        colls = c.execute(
            "SELECT COUNT(*) n FROM conversations WHERE kind='collection'").fetchone()["n"]
    by_status = {r["status"]: r["n"] for r in rows}
    return {"by_status": by_status, "total": sum(by_status.values()),
            "docs_ingested": agg["d"], "chunks_created": agg["k"],
            "collections": colls}


# ------------------------------------------------------------------ worker side

def claim_next(worker_id: str) -> Job | None:
    """Atomically take the oldest queued job. None when the queue is empty."""
    now = _now()
    with conn() as c:
        r = c.execute("SELECT job_id FROM jobs WHERE status='queued' "
                      "ORDER BY created_at LIMIT 1").fetchone()
        if not r:
            return None
        claimed = c.execute(
            "UPDATE jobs SET status='claimed', worker_id=?, heartbeat_at=?, started_at=? "
            "WHERE job_id=? AND status='queued'",
            (worker_id, now, now, r["job_id"])).rowcount
    return get_job(r["job_id"]) if claimed else None


def heartbeat(job_id: str) -> None:
    with conn() as c:
        c.execute("UPDATE jobs SET heartbeat_at=? WHERE job_id=?", (_now(), job_id))


def reclaim_stale(lease_seconds: int = LEASE_SECONDS) -> int:
    """Requeue jobs whose worker stopped reporting. Run at worker startup."""
    with conn() as c:
        return c.execute(
            f"UPDATE jobs SET status='queued', worker_id=NULL, heartbeat_at=NULL, "
            f"current_phase=NULL, error='Requeued: worker lease expired.' "
            f"WHERE status IN ('claimed','running') AND (heartbeat_at IS NULL OR "
            f"heartbeat_at < datetime('now','-{int(lease_seconds)} seconds'))").rowcount


def set_status(job_id: str, status: JobStatus, error: str | None = None) -> None:
    sets, params = ["status=?"], [status.value]
    if status.terminal:
        sets.append("finished_at=?")
        params.append(_now())
    if error is not None:
        sets.append("error=?")
        params.append(error)
    params.append(job_id)
    with conn() as c:
        c.execute(f"UPDATE jobs SET {','.join(sets)} WHERE job_id=?", params)


def is_cancel_requested(job_id: str) -> bool:
    with conn() as c:
        r = c.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return bool(r) and r["status"] in ("cancelling", "cancelled")


def request_cancel(job_id: str) -> bool:
    """Queued jobs cancel at once; running ones are asked to stop at the next
    document boundary, so partial work stays consistent."""
    with conn() as c:
        r = c.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not r:
            return False
        if r["status"] == "queued":
            c.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?",
                      (_now(), job_id))
        elif r["status"] in ("claimed", "running"):
            c.execute("UPDATE jobs SET status='cancelling' WHERE job_id=?", (job_id,))
        else:
            return False
    return True


_PROGRESS_FIELDS = {"current_phase", "progress_pct", "files_discovered", "docs_ok",
                    "docs_failed", "docs_quarantined", "docs_skipped", "chunks_total"}


def update_progress(job_id: str, **fields) -> None:
    sets, params = [], []
    for k, v in fields.items():
        if k in _PROGRESS_FIELDS and v is not None:
            sets.append(f"{k}=?")
            params.append(v.value if hasattr(v, "value") else v)
    if not sets:
        return
    params.append(job_id)
    with conn() as c:
        c.execute(f"UPDATE jobs SET {','.join(sets)} WHERE job_id=?", params)


def update_phase(job_id: str, phase: Phase, **fields) -> None:
    allowed = {"status", "started_at", "finished_at", "items_done", "items_total", "error"}
    sets, params = [], []
    for k, v in fields.items():
        if k in allowed and v is not None:
            sets.append(f"{k}=?")
            params.append(v.value if hasattr(v, "value") else v)
    if "tally" in fields:
        sets.append("tally_json=?")
        params.append(json.dumps(fields["tally"]))
    if not sets:
        return
    params += [job_id, phase.value]
    with conn() as c:
        c.execute(f"UPDATE job_phases SET {','.join(sets)} WHERE job_id=? AND phase=?",
                  params)


# ------------------------------------------------------------------ events

def add_event(job_id: str, message: str, level: str = "info",
              phase: str | None = None) -> None:
    with conn() as c:
        c.execute("INSERT INTO job_events (job_id,ts,level,phase,message) "
                  "VALUES (?,?,?,?,?)", (job_id, _now(), level, phase, message[:2000]))


def get_events(job_id: str, after_id: int = 0, limit: int = 300) -> list[JobEvent]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM job_events WHERE job_id=? AND event_id > ? "
            "ORDER BY event_id LIMIT ?", (job_id, after_id, limit)).fetchall()
    return [JobEvent(event_id=r["event_id"], job_id=r["job_id"], ts=r["ts"],
                     level=r["level"], phase=r["phase"], message=r["message"])
            for r in rows]


def prune_events(job_id: str, keep: int = 2000) -> None:
    with conn() as c:
        c.execute(
            "DELETE FROM job_events WHERE job_id=? AND event_id NOT IN "
            "(SELECT event_id FROM job_events WHERE job_id=? ORDER BY event_id DESC "
            "LIMIT ?)", (job_id, job_id, keep))


# ------------------------------------------------------------------ worker registry
# Separate from job claiming: a worker with an empty queue still calls
# register_worker() on a timer so the dashboard can tell "no worker running"
# apart from "worker running, nothing queued" — those need different operator
# reactions and previously looked identical from the frontend.

WORKER_STALE_SECONDS = 20   # a worker pings ~every 10s; miss two beats -> gone


def register_worker(worker_id: str, pid: int | None = None,
                    hostname: str | None = None) -> None:
    now = _now()
    with conn() as c:
        c.execute(
            "INSERT INTO workers (worker_id,started_at,last_seen,pid,hostname) "
            "VALUES (?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET last_seen=excluded.last_seen",
            (worker_id, now, now, pid, hostname))


def worker_ping(worker_id: str) -> None:
    with conn() as c:
        c.execute("UPDATE workers SET last_seen=? WHERE worker_id=?", (_now(), worker_id))


def deregister_worker(worker_id: str) -> None:
    """Called on clean shutdown so a stopped worker disappears immediately
    rather than waiting out the stale timeout."""
    with conn() as c:
        c.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))


def active_workers(stale_after: int = WORKER_STALE_SECONDS) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            f"SELECT * FROM workers WHERE last_seen > datetime('now','-{int(stale_after)} seconds') "
            f"ORDER BY started_at").fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ mapping

def _row_to_job(row, phase_rows) -> Job:
    return Job(
        job_id=row["job_id"], name=row["name"], status=JobStatus(row["status"]),
        collection_id=row["collection_id"],
        collection_name=(row["collection_name"] if "collection_name" in row.keys()
                         else "") or "",
        set_id=row["set_id"],
        folders=[FolderSpec(**f) for f in json.loads(row["folders_json"])],
        options=JobOptions(**json.loads(row["options_json"])),
        created_by=row["created_by"], created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
        worker_id=row["worker_id"], heartbeat_at=row["heartbeat_at"],
        current_phase=Phase(row["current_phase"]) if row["current_phase"] else None,
        progress_pct=row["progress_pct"], files_discovered=row["files_discovered"],
        docs_ok=row["docs_ok"], docs_failed=row["docs_failed"],
        docs_quarantined=row["docs_quarantined"], docs_skipped=row["docs_skipped"],
        chunks_total=row["chunks_total"], error=row["error"],
        phases=[PhaseRun(phase=Phase(p["phase"]), seq=p["seq"], status=p["status"],
                         started_at=p["started_at"], finished_at=p["finished_at"],
                         items_done=p["items_done"], items_total=p["items_total"],
                         tally=json.loads(p["tally_json"] or "{}"), error=p["error"])
                for p in phase_rows],
    )
