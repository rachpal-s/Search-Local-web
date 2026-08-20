"""routers/jobs.py — the Jobs dashboard: page, REST surface, live SSE feed.

The API never runs a pipeline phase. It writes a job row and returns; the worker
process picks it up. That separation is what keeps a four-hour ingestion from
competing with chat requests for CPU in the same process.

Progress reaches the browser over SSE by polling the job tables once a second
and emitting only what changed. Polling a local SQLite file in WAL mode is
cheaper than the coordination a push-based design would need between two
processes, and it recovers automatically if the worker restarts mid-job.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from config import get_settings
from docstore import collections as coll
from docstore import filetypes, store
from jobs import store as jobstore
from jobs.models import FolderSet, FolderSpec, JobOptions

cfg = get_settings()
router = APIRouter(tags=["jobs"])
templates = Jinja2Templates(directory="templates")


def _allowed_roots() -> list[Path]:
    raw = getattr(cfg, "ingest_allowed_roots", "") or "data/incoming"
    return [Path(p.strip()).expanduser().resolve()
            for p in raw.split(os.pathsep) if p.strip()]


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        pass
    # Path.relative_to() compares raw path segments and is case-SENSITIVE even
    # on Windows, where the filesystem itself is not. A folder picked from
    # Explorer carries its on-disk casing (e.g. "IN010M80866"); a hand-typed
    # INGEST_ALLOWED_ROOTS entry easily ends up differently cased. Those are
    # the same directory on disk and relative_to() wrongly calls it "outside".
    # os.path.normcase folds case on Windows and is a no-op on POSIX, so this
    # fixes the Windows false-negative without loosening anything on Linux/Mac.
    c, p = os.path.normcase(str(child)), os.path.normcase(str(parent))
    return c == p or c.startswith(p + os.sep)


def _validate_folder(spec: FolderSpec) -> FolderSpec:
    """Resolve and confine a folder path.

    Without this the folder textbox on the job form is an arbitrary-file-read
    endpoint: "/etc" is a valid path and the pipeline would happily index it
    into a collection any chat can attach.
    """
    try:
        resolved = Path(spec.path).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        raise HTTPException(400, f"Cannot resolve that path: {spec.path} ({e})")

    roots = _allowed_roots()
    if not any(_within(resolved, r) for r in roots):
        raise HTTPException(
            403,
            f"{resolved} is outside the permitted ingestion roots "
            f"({', '.join(str(r) for r in roots)}). "
            f"Widen INGEST_ALLOWED_ROOTS to allow it.")

    for suffix in spec.include:
        if not filetypes.is_supported(suffix):
            raise HTTPException(400, f"{suffix} is not an ingestible file type.")

    return FolderSpec(path=str(resolved), recursive=spec.recursive,
                      include=spec.include, exclude=spec.exclude,
                      classification_hint=spec.classification_hint)


def _specs(raw: list[dict]) -> list[FolderSpec]:
    return [_validate_folder(FolderSpec(**f)) for f in raw]


# ------------------------------------------------------------------ page

@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html", {
        "stats": jobstore.stats(),
        "collections": coll.list_collections(),
        "folder_sets": [fs.to_dict() for fs in jobstore.list_folder_sets()],
        "filetypes": filetypes.as_catalog(),
        "allowed_roots": [str(r) for r in _allowed_roots()],
    })


# ------------------------------------------------------------------ jobs

@router.get("/api/jobs")
async def list_jobs(limit: int = Query(25, ge=1, le=100),
                    offset: int = Query(0, ge=0),
                    status: str | None = None):
    jobs = jobstore.list_jobs(limit=limit, offset=offset, status=status)
    return {"jobs": [j.to_dict() for j in jobs],
            "total": jobstore.count_jobs(status),
            "limit": limit, "offset": offset}


@router.post("/api/jobs", status_code=201)
async def create_job(payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Give the job a name so you can find it later.")

    folders_raw = payload.get("folders") or []
    set_id = payload.get("set_id") or None
    if set_id and not folders_raw:
        fs = jobstore.get_folder_set(set_id)
        if not fs:
            raise HTTPException(404, "That folder set no longer exists.")
        folders_raw = [f.to_dict() for f in fs.folders]
    if not folders_raw:
        raise HTTPException(400, "Add at least one folder, or pick a folder set.")

    folders = _specs(folders_raw)

    # Target collection: an existing one, or a new one named by the operator.
    collection_id = payload.get("collection_id") or ""
    if collection_id:
        if not coll.get_collection(collection_id):
            raise HTTPException(404, "That collection no longer exists.")
    else:
        new_name = (payload.get("collection_name") or name).strip()
        collection_id = coll.ensure_collection(
            new_name, payload.get("collection_description", ""))["id"]

    try:
        options = JobOptions(**(payload.get("options") or {}))
    except TypeError as e:
        raise HTTPException(400, f"Unrecognised option: {e}")

    job = jobstore.create_job(name=name, collection_id=collection_id,
                              folders=folders, options=options,
                              set_id=set_id, created_by="ui")
    return job.to_dict()


@router.get("/api/jobs/stats")
async def job_stats():
    return jobstore.stats()


@router.get("/api/jobs/workers")
async def worker_status():
    """Whether anything is actually watching the queue.

    Distinguishes 'no worker running — queued jobs won't move' from 'a worker
    is up but idle', which look identical if you only ever look at job rows.
    """
    workers = jobstore.active_workers()
    return {"active": len(workers), "workers": workers}


@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobstore.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job.to_dict()


@router.get("/api/jobs/{job_id}/documents")
async def job_documents(job_id: str):
    if not jobstore.get_job(job_id):
        raise HTTPException(404, "Job not found.")
    return {"documents": store.documents_by_job(job_id),
            "tally": store.job_document_tally(job_id)}


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if not jobstore.get_job(job_id):
        raise HTTPException(404, "Job not found.")
    if not jobstore.request_cancel(job_id):
        raise HTTPException(409, "This job has already finished.")
    jobstore.add_event(job_id, "Cancel requested.", "warn")
    return {"status": "cancelling", "job_id": job_id}


@router.post("/api/jobs/{job_id}/rerun", status_code=201)
async def rerun_job(job_id: str, force: bool = False):
    """Queue the same work again.

    For an ingest job: without `force` this is cheap — every document already
    indexed is skipped by content hash, so a re-run picks up only what is new
    or changed. With `force` the whole collection is rebuilt.

    For a graph_build job: `force` has no effect — a graph rebuild always
    reprocesses everything currently in the collection, since it is a
    from-scratch pass over the collection's current chunks, not an
    incremental one. Re-running it is exactly the same operation whether or
    not `force` was set.
    """
    old = jobstore.get_job(job_id)
    if not old:
        raise HTTPException(404, "Job not found.")

    if old.kind.value == "graph_build":
        job = jobstore.create_graph_job(f"{old.name} (rebuild)", old.collection_id)
        return job.to_dict()

    options = JobOptions(**{**old.options.to_dict(), "force": force})
    job = jobstore.create_job(
        name=f"{old.name} (re-run)", collection_id=old.collection_id,
        folders=old.folders, options=options, set_id=old.set_id, created_by="ui")
    return job.to_dict()


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, after: int = 0, limit: int = Query(300, le=1000)):
    if not jobstore.get_job(job_id):
        raise HTTPException(404, "Job not found.")
    events = jobstore.get_events(job_id, after_id=after, limit=limit)
    return {"events": [e.to_dict() for e in events],
            "last_id": events[-1].event_id if events else after}


@router.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    if not jobstore.get_job(job_id):
        raise HTTPException(404, "Job not found.")

    async def gen():
        last_event_id = 0
        last_snapshot = None
        ticks = 0

        while True:
            job = jobstore.get_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'type': 'gone'})}\n\n"
                return

            snapshot = job.to_dict()
            if snapshot != last_snapshot:
                yield f"data: {json.dumps({'type': 'job', 'job': snapshot})}\n\n"
                last_snapshot = snapshot

            events = jobstore.get_events(job_id, after_id=last_event_id, limit=200)
            if events:
                last_event_id = events[-1].event_id
                payload = {"type": "events", "events": [e.to_dict() for e in events]}
                yield f"data: {json.dumps(payload)}\n\n"

            if job.status.terminal:
                yield f"data: {json.dumps({'type': 'done', 'status': job.status.value})}\n\n"
                return

            ticks += 1
            if ticks % 20 == 0:      # keep proxies from reaping an idle stream
                yield ": keep-alive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------ collections

@router.get("/api/collections/search")
async def search_collections(q: str, top_k: int = 3):
    """Find which collection(s) have something relevant, before attaching any.

    This is the fix for 'I don't know which corpus has my data' — search
    everywhere first, see counts and snippets, attach only what's actually
    relevant. Read-only: nothing here changes any chat's retrieval scope.
    """
    q = q.strip()
    if not q:
        raise HTTPException(400, "Give something to search for.")
    results = await coll.search_all_collections(q, top_k_per_collection=top_k)
    return {"query": q, "results": results}


@router.post("/api/collections/{collection_id}/graph/build", status_code=201)
async def build_collection_graph(collection_id: str):
    """Queue a graph_build job for this collection.

    A separate job KIND, not a flag on ingestion — building a graph is
    something an operator chooses to do on a collection that already exists,
    at a moment of their choosing, never a side effect of a folder-ingestion
    run that would make it slower without being asked.
    """
    c = coll.get_collection(collection_id)
    if not c:
        raise HTTPException(404, "Collection not found.")
    if c["chunks"] == 0:
        raise HTTPException(400, "This collection has no chunks yet — ingest "
                             "documents into it before building a graph.")
    if c["graph"]["status"] == "building":
        raise HTTPException(409, "A graph build is already running for this collection.")

    job = jobstore.create_graph_job(f"{c['name']} — knowledge graph", collection_id)
    return job.to_dict()


@router.get("/api/collections/{collection_id}/graph")
async def collection_graph_status(collection_id: str):
    c = coll.get_collection(collection_id)
    if not c:
        raise HTTPException(404, "Collection not found.")
    return c["graph"]


@router.get("/api/collections")
async def list_collections():
    return {"collections": coll.list_collections()}


@router.post("/api/collections", status_code=201)
async def create_collection(payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "A collection needs a name.")
    return coll.create_collection(name, payload.get("description", ""))


@router.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str):
    """Deletes the collection and everything ingested into it."""
    if not coll.delete_collection(collection_id):
        raise HTTPException(404, "Collection not found.")
    return {"status": "deleted", "collection_id": collection_id}


@router.get("/api/conversations/{conversation_id}/collections")
async def conversation_collections(conversation_id: str):
    return {"attached": coll.attached(conversation_id),
            "available": coll.list_collections()}


@router.post("/api/conversations/{conversation_id}/collections/{collection_id}")
async def attach_collection(conversation_id: str, collection_id: str):
    if not store.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found.")
    if not coll.attach(conversation_id, collection_id):
        raise HTTPException(404, "Collection not found.")
    return {"status": "attached", "attached": coll.attached(conversation_id)}


@router.delete("/api/conversations/{conversation_id}/collections/{collection_id}")
async def detach_collection(conversation_id: str, collection_id: str):
    coll.detach(conversation_id, collection_id)
    return {"status": "detached", "attached": coll.attached(conversation_id)}


# ------------------------------------------------------------------ folder sets

@router.get("/api/folder-sets")
async def list_folder_sets():
    return {"sets": [fs.to_dict() for fs in jobstore.list_folder_sets()]}


@router.post("/api/folder-sets", status_code=201)
async def save_folder_set(payload: dict = Body(...)):
    folders_raw = payload.get("folders") or []
    if not folders_raw:
        raise HTTPException(400, "A folder set needs at least one folder.")
    fs = FolderSet(set_id=payload.get("set_id") or "",
                   name=(payload.get("name") or "Untitled").strip(),
                   description=payload.get("description", ""),
                   folders=_specs(folders_raw))
    return jobstore.save_folder_set(fs).to_dict()


@router.delete("/api/folder-sets/{set_id}")
async def delete_folder_set(set_id: str):
    if not jobstore.delete_folder_set(set_id):
        raise HTTPException(404, "Folder set not found.")
    return {"status": "deleted", "set_id": set_id}


# ------------------------------------------------------------------ helpers

@router.get("/api/folders/preview")
async def preview_folder(path: str, include: str = "", exclude: str = ""):
    """Count what a folder would ingest, before committing to a run.

    Walks names only and never opens a file, so it answers in seconds. Respects
    exclude patterns — this is the check that lets an operator confirm "f2 and
    f3 are actually blocked" before starting a job, rather than discovering it
    was wrong after an hour of ingestion.
    """
    spec = _validate_folder(FolderSpec(
        path=path,
        include=[s for s in include.split(",") if s.strip()],
        exclude=[s for s in exclude.split(",") if s.strip()]))
    root = Path(spec.path)
    if not root.exists():
        raise HTTPException(404, f"{spec.path} does not exist.")
    if not root.is_dir():
        raise HTTPException(400, f"{spec.path} is a file, not a folder.")

    allow = set(spec.include) if spec.include else None
    by_ext: dict[str, int] = {}
    seen = ingestible = archives = excluded_count = 0
    total_bytes = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        seen += 1
        ext = p.suffix.lower()
        if allow and ext not in allow:
            continue
        if not filetypes.is_supported(ext):
            continue
        if filetypes.is_excluded(p, root, spec.exclude):
            excluded_count += 1
            continue
        ingestible += 1
        by_ext[ext] = by_ext.get(ext, 0) + 1
        if ext in filetypes.ARCHIVE_SUFFIXES:
            archives += 1
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass

    return {"path": spec.path, "files_seen": seen, "files_ingestible": ingestible,
            "files_excluded": excluded_count, "archives": archives,
            "total_bytes": total_bytes,
            "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1]))}


@router.get("/api/filetypes")
async def get_filetypes():
    """One allowlist, served to both chat.js and jobs.js."""
    catalog = filetypes.as_catalog()
    catalog["upload_max_mb"] = int(getattr(cfg, "upload_max_mb", 25))
    catalog["upload_max_files"] = int(getattr(cfg, "upload_max_files", 10))
    return catalog
