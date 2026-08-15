"""routers/uploads.py — multi-file upload + ingestion status.

Contract with the UI:
    POST /api/uploads              accept files, return rows at status=pending,
                                   ingest in the background
    GET  /api/uploads/{conv_id}    poll status for every doc in the thread
    DELETE /api/uploads/doc/{id}   drop one document and its chunks

The endpoint returns before ingestion finishes on purpose. A 40-page scanned PDF
can take a minute of vision calls; blocking the response on it would mean the
user stares at a spinner with no idea which of their five files is the slow one.
Returning immediately with per-file rows lets the composer show real progress and
lets the user start typing while the corpus builds behind them.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import get_settings
from docstore import archive, filetypes, ingest, store

cfg = get_settings()
router = APIRouter(prefix="/api/uploads", tags=["uploads"])

# Serialised behind a semaphore rather than a queue: uploads are bursty and
# short-lived, and the real bottleneck downstream is the embed endpoint, which
# already has its own concurrency cap.
_ingest_sem: asyncio.Semaphore | None = None


def _sem() -> asyncio.Semaphore:
    global _ingest_sem
    if _ingest_sem is None:
        _ingest_sem = asyncio.Semaphore(int(getattr(cfg, "ingest_concurrency", 2)))
    return _ingest_sem


async def _ingest_guarded(conversation_id: str, doc_id: str) -> None:
    async with _sem():
        try:
            await ingest.ingest_document(conversation_id, doc_id)
        except Exception as e:  # noqa: BLE001 — a background task must not vanish silently
            store.set_document_status(doc_id, "failed",
                                      reason_code=f"ingest_crash:{type(e).__name__}")
            print(f"[uploads] 🔥 ingest crashed for {doc_id}: {e}")


async def _expand_and_register(conversation_id: str, parent_row: dict,
                              accepted: list, rejected: list) -> int:
    """Fan one uploaded archive into N child documents.

    register_upload() is one upload -> one document row, which is right for
    every type except an archive. Rather than widen that contract, the archive
    keeps its own row as a container (status `expanding`, never ingested) and
    its children are registered as ordinary documents that reference it.

    Child display names carry the archive prefix — "specs.zip → design/api.md" —
    so an attachment chip never shows an orphaned api.md the user does not
    remember uploading.
    """
    parent_path = Path(parent_row["stored_path"])
    staging = parent_path.parent / f"{parent_path.stem}__expanded"

    store.set_document_status(parent_row["id"], "expanding")
    result = await asyncio.to_thread(archive.expand_file, parent_path, staging)

    for r in result.rejected:
        label = r["source_uri"].split(archive.PROVENANCE_SEP)[-1] \
            if archive.PROVENANCE_SEP in r["source_uri"] else parent_row["file_name"]
        rejected.append({"file_name": f"{parent_row['file_name']} → {label}",
                         "reason": archive.reason_text(r["reason_code"])})

    children = 0
    for staged_str, source_uri in sorted(result.provenance.items()):
        staged = Path(staged_str)
        if not staged.exists():
            continue
        relative = source_uri.split(archive.PROVENANCE_SEP)[-1]
        child = await ingest.register_file(
            conversation_id, staged,
            display_name=f"{parent_row['file_name']} → {relative}",
            source_uri=source_uri, parent_doc_id=parent_row["id"])
        children += 1
        accepted.append({
            "doc_id": child["id"], "file_name": child["file_name"],
            "size_bytes": child["size_bytes"], "media_type": child["media_type"],
            "status": child.get("status", "pending"),
            "duplicate": child.get("duplicate", False),
            "parent_doc_id": parent_row["id"],
        })
        if not child.get("duplicate"):
            asyncio.create_task(_ingest_guarded(conversation_id, child["id"]))

    # The container row is terminal either way — it is a manifest, not content.
    store.set_document_status(
        parent_row["id"],
        "indexed" if children else "quarantined",
        reason_code=None if children else "archive_empty",
        chunk_count=0)
    return children


@router.post("")
async def upload_files(conversation_id: str = Form(...),
                       files: list[UploadFile] = File(...)):
    """Accept N files, persist them, kick off ingestion, return immediately."""
    if not store.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    if not files:
        raise HTTPException(400, "No files supplied")

    max_files = int(getattr(cfg, "upload_max_files", 10))
    if len(files) > max_files:
        raise HTTPException(413, f"Up to {max_files} files per upload")

    default_mb = int(getattr(cfg, "upload_max_mb", 25))
    accepted, rejected = [], []

    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if not filetypes.is_supported(suffix):
            rejected.append({
                "file_name": f.filename,
                "reason": f"{suffix or 'Files with no extension'} is not a type "
                          f"this system can read."})
            continue

        raw = await f.read()
        # Per-type ceiling: archives get 10 MB because they expand, while a PDF
        # keeps the general limit.
        cap = filetypes.max_bytes_for(suffix, default_mb)
        if len(raw) > cap:
            rejected.append({
                "file_name": f.filename,
                "reason": f"Over the {cap // (1024 * 1024)} MB limit for "
                          f"{suffix} files."})
            continue
        if not raw:
            rejected.append({"file_name": f.filename, "reason": "The file is empty."})
            continue

        row = await ingest.register_upload(conversation_id, f.filename or "upload", raw)

        if archive.is_archive(row["file_name"]) and not row.get("duplicate"):
            n = await _expand_and_register(conversation_id, row, accepted, rejected)
            accepted.append({
                "doc_id": row["id"], "file_name": row["file_name"],
                "size_bytes": row["size_bytes"], "media_type": row["media_type"],
                "status": "indexed" if n else "quarantined",
                "duplicate": False, "child_count": n})
            continue

        accepted.append({
            "doc_id": row["id"], "file_name": row["file_name"],
            "size_bytes": row["size_bytes"], "media_type": row["media_type"],
            "status": row.get("status", "pending"),
            "duplicate": row.get("duplicate", False),
        })
        if not row.get("duplicate"):
            asyncio.create_task(_ingest_guarded(conversation_id, row["id"]))

    if not accepted and rejected:
        raise HTTPException(415, {"rejected": rejected})

    return {"accepted": accepted, "rejected": rejected,
            "corpus": store.corpus_stats(conversation_id)}


@router.get("/{conversation_id}")
async def upload_status(conversation_id: str):
    """Poll target for the composer's attachment chips."""
    if not store.get_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    docs = store.list_documents(conversation_id)
    # `expanding` is deliberately absent: an archive is unpacking and its
    # children have not been registered yet, so declaring the batch settled here
    # would stop chat.js polling before the child chips ever appear.
    settled = {"indexed", "failed", "quarantined", "degraded"}
    return {
        "documents": docs,
        "corpus": store.corpus_stats(conversation_id),
        "all_settled": all(d["status"] in settled for d in docs) if docs else True,
    }


@router.delete("/doc/{doc_id}")
async def delete_doc(doc_id: str):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    store.delete_document(doc_id)
    if doc.get("stored_path"):
        p = Path(doc["stored_path"])
        for candidate in (p, p.with_suffix(p.suffix + ".txt")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
    return {"status": "deleted", "doc_id": doc_id}
