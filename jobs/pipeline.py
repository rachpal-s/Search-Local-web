"""jobs/pipeline.py — batch ingestion across many cores.

Why this is not just `asyncio.gather(ingest_document(...))`
-----------------------------------------------------------
docstore/ingest.py runs one document as a sequence of awaits, with the CPU-bound
stages parked on `asyncio.to_thread`. For an interactive upload that is exactly
right — one document, one user waiting, and threads keep the event loop
responsive.

For ten thousand documents it is wrong, because `to_thread` runs on the GIL.
PDF parsing, chunking and spaCy enrichment are all CPU-bound Python, so N
threads give you roughly one core's worth of throughput no matter how many
cores the box has.

So the batch path splits the work by where it actually blocks:

    ProcessPoolExecutor   extract -> classify -> chunk -> enrich
                          Real parallelism across cores. Each task is one
                          document and returns plain dicts, so nothing
                          unpicklable crosses the boundary.

    parent event loop     database writes and embedding
                          Embedding is an HTTP call, so it wants concurrency,
                          not cores. Writes stay in the parent because SQLite
                          with many writer processes is a lock-contention
                          problem nobody needs.

Vision extraction for images is an HTTP call that happens to sit inside the
CPU-bound half. It blocks a pool process rather than a core, which is wasteful
but bounded, and `skip_images` exists for corpora where those files are not
worth the wall-clock.

The stage LOGIC is unchanged and still imported from docstore.* — this module
schedules that code, it does not reimplement it.
"""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from docstore import filetypes


# ------------------------------------------------------------------ worker side
# Everything below runs in a POOL PROCESS. It must be importable at module level
# and must not touch the parent's event loop, database connections or sockets.

@dataclass
class DocResult:
    """What one pool process hands back. Plain data, all picklable."""
    doc_id: str
    file_name: str
    status: str                       # ok | quarantined | failed
    reason_code: str = ""
    text: str = ""
    extractor: str = ""
    quality: float = 1.0
    page_count: int | None = None
    warnings: list[str] | None = None
    classification: dict | None = None
    chunks: list[dict] | None = None
    elapsed_ms: int = 0


def _process_document(args: dict) -> DocResult:
    """extract -> classify -> chunk -> enrich for one file, in a pool process."""
    t0 = time.perf_counter()
    path = Path(args["path"])
    doc_id = args["doc_id"]
    file_name = args["file_name"]

    from docstore import ingest
    from docstore.extractors import ExtractError, select

    def fail(status: str, code: str) -> DocResult:
        return DocResult(doc_id=doc_id, file_name=file_name, status=status,
                         reason_code=code,
                         elapsed_ms=int((time.perf_counter() - t0) * 1000))

    # ---- I1 extract ----
    try:
        extractor = select(path)
        res = extractor.extract(path)
    except ExtractError as e:
        code = getattr(e, "reason_code", "extract_error")
        # `deferred_tabular` is a designed outcome, not a bug — spreadsheets need
        # a row-oriented chunker. Quarantine keeps the corpus honest about it,
        # matching how the upload path reports the same case.
        return fail("quarantined" if str(code).startswith("deferred") else "failed",
                    str(code))
    except Exception as e:  # noqa: BLE001
        return fail("failed", f"extract_error:{type(e).__name__}")

    if not (res.text or "").strip():
        return fail("failed", "empty_extraction")

    # ---- I2 classify ----
    cls = ingest.classify(res.text, file_name)
    if args.get("classification_hint"):
        cls["data_classification"] = ingest._escalate(
            cls["data_classification"], args["classification_hint"])

    # ---- I3 chunk ----
    try:
        from docstore.code_chunker import chunk_code, is_code_extraction
        meta = {**cls, "file_name": file_name}
        if is_code_extraction(res.extractor):
            records = chunk_code(res.text, doc_id, meta)
        else:
            from docstore.chunker import ChunkerConfig, chunk
            from config import get_settings
            c = get_settings()
            records = chunk(res.text, doc_id, meta, ChunkerConfig(
                target_chars=int(getattr(c, "chunk_target_chars", 1200)),
                max_chars=int(getattr(c, "chunk_max_chars", 2400)),
                overlap_chars=int(getattr(c, "chunk_overlap_chars", 150)),
                min_chars=int(getattr(c, "chunk_min_chars", 80))))
    except Exception as e:  # noqa: BLE001
        return fail("failed", f"chunk_error:{type(e).__name__}")

    if not records:
        return fail("failed", "no_chunks")

    # ---- I4 enrich (fail open, same contract as the upload path) ----
    try:
        records = ingest.enrich_chunks(records)
    except Exception:  # noqa: BLE001
        pass

    return DocResult(
        doc_id=doc_id, file_name=file_name, status="ok",
        reason_code="success", text=res.text,
        extractor=f"{res.extractor}@{res.extractor_version}",
        quality=res.quality, page_count=res.page_count,
        warnings=list(res.warnings or []), classification=cls,
        chunks=[r.to_dict() for r in records],
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


# ------------------------------------------------------------------ parent side

def auto_workers(requested: int = 0) -> int:
    """Cores minus 25% headroom, so ingestion never starves the API process."""
    if requested and requested > 0:
        return requested
    cores = os.cpu_count() or 2
    return max(1, cores - max(1, int(cores * 0.25)))


def discover_files(folders: Iterable, skip_images: bool = False,
                   max_files: int = 0) -> tuple[list[tuple[Path, Any]], int]:
    """Walk the configured roots and return (path, folder_spec) for every
    ingestible file, plus how many were dropped by an exclude pattern.

    Names only — nothing is opened here. The point is to answer "how much work
    is this?" in seconds rather than after the first hour, and to let the
    dashboard show a real denominator for the progress bar.
    """
    out: list[tuple[Path, Any]] = []
    seen: set[str] = set()
    excluded = 0

    for spec in folders:
        root = Path(spec.path)
        if not root.exists():
            continue
        allow = set(spec.include) if spec.include else None
        walker = root.rglob("*") if spec.recursive else root.glob("*")

        for p in sorted(walker):
            if not p.is_file():
                continue
            suffix = p.suffix.lower()
            # Archives were expanded in the previous phase; the .zip itself is
            # not a document and must not reach an extractor.
            if suffix in filetypes.ARCHIVE_SUFFIXES:
                continue
            if not filetypes.is_supported(suffix):
                continue
            if allow and suffix not in allow:
                continue
            if skip_images and suffix in filetypes.IMAGE_SUFFIXES:
                continue
            if filetypes.is_excluded(p, root, spec.exclude):
                excluded += 1
                continue

            # normcase folds case on Windows so overlapping-root dedup is not
            # fooled by two differently-cased references to the same file —
            # the same gap that made _within() misfire in routers/jobs.py.
            key = os.path.normcase(str(p.resolve()))
            if key in seen:          # overlapping roots must not double-ingest
                continue
            seen.add(key)
            out.append((p, spec))

            if max_files and len(out) >= max_files:
                return out, excluded
    return out, excluded


async def run_batch(
    scope_id: str,
    work: list[dict],
    *,
    workers: int = 0,
    embed_concurrency: int = 4,
    on_result: Callable[[DocResult], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Ingest a work list into one scope. Returns a status tally.

    Results are consumed as they complete rather than gathered at the end, so
    the dashboard shows progress from the first document and a cancel takes
    effect within one document rather than at the end of the run.
    """
    from docstore import ingest, store

    workers = auto_workers(workers)
    tally: dict[str, int] = {}
    loop = asyncio.get_running_loop()
    embed_sem = asyncio.Semaphore(max(1, embed_concurrency))

    def bump(key: str) -> None:
        tally[key] = tally.get(key, 0) + 1

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            loop.run_in_executor(pool, _process_document, item): item
            for item in work
        }
        if not pending:
            return tally

        for future in asyncio.as_completed(list(pending)):
            if should_cancel and should_cancel():
                for f in pending:
                    f.cancel()
                bump("cancelled")
                break

            try:
                result: DocResult = await future
            except Exception as e:  # noqa: BLE001 — one bad doc must not end the run
                bump("failed")
                print(f"[batch] worker crashed: {type(e).__name__}: {e}")
                continue

            if result.status != "ok":
                store.set_document_status(result.doc_id, result.status,
                                          reason_code=result.reason_code)
                bump(result.status)
                if on_result:
                    on_result(result)
                continue

            # ---- persist chunks (parent process owns the database) ----
            cls = result.classification or {}
            store.set_document_status(
                result.doc_id, "embedding",
                extractor=result.extractor, quality=result.quality,
                page_count=result.page_count, warnings=result.warnings or [],
                data_classification=cls.get("data_classification", "internal"),
                allowed_principals=cls.get("allowed_principals", ["*"]),
                pii_tags=cls.get("pii_tags", []),
                chunk_count=len(result.chunks or []))
            store.save_chunks(scope_id, result.chunks or [])

            # ---- I5 embed: network-bound, so concurrency not cores ----
            missing = store.chunks_missing_vectors(result.doc_id)
            if missing:
                try:
                    async with embed_sem:
                        model = ingest.embed_model_name()
                        batch = 16
                        for i in range(0, len(missing), batch):
                            window = missing[i:i + batch]
                            vecs = await ingest.embed_texts([t for _, t in window])
                            store.save_vectors(
                                scope_id, model,
                                list(zip([c for c, _ in window], vecs)))
                except Exception as e:  # noqa: BLE001
                    # Chunks are already saved, so the lexical arm still finds
                    # this document. `degraded` says so honestly instead of
                    # `failed`, which would imply nothing was indexed.
                    store.set_document_status(
                        result.doc_id, "degraded",
                        reason_code=f"embed_error:{type(e).__name__}")
                    bump("degraded")
                    if on_result:
                        on_result(result)
                    continue

            store.set_document_status(result.doc_id, "indexed",
                                      chunk_count=len(result.chunks or []))
            bump("indexed")
            if on_result:
                on_result(result)

    return tally
