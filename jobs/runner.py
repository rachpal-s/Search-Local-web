"""jobs/runner.py — runs one job to completion.

Four phases, each writing its own progress so the dashboard shows where a long
run actually is:

    expand    unpack archives into a staging directory
    discover  walk the folders, register document rows, build the work list
    ingest    the parallel per-document pipeline
    verify    count what landed and report an honest tally

Cancellation is cooperative and checked at document boundaries. A hard kill mid
document would leave chunks saved with no vectors, which the retrieval layer
would happily serve as a partially-indexed document with no indication anything
was wrong. Stopping between documents costs a few seconds and keeps every row
in the database meaning what it says.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from config import get_settings
from docstore import archive, collections, store
from docstore import graph_store
from jobs import pipeline
from jobs import graph_pipeline
from jobs import store as jobstore
from jobs.models import (
    GRAPH_PHASE_WEIGHTS,
    PHASE_WEIGHTS,
    FolderSpec,
    Job,
    JobKind,
    JobStatus,
    Phase,
    utcnow,
)


class Cancelled(RuntimeError):
    pass


class JobRunner:
    """One instance per job. Never reused."""

    def __init__(self, job: Job):
        self.job = job
        self.scope_id = job.collection_id
        self.staging = Path("data/staging") / job.job_id
        self._done_weight = 0.0
        self._weights = (GRAPH_PHASE_WEIGHTS if job.kind == JobKind.graph_build
                         else PHASE_WEIGHTS)
        self._total_weight = sum(self._weights.values()) or 1.0
        self._last_beat = 0.0

    # ------------------------------------------------------------ helpers

    def log(self, message: str, level: str = "info", phase: Phase | None = None) -> None:
        jobstore.add_event(self.job.job_id, message, level,
                           phase.value if phase else None)

    def _beat(self) -> None:
        now = time.time()
        if now - self._last_beat > 15:
            jobstore.heartbeat(self.job.job_id)
            self._last_beat = now

    def _cancelled(self) -> bool:
        self._beat()
        return jobstore.is_cancel_requested(self.job.job_id)

    def _check_cancel(self) -> None:
        if self._cancelled():
            raise Cancelled("Cancel requested by operator.")

    def _set_progress(self, phase: Phase, fraction: float) -> None:
        pct = 100.0 * (self._done_weight + self._weights[phase] * fraction) / self._total_weight
        jobstore.update_progress(self.job.job_id,
                                 progress_pct=round(min(pct, 99.5), 1))

    # ------------------------------------------------------------ phases

    def _phase_expand(self) -> dict:
        """Unpack archives so their contents become ordinary files to walk."""
        tally = {"archives": 0, "files": 0, "rejected": 0}
        found_any = False

        for spec in list(self.job.folders):
            root = Path(spec.path)
            if not root.exists():
                self.log(f"Folder not found, skipping: {spec.path}",
                         "warn", Phase.expand)
                continue
            dest = self.staging / re.sub(r"[^\w.-]", "_", root.name)
            result = archive.expand_tree(root, dest)
            tally["archives"] += result.archives_seen
            tally["files"] += result.files_written
            tally["rejected"] += len(result.rejected)

            for r in result.rejected[:50]:
                self.log(f"{archive.reason_text(r['reason_code'])} — {r['source_uri']}",
                         "warn", Phase.expand)

            if result.files_written:
                found_any = True
                self.log(f"{root.name}: {result.archives_seen} archive(s) → "
                         f"{result.files_written} file(s)", phase=Phase.expand)

        if found_any:
            # Staged content becomes an extra root for the discovery walk. It is
            # appended to the in-memory job only — the persisted folder list stays
            # as the operator wrote it, so a re-run expands afresh rather than
            # walking last run's staging directory.
            self.job.folders.append(FolderSpec(path=str(self.staging)))
        elif tally["archives"] == 0:
            self.log("No archives found.", phase=Phase.expand)

        return tally

    async def _phase_discover(self) -> list[dict]:
        """Walk folders, register document rows, and build the work list."""
        opts = self.job.options
        found, excluded = pipeline.discover_files(self.job.folders,
                                                   skip_images=opts.skip_images,
                                                   max_files=opts.max_files)
        jobstore.update_progress(self.job.job_id, files_discovered=len(found))
        jobstore.update_phase(self.job.job_id, Phase.discover,
                              items_total=len(found))
        self.log(f"Found {len(found):,} ingestible file(s)."
                 + (f" {excluded:,} file(s) skipped by folder exclude rules."
                    if excluded else ""), phase=Phase.discover)

        if not found:
            self.log("Nothing to ingest. Check the folder paths, include filters, "
                     "and exclude rules.", "warn", Phase.discover)
            return []

        from docstore import ingest

        work: list[dict] = []
        skipped = 0
        for i, (path, spec) in enumerate(found, 1):
            self._check_cancel()
            try:
                row = await ingest.register_file(
                    self.scope_id, path,
                    source_uri=str(path.resolve()),
                    job_id=self.job.job_id,
                    classification_hint=spec.classification_hint)
            except OSError as e:
                self.log(f"Could not read {path.name}: {e}", "warn", Phase.discover)
                continue

            if row.get("duplicate") and not self.job.options.force:
                skipped += 1
            else:
                work.append({
                    "path": str(path),
                    "doc_id": row["id"],
                    "file_name": row["file_name"],
                    "classification_hint": spec.classification_hint,
                })

            if i % 100 == 0 or i == len(found):
                jobstore.update_phase(self.job.job_id, Phase.discover, items_done=i)
                jobstore.update_progress(self.job.job_id, docs_skipped=skipped)
                self._set_progress(Phase.discover, i / len(found))

        if skipped:
            self.log(f"{skipped:,} file(s) already indexed — skipping. "
                     f"Re-run with 'reprocess everything' to force them.",
                     phase=Phase.discover)
        self.log(f"{len(work):,} file(s) queued for ingestion.", phase=Phase.discover)
        return work

    async def _phase_ingest(self, work: list[dict]) -> dict:
        opts = self.job.options
        total = len(work)
        if not total:
            return {}

        jobstore.update_phase(self.job.job_id, Phase.ingest, items_total=total)
        workers = pipeline.auto_workers(opts.workers)
        self.log(f"Ingesting {total:,} document(s) across {workers} worker process(es).",
                 phase=Phase.ingest)

        counters = {"done": 0, "ok": 0, "failed": 0, "quarantined": 0}

        def on_result(result: pipeline.DocResult) -> None:
            counters["done"] += 1
            if result.status == "ok":
                counters["ok"] += 1
            elif result.status == "quarantined":
                counters["quarantined"] += 1
                self.log(f"{result.file_name}: {archive.reason_text(result.reason_code)}",
                         "warn", Phase.ingest)
            else:
                counters["failed"] += 1
                self.log(f"{result.file_name} failed: {result.reason_code}",
                         "error", Phase.ingest)

            n = counters["done"]
            if n % 10 == 0 or n == total:
                jobstore.update_phase(self.job.job_id, Phase.ingest, items_done=n)
                jobstore.update_progress(
                    self.job.job_id, docs_ok=counters["ok"],
                    docs_failed=counters["failed"],
                    docs_quarantined=counters["quarantined"])
                self._set_progress(Phase.ingest, n / total)
            if n % 50 == 0:
                self.log(f"{n:,}/{total:,} — {counters['ok']:,} indexed, "
                         f"{counters['quarantined']:,} quarantined, "
                         f"{counters['failed']:,} failed", phase=Phase.ingest)

        tally = await pipeline.run_batch(
            self.scope_id, work,
            workers=opts.workers,
            embed_concurrency=opts.embed_concurrency,
            on_result=on_result,
            should_cancel=self._cancelled,
        )

        jobstore.update_phase(self.job.job_id, Phase.ingest,
                              items_done=counters["done"], tally=tally)
        if "cancelled" in tally:
            raise Cancelled("Cancel requested by operator.")
        return tally

    def _phase_verify(self) -> dict:
        """Report what actually landed, from the database rather than counters."""
        stats = store.corpus_stats(self.scope_id)
        by_status = store.job_document_tally(self.job.job_id)

        jobstore.update_progress(
            self.job.job_id,
            chunks_total=stats["chunks"],
            docs_ok=by_status.get("indexed", 0) + by_status.get("degraded", 0),
            docs_failed=by_status.get("failed", 0),
            docs_quarantined=by_status.get("quarantined", 0))

        self.log(f"Collection now holds {stats['documents']:,} document(s), "
                 f"{stats['chunks']:,} chunk(s), {stats['vectors']:,} vector(s).",
                 phase=Phase.verify)

        if stats["chunks"] and stats["vectors"] < stats["chunks"]:
            missing = stats["chunks"] - stats["vectors"]
            self.log(f"{missing:,} chunk(s) have no vector — those documents are "
                     f"searchable by keyword but not by meaning. Re-run to retry "
                     f"the embedding step.", "warn", Phase.verify)

        return by_status

    # ------------------------------------------------------------ orchestration

    async def run(self) -> JobStatus:
        job_id = self.job.job_id
        jobstore.set_status(job_id, JobStatus.running)

        coll = collections.get_collection(self.scope_id)
        self.log(f"Started. Target collection: {coll['name'] if coll else self.scope_id}")

        try:
            if self.job.kind == JobKind.graph_build:
                await self._run_graph_build()
            else:
                await self._run_ingest()

            jobstore.update_progress(job_id, progress_pct=100.0, current_phase=None)
            jobstore.set_status(job_id, JobStatus.succeeded)
            self.log("Finished.")
            return JobStatus.succeeded

        except Cancelled as e:
            self._mark_unfinished("skipped")
            jobstore.set_status(job_id, JobStatus.cancelled, str(e))
            self.log("Cancelled. Work already completed remains; re-running "
                     "resumes from where this stopped.", "warn")
            return JobStatus.cancelled

        except Exception as e:  # noqa: BLE001
            self._mark_unfinished("pending")
            jobstore.set_status(job_id, JobStatus.failed, f"{type(e).__name__}: {e}")
            self.log(f"Failed: {type(e).__name__}: {e}", "error")
            if self.job.kind == JobKind.graph_build:
                collections.set_graph_build_failed(self.scope_id)
            return JobStatus.failed

        finally:
            shutil.rmtree(self.staging, ignore_errors=True)
            jobstore.prune_events(job_id)

    async def _run_ingest(self) -> None:
        job_id = self.job.job_id

        # ---- expand ----
        self._run_phase_sync(Phase.expand, self._phase_expand)

        # ---- discover ----
        jobstore.update_phase(job_id, Phase.discover, status="running",
                              started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=Phase.discover)
        work = await self._phase_discover()
        jobstore.update_phase(job_id, Phase.discover, status="succeeded",
                              finished_at=utcnow())
        self._done_weight += self._weights[Phase.discover]

        # ---- ingest ----
        jobstore.update_phase(job_id, Phase.ingest, status="running",
                              started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=Phase.ingest)
        await self._phase_ingest(work)
        jobstore.update_phase(job_id, Phase.ingest,
                              status="succeeded" if work else "skipped",
                              finished_at=utcnow())
        self._done_weight += self._weights[Phase.ingest]

        # ---- verify ----
        self._run_phase_sync(Phase.verify, self._phase_verify)

    async def _run_graph_build(self) -> None:
        """resolve -> write -> verify, over an EXISTING collection's chunks.

        No folders, no extraction, no embedding — this reads what I4 enrich
        already produced and is the only thing this job kind touches.
        """
        from docstore import collections as coll_mod

        job_id = self.job.job_id
        coll_mod.set_graph_status(self.scope_id, "building")

        if not graph_store.is_available():
            reason = graph_store.unavailable_reason() or "unknown reason"
            raise RuntimeError(f"Neo4j is not reachable ({reason}). "
                              f"Check neo4j_uri/neo4j_user/neo4j_password.")

        # ---- resolve ----
        jobstore.update_phase(job_id, Phase.resolve, status="running", started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=Phase.resolve)
        self._check_cancel()

        threshold = float(getattr(get_settings(), "graph_merge_threshold", 0.62))
        workers = self.job.options.workers if self.job.options else 0
        result = await self._to_thread_build_graph(threshold, workers)

        jobstore.update_phase(job_id, Phase.resolve, status="succeeded",
                              finished_at=utcnow(),
                              tally={"mentions": result.mention_count,
                                    "entities": len(result.entities)})
        self.log(f"Resolved {result.mention_count:,} mention(s) into "
                 f"{len(result.entities):,} entities and {len(result.edges):,} "
                 f"relationship(s) in {result.elapsed_ms:,}ms.", phase=Phase.resolve)
        self._done_weight += self._weights[Phase.resolve]
        self._set_progress(Phase.resolve, 1.0)

        # ---- write ----
        jobstore.update_phase(job_id, Phase.write, status="running", started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=Phase.write)
        self._check_cancel()

        graph_store.ensure_constraints()
        graph_store.clear_collection(self.scope_id)   # clean rebuild, never orphaned nodes
        n_entities = graph_store.upsert_entities(self.scope_id, result.entities)
        n_edges = graph_store.upsert_cooccurrence_edges(self.scope_id, result.edges)

        jobstore.update_phase(job_id, Phase.write, status="succeeded",
                              finished_at=utcnow(),
                              tally={"entities_written": n_entities, "edges_written": n_edges})
        self.log(f"Wrote {n_entities:,} entity node(s) and {n_edges:,} "
                 f"relationship(s) to Neo4j.", phase=Phase.write)
        self._done_weight += self._weights[Phase.write]
        self._set_progress(Phase.write, 1.0)

        # ---- verify ----
        jobstore.update_phase(job_id, Phase.verify, status="running", started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=Phase.verify)
        stats = graph_store.collection_stats(self.scope_id)
        chunk_count = store.corpus_stats(self.scope_id)["chunks"]
        coll_mod.set_graph_status(self.scope_id, "ready",
                                  chunk_count_at_build=chunk_count, stats=stats)
        self.log(f"Graph ready: {stats['nodes']:,} nodes, {stats['edges']:,} "
                 f"edges. This graph reflects {chunk_count:,} chunk(s) as of now — "
                 f"re-ingesting more documents into this collection will make it "
                 f"stale until rebuilt.", phase=Phase.verify)
        jobstore.update_phase(job_id, Phase.verify, status="succeeded", finished_at=utcnow())
        self._done_weight += self._weights[Phase.verify]
        self._set_progress(Phase.verify, 1.0)

    async def _to_thread_build_graph(self, threshold: float, workers: int):
        import asyncio
        return await asyncio.to_thread(
            graph_pipeline.build_graph, self.scope_id, threshold, workers)

    def _run_phase_sync(self, phase: Phase, fn) -> None:
        job_id = self.job.job_id
        jobstore.update_phase(job_id, phase, status="running", started_at=utcnow())
        jobstore.update_progress(job_id, current_phase=phase)
        tally = fn() or {}
        jobstore.update_phase(job_id, phase, status="succeeded",
                              finished_at=utcnow(), tally=tally)
        self._done_weight += self._weights[phase]
        self._set_progress(phase, 1.0)

    def _mark_unfinished(self, status: str) -> None:
        for pr in self.job.phases:
            if pr.status in ("pending", "running"):
                jobstore.update_phase(self.job.job_id, pr.phase,
                                      status=status, finished_at=utcnow())
