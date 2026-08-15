"""jobs/worker.py — the ingestion worker daemon.

    python -m jobs.worker

Run this as a separate process from uvicorn. Not for tidiness — for two
concrete reasons:

  * A batch job spawns a ProcessPoolExecutor sized to the machine's cores. Doing
    that inside the API process means an ingestion run competes with chat
    requests for CPU, and a user asking a question during a large ingest waits
    behind PDF parsing.

  * A pool that forks from a process holding open SQLite connections and an
    asyncio event loop is a reliable source of hangs that only appear under
    load. A clean process is a clean fork.

Concurrency is one job at a time, deliberately. Two jobs writing the same
collection would interleave their document rows harmlessly, but they would also
double the process pool and oversubscribe the box, which makes both slower than
running them in sequence.

Shutdown is graceful: SIGTERM stops the claim loop and lets the current job
reach a document boundary, so a deploy never leaves chunks saved without vectors.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import sys
import traceback
import uuid

from docstore import store as docstore_store
from jobs import store as jobstore
from jobs.models import JobStatus
from jobs.runner import JobRunner

_shutdown = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    _shutdown = True
    print(f"[worker] signal {signum} — finishing the current job, then exiting.",
          flush=True)


async def _loop(worker_id: str, poll_seconds: float, once: bool) -> None:
    last_ping = 0.0

    while not _shutdown:
        # Ping on a timer independent of job activity — this is what lets the
        # dashboard show "worker running, queue empty" instead of nothing at
        # all when there's no job to attach a heartbeat to.
        now = asyncio.get_event_loop().time()
        if now - last_ping > 10:
            jobstore.worker_ping(worker_id)
            last_ping = now

        job = jobstore.claim_next(worker_id)
        if job is None:
            if once:
                return
            await asyncio.sleep(poll_seconds)
            continue

        print(f"[worker] claimed {job.job_id} — {job.name}", flush=True)
        try:
            status = await JobRunner(job).run()
            print(f"[worker] {job.job_id} → {status.value}", flush=True)
        except Exception as e:  # noqa: BLE001
            # The runner handles its own failures; reaching here means the runner
            # itself broke. Record it rather than letting the worker die quietly
            # with the job stuck in `running`.
            traceback.print_exc()
            jobstore.set_status(job.job_id, JobStatus.failed,
                                f"Worker crashed: {type(e).__name__}: {e}")
            jobstore.add_event(job.job_id, f"Worker crashed: {e}", "error")


def main() -> None:
    ap = argparse.ArgumentParser(prog="jobs.worker")
    ap.add_argument("--poll-seconds", type=float, default=3.0)
    ap.add_argument("--once", action="store_true",
                    help="Drain the queue and exit (for cron or CI).")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    docstore_store.init_db()
    jobstore.init_db()

    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
    reclaimed = jobstore.reclaim_stale()
    if reclaimed:
        print(f"[worker] requeued {reclaimed} job(s) from an expired lease.", flush=True)

    jobstore.register_worker(worker_id, pid=os.getpid(), hostname=socket.gethostname())
    print(f"[worker] id={worker_id} db={docstore_store._db_path()}", flush=True)

    try:
        asyncio.run(_loop(worker_id, args.poll_seconds, args.once))
    except KeyboardInterrupt:
        pass
    finally:
        # Deregistering here means a Ctrl-C'd worker disappears from the
        # dashboard immediately instead of appearing "alive" for the next 20s.
        jobstore.deregister_worker(worker_id)
    print("[worker] stopped.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
