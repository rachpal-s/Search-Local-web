"""workflow/inflight.py — run-scoped registry for deferred background work.

Why this exists
---------------
A LangGraph node's asyncio task ends when the node returns. So a worker that
wants to keep working after it has answered cannot hold the background task
itself, and it cannot park it in graph state either (state is serialised
data, not live tasks). This module holds those tasks outside the graph,
keyed by a run id, and exposes:

    defer()        - schedule background work for the current run
    pending_urls() - what is still in flight (supervisor/critic read this)
    drain()        - collect finished results without blocking
    wait_all()     - bounded wait for the stragglers before the final answer
    cleanup()      - cancel leftovers and drop the run

The run id travels via a ContextVar rather than through graph state, because
`Send(agent_name, payload)` in workflow/routing.py passes ONLY the task
payload to the worker — there is no channel to thread an extra field through
without changing the router and every worker's TypedDict. asyncio tasks
inherit a copy of the context at creation, so any node running inside the
graph's task tree reads the id that main.py set.

IMPORTANT CAVEAT ON CANCELLATION
--------------------------------
The Playwright leg runs via asyncio.to_thread, and threads are not
cancellable in Python. cancel_all() stops the *awaiting* coroutine, so the
run is not held open, but a Chromium render already in progress will finish
on its own. _playwright() closes its own browser in a context manager, so
this leaks time, not processes. Keep scrape_max_concurrent_heavy small.
"""
import asyncio
import collections
import contextvars
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import get_settings

cfg = get_settings()

# Set by main.py at the start of a request; read by any node in the graph.
current_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_run_id", default="default"
)


class _Run:
    """Live background work for a single graph run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started = time.time()
        self.tasks: Dict[str, asyncio.Task] = {}      # key -> task
        # Flipped by request_stop() from a DIFFERENT request/task. Plain
        # bool, no lock: single-process, GIL-atomic write, read-only
        # everywhere else.
        self.stop_requested = False
        # Task identities actually dispatched this turn, and the ones the
        # router refused. Both live here rather than in graph state because
        # a conditional-edge router returns a transition, not a state update
        # — it has no channel to write to. See workflow/routing.py.
        self.dispatched_keys: set = set()
        self.dropped: List[Dict[str, str]] = []
        # deque, NOT asyncio.Queue: supervisor_node and critic_node are sync
        # functions, so LangGraph runs them in a worker thread. deque append
        # and popleft are atomic under the GIL; asyncio.Queue is not designed
        # for cross-thread access.
        self.done: "collections.deque[Any]" = collections.deque()
        self.completed = 0
        self.failed = 0


_RUNS: Dict[str, _Run] = {}


# ── run lifecycle ─────────────────────────────────────────────────────────────

def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def set_run(run_id: Optional[str] = None) -> str:
    """Bind a run id to the current context and register it. Returns the id."""
    run_id = run_id or new_run_id()
    current_run_id.set(run_id)
    _RUNS.setdefault(run_id, _Run(run_id))
    return run_id


def get_run_id() -> str:
    return current_run_id.get()


def _run(run_id: Optional[str] = None) -> _Run:
    rid = run_id or get_run_id()
    if rid not in _RUNS:
        _RUNS[rid] = _Run(rid)
    return _RUNS[rid]


# ── scheduling ────────────────────────────────────────────────────────────────

async def _wrap(run: _Run, key: str,
                work: Callable[[], Awaitable[Any]],
                formatter: Optional[Callable[[Any], Any]]) -> None:
    """Await the deferred work and queue its formatted result."""
    payload: Any
    try:
        result = await work()
        payload = formatter(result) if formatter else result
        run.completed += 1
    except asyncio.CancelledError:
        run.tasks.pop(key, None)
        raise
    except Exception as e:                       # never let a background task
        run.failed += 1                          # take down the event loop
        payload = f"Deferred task failed for {key}: {e}"
    run.tasks.pop(key, None)
    run.done.append(payload)


def defer(key: str, work: Callable[[], Awaitable[Any]],
          formatter: Optional[Callable[[Any], Any]] = None,
          run_id: Optional[str] = None) -> bool:
    """Schedule `work` in the background for this run.

    `key` is the dedupe/reporting handle (a URL, for the scraper). Returns
    False if the work was NOT scheduled — already in flight, or the run has
    hit scrape_max_deferred — so the caller can fall back to reporting the
    partial result as final.
    """
    run = _run(run_id)
    if key in run.tasks:
        return False
    limit = int(getattr(cfg, "scrape_max_deferred", 6))
    if len(run.tasks) >= limit:
        return False
    try:
        task = asyncio.create_task(_wrap(run, key, work, formatter))
    except RuntimeError:
        # No running loop — caller is in a sync context and cannot defer.
        # Return False so the caller finishes the work inline instead of
        # dropping it. (scraper_node is async, so this is a safety net.)
        return False
    run.tasks[key] = task
    return True


# ── cooperative stop ("finish now") ───────────────────────────────────────────
#
# POST /chat/stop arrives as a SEPARATE request, in a separate asyncio task,
# with its own copy of the context — so it CANNOT set the running graph's
# ContextVar. It flips a flag on the shared _RUNS entry instead, and the
# routers (which run inside the graph and therefore DO inherit
# current_run_id) read it on their next decision. That module-global dict is
# the whole reason this works; do not "tidy" the flag into a ContextVar.
#
# Both helpers use _RUNS.get(), NOT the _run() helper above. _run()
# auto-creates a missing entry, which is right for defer() but wrong here: a
# browser posting a stale run_id after its turn finished would silently
# create a phantom run that nothing ever cleans up. An unknown run id means
# "that turn is already over" — say so and do nothing.


def request_stop(run_id: str) -> bool:
    """Ask the run to finish at its next routing decision.

    Returns False if the run is unknown (already finished or never existed),
    so the caller can report that honestly rather than pretending it worked.
    """
    run = _RUNS.get(run_id)
    if run is None:
        return False
    run.stop_requested = True
    return True


def stop_requested(run_id: Optional[str] = None) -> bool:
    """True if this run has been asked to finish early. Never creates a run."""
    run = _RUNS.get(run_id or get_run_id())
    return bool(run and run.stop_requested)


# ── dispatch ledger ───────────────────────────────────────────────────────────
#
# Why here and not in AgentState: route_from_supervisor is a conditional-edge
# function. LangGraph uses its RETURN VALUE as the next transition, so it
# cannot also emit a state update — there is nowhere for it to record what it
# dispatched or refused. The run registry is the channel that already exists
# for exactly this problem (agents/supervisor.py reads drain()/pending_urls()
# from here for the same reason), so the ledger lives here too and is dropped
# by cleanup() with the rest of the run.


def note_dispatch(key: str, run_id: Optional[str] = None) -> None:
    """Record that a specific (agent, payload) identity has been dispatched."""
    _run(run_id).dispatched_keys.add(key)


def was_dispatched(key: str, run_id: Optional[str] = None) -> bool:
    run = _RUNS.get(run_id or get_run_id())
    return bool(run and key in run.dispatched_keys)


def note_dropped(agent: str, detail: str, reason: str,
                 run_id: Optional[str] = None) -> None:
    """Record a task the router refused, for the supervisor to read next loop.

    Without this the refusal was a print() and nothing more: the supervisor's
    prompt carries context, feedback and pending URLs but never learned that
    two of the four scrapes it asked for were silently dropped. Its only
    rational move was to ask again, which produced no new context, which
    tripped the stagnation interceptor. The loop was caused by the silence,
    not by the cap.
    """
    _run(run_id).dropped.append(
        {"agent": agent, "detail": detail, "reason": reason})


def take_dropped(run_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Read and CLEAR the drop list.

    Cleared on read so a drop is reported to the supervisor exactly once. Left
    in place it would be restated every loop, and a permanent 'this was
    refused' line in the prompt reads as a standing instruction rather than
    news about the last decision.
    """
    run = _RUNS.get(run_id or get_run_id())
    if not run or not run.dropped:
        return []
    out, run.dropped = list(run.dropped), []
    return out


# ── inspection & collection ───────────────────────────────────────────────────

def pending_urls(run_id: Optional[str] = None) -> List[str]:
    """Keys still in flight. The supervisor must not re-dispatch these."""
    return [k for k, t in _run(run_id).tasks.items() if not t.done()]


def pending_count(run_id: Optional[str] = None) -> int:
    return len(pending_urls(run_id))


def drain(run_id: Optional[str] = None) -> List[Any]:
    """Non-blocking: pull everything that has finished since the last drain."""
    run = _run(run_id)
    out: List[Any] = []
    while True:
        try:
            out.append(run.done.popleft())
        except IndexError:
            break
    return out


async def wait_all(run_id: Optional[str] = None,
                   timeout: Optional[float] = None) -> List[Any]:
    """Bounded wait for stragglers, then drain. Never raises on timeout."""
    run = _run(run_id)
    if timeout is None:
        timeout = float(getattr(cfg, "scrape_late_wait_seconds", 25.0))
    tasks = [t for t in run.tasks.values() if not t.done()]
    if tasks:
        try:
            await asyncio.wait(tasks, timeout=timeout)
        except Exception:
            pass
    return drain(run_id)


def stats(run_id: Optional[str] = None) -> Dict[str, Any]:
    run = _run(run_id)
    return {
        "run_id": run.run_id,
        "pending": pending_count(run_id),
        "completed": run.completed,
        "failed": run.failed,
        "age_s": round(time.time() - run.started, 1),
    }


# ── teardown ──────────────────────────────────────────────────────────────────

def cancel_all(run_id: Optional[str] = None) -> int:
    """Cancel outstanding tasks. See the module docstring on thread caveats."""
    run = _run(run_id)
    n = 0
    for t in list(run.tasks.values()):
        if not t.done():
            t.cancel()
            n += 1
    return n


def cleanup(run_id: Optional[str] = None) -> None:
    """Cancel leftovers and forget the run. Always call this in a finally."""
    rid = run_id or get_run_id()
    if rid in _RUNS:
        cancel_all(rid)
        _RUNS.pop(rid, None)