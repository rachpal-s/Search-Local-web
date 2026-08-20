"""observability.py — self-hosted Arize Phoenix telemetry.

Second implementation of this module. The first targeted LangSmith; that was
abandoned when self-hosting turned out to be an Enterprise-licensed add-on,
discovered only after building against it. Phoenix is MIT-licensed and
genuinely self-hostable — this rewrite is smaller than it looks because the
call sites in main.py barely changed; only what happens inside this file did.

Scope, unchanged from the LangSmith version: LIVE CHAT CONVERSATIONS ONLY.
Ingestion (embedding calls, batch jobs) stays out of scope for this pass —
different volume profile, different urgency.

The core guarantee is also unchanged: when `phoenix_tracing_enabled` is False
(the default), NOTHING here runs. No instrumentor activated, no tracer
created, zero latency risk, zero behavior change. Every function checks the
flag first; nothing here is ever allowed to raise into the request path.

How this actually differs from the LangSmith attempt, mechanically
------------------------------------------------------------------
LangSmith's LangChain integration is env-var-triggered: set
LANGCHAIN_TRACING_V2=true and its callback system activates ambiently, no
code changes needed anywhere else. Phoenix's is similar in SPIRIT —
LangChainInstrumentor().instrument() at startup globally patches LangChain,
so the two real LLM calls (supervisor_node, critic_node, both ChatOllama) and
every other graph node need zero per-call-site changes to become traced spans.

Where it genuinely differs: attaching conversation/user identity to those
spans is NOT done via LangChain's own `config={"metadata": ...}` dict the way
LangSmith reads it. OpenInference has its own context managers for this —
using_session(), using_user(), using_metadata() — which set OpenTelemetry
context for their `with` block's duration. tracing_context() below composes
all three into one context manager the call sites wrap their graph
invocation in.

One honest gap: I could not execute-test the context-manager-around-a-
long-lived-async-generator pattern against a real Phoenix instance — no
server available in the environment this was built in. Python's contextvars
(what OpenTelemetry context propagation is built on) correctly cross `await`
points within the same async task, so this should work, but "should" is
doing real work in that sentence. Worth watching on first real deployment:
if user_id/session_id don't show up in Phoenix's UI, that's where to look —
the traces themselves (via LangChainInstrumentor's global patch) do not
depend on this working, since that part activates independently.

Package name trap, worth repeating here since it's easy to get wrong:
`arize-otel` (no "phoenix") points at Arize's separate COMMERCIAL cloud
platform and needs a paid space_id/api_key. `arize-phoenix-otel` is the
open-source, self-hosted one this module actually uses.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

from config import get_settings

cfg = get_settings()

_tracer_provider = None
_langchain_instrumentor = None
_spawned_process: subprocess.Popen | None = None


def _phoenix_reachable(timeout: float = 1.5) -> bool:
    """Quick HTTP check — stdlib only, no new dependency for something this
    small. Not the same check as Neo4j's driver-level verify_connectivity();
    Phoenix's OTLP exporter doesn't block-verify a connection at setup time,
    it only fails on first export, so this is the only way to know BEFORE
    deciding whether to spawn a redundant second server.
    """
    try:
        urllib.request.urlopen(cfg.phoenix_endpoint.rstrip("/") + "/", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True   # any HTTP response at all means something is listening
    except Exception:  # noqa: BLE001 — connection refused, timeout, DNS, etc.
        return False


def _spawn_phoenix_server() -> bool:
    """Launch `phoenix serve` as a genuine separate OS process.

    Deliberately NOT px.launch_app() — that is Arize's own notebook-only
    in-process mode, explicitly documented as possibly not working outside
    one. `python -m phoenix.server.main serve` is their documented
    process-based (production) launch path; this just automates running it
    as a subprocess instead of you typing it in a second terminal.

    start_new_session=True (POSIX) / a new process group (Windows) means
    this process does NOT die when uvicorn restarts or reloads — it keeps
    running independently, closer to a properly supervised service than a
    child tied to this app's exact lifecycle. It will NOT survive the whole
    machine restarting, though — this is convenience, not a substitute for
    a real supervisor (systemd, Docker, Task Scheduler) if you want that.
    """
    global _spawned_process
    try:
        log_path = "data/phoenix_server.log"
        import os
        os.makedirs("data", exist_ok=True)
        log_file = open(log_path, "a")

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        _spawned_process = subprocess.Popen(
            [sys.executable, "-m", "phoenix.server.main", "serve"],
            stdout=log_file, stderr=log_file, **kwargs)
        print(f"[observability] Spawned `phoenix serve` (pid {_spawned_process.pid}), "
              f"log -> {log_path}")

        # Bounded wait — this only runs once at startup, not per-request, so
        # a brief blocking poll here is fine (same reasoning as store.init_db()
        # being synchronous at boot).
        for _ in range(20):   # ~10s at 0.5s intervals
            if _phoenix_reachable(timeout=0.5):
                return True
            time.sleep(0.5)
        print("[observability] ⚠️ `phoenix serve` was launched but did not "
              "become reachable within 10s. Check data/phoenix_server.log.")
        return False
    except FileNotFoundError:
        print("[observability] ⚠️ Could not spawn `phoenix serve` — is "
              "arize-phoenix (not just arize-phoenix-otel) installed? "
              "`pip install arize-phoenix`.")
        return False
    except Exception as e:  # noqa: BLE001 — must never block app startup
        print(f"[observability] ⚠️ Could not spawn Phoenix server: {e}")
        return False


def _http_header_safe(value: str) -> str:
    """HTTP header values must be latin-1 encodable — that's an HTTP spec
    constraint enforced by http.client/urllib3, not an OpenTelemetry one, and
    it crashes hard (UnicodeEncodeError) rather than degrading gracefully.

    An em-dash, a smart quote, anything pasted from Word/Outlook autocorrect
    into an env var (PHOENIX_PROJECT, PHOENIX_API_KEY) is enough to trigger
    it. This strips anything outside latin-1 rather than trusting config
    values to already be header-safe — cheap, and closes the whole bug class
    regardless of which config value turns out to be the culprit.
    """
    return value.encode("latin-1", errors="ignore").decode("latin-1")


def init_tracing() -> None:
    """Call once at app startup. No-ops entirely when disabled."""
    global _tracer_provider, _langchain_instrumentor
    if _tracer_provider is not None or not cfg.phoenix_tracing_enabled:
        return

    if not _phoenix_reachable():
        if cfg.phoenix_auto_launch:
            print(f"[observability] Phoenix not reachable at {cfg.phoenix_endpoint} "
                  f"— launching it as a separate process (phoenix_auto_launch=True)...")
            if not _spawn_phoenix_server():
                print("[observability] ⚠️ Continuing without tracing — spans will "
                      "fail to export until Phoenix is reachable, but nothing in "
                      "the app is blocked by this.")
                return
        else:
            print(f"[observability] ⚠️ Phoenix not reachable at {cfg.phoenix_endpoint} "
                  f"and phoenix_auto_launch=False — start it yourself, or set "
                  f"phoenix_auto_launch=True to have this app do it. Continuing "
                  f"without tracing for now.")
            return

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        headers = {}
        if cfg.phoenix_api_key:
            headers["Authorization"] = f"Bearer {_http_header_safe(cfg.phoenix_api_key)}"

        project_name = _http_header_safe(cfg.phoenix_project)
        endpoint = _http_header_safe(cfg.phoenix_endpoint.rstrip("/"))

        resource = Resource.create({"openinference.project.name": project_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
        # Batched, background-thread export — the same reason LangSmith's
        # client doesn't block the request path: spans queue and upload
        # asynchronously, not inline with the graph call that created them.
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace_api.set_tracer_provider(provider)

        # Explicit instrumentation, not auto_instrument=True. The latter
        # activates every OpenInference package it finds installed —
        # predictable today, a silent surprise later if something else in
        # requirements.txt ever transitively pulls in another one.
        instrumentor = LangChainInstrumentor()
        instrumentor.instrument(tracer_provider=provider)

        _tracer_provider = provider
        _langchain_instrumentor = instrumentor
        print(f"[observability] Phoenix tracing enabled -> "
              f"{cfg.phoenix_endpoint} (project: {cfg.phoenix_project})")
    except Exception as e:  # noqa: BLE001 — tracing setup must never block boot
        print(f"[observability] ⚠️ Could not initialize Phoenix tracing: {e}. "
              f"Continuing without it.")


def _compose(*context_managers) -> contextlib.ExitStack:
    """Combine several context managers into one, entered/exited together."""
    stack = contextlib.ExitStack()
    for cm in context_managers:
        stack.enter_context(cm)
    return stack


def tracing_context(conversation_id: str, user_id: str):
    """Context manager wrapping one graph invocation.

    Usage:
        with tracing_context(conversation_id, user_id):
            async for event in app_graph.astream_events(...): ...

    Returns contextlib.nullcontext() when tracing is disabled, so call sites
    never need an if/else — `with tracing_context(...):` is always safe to
    write, disabled or not.

    conversation_id doubles as the OpenInference "session" — a chat thread
    IS a session in the sense that concept means here, so no separate
    identifier was invented for it.
    """
    if not cfg.phoenix_tracing_enabled or _tracer_provider is None:
        return contextlib.nullcontext()
    try:
        from openinference.instrumentation import using_metadata, using_session, using_user
        return _compose(
            using_session(conversation_id),
            using_user(user_id),
            using_metadata({"conversation_id": conversation_id, "user_id": user_id}),
        )
    except Exception as e:  # noqa: BLE001 — must never break the request path
        print(f"[observability] ⚠️ tracing_context failed, continuing without it: {e}")
        return contextlib.nullcontext()


def run_config(conversation_id: str, user_id: str) -> dict:
    """LangChain-native metadata/tags, passed alongside tracing_context() as
    a redundant second path — kept from the LangSmith version rather than
    dropped, since OpenInference's LangChain instrumentor hooks the same
    callback/run system LangChain's own config metadata flows through, and
    there is a real chance some of this also surfaces as span attributes.
    Costs nothing to include; only tracing_context() above is confirmed to
    be the documented, primary mechanism.
    """
    if not cfg.phoenix_tracing_enabled:
        return {}
    return {
        "metadata": {"conversation_id": conversation_id, "user_id": user_id},
        "tags": [f"user:{user_id}", f"conversation:{conversation_id}"],
        "run_name": "chat_turn",
    }


def trace_url(conversation_id: str) -> str | None:
    """Link to the Phoenix UI, or None when tracing is disabled.

    Points at the general UI root, not a guessed per-conversation deep link
    — Phoenix's exact session-deep-link URL structure wasn't confirmable
    from documentation, and a wrong guess here is a dead link that looks
    like it should work, which is worse than an honest general link. Filter
    by session_id (== conversation_id) in Phoenix's own UI until the real
    deep-link path is confirmed against a live instance.
    """
    if not cfg.phoenix_tracing_enabled:
        return None
    try:
        return cfg.phoenix_trace_url_template.format(
            endpoint=cfg.phoenix_endpoint.rstrip("/"),
            project=cfg.phoenix_project,
            conversation_id=conversation_id,
        )
    except Exception:  # noqa: BLE001 — a bad template must not break the response
        return None
