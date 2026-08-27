"""main.py — FastAPI application entry point.

Agent and graph logic lives under agents/ and workflow/; document ingestion and
chat persistence live under docstore/; the HTTP surface for conversations and
uploads lives under routers/. This file wires them together and owns the two
chat endpoints.
"""
import json
import os
import re
import urllib.parse

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import get_settings
from docstore import collections as doc_collections
from docstore import graph_store
from docstore import session, store
from docstore.retrieve import context_for_query
from observability import init_tracing, run_config, trace_url
# tracing_context() also exists in observability.py (session/user attribution
# via OpenInference context managers) but is deliberately NOT used at either
# graph-invocation call site below. Wrapping the streaming astream_events
# loop in it caused final_response to come back empty from an otherwise
# successful run — critic scoring the answer, the graph completing cleanly
# server-side, but the frontend receiving nothing. Root cause not yet
# isolated; reverted rather than left in while broken. Session/user
# attribution for Phoenix needs a different mechanism before being
# reintroduced — see the note in observability.py.
from routers import conversations, jobs as jobs_router, uploads
from jobs import store as jobstore
from workflow import app_graph, inflight, llm as llm_select
from workflow.registry import AGENT_REGISTRY
from workflow.streaming import PartialJSONFieldStreamer

cfg = get_settings()
print(f"Model used for inferencing: {cfg.ollama_inference_model}")

app = FastAPI(title=cfg.app_title)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
DOWNLOAD_DIR = "static/downloads"

app.include_router(conversations.router)
app.include_router(uploads.router)
# The jobs router owns /jobs, /api/jobs*, /api/collections* and /api/filetypes.
# It only ever writes rows — the worker process (python -m jobs.worker) does the
# ingesting, so a four-hour batch never competes with chat for CPU here.
app.include_router(jobs_router.router)
from routers import telemetry as telemetry_router
app.include_router(telemetry_router.router)


def _check_neo4j_at_boot() -> None:
    """Loud, specific, and actionable — unlike graph_store's own runtime
    checks, which stay quiet by design (a query that never touched a graph
    feature has no reason to know or care about Neo4j's state). At boot,
    "the operator is watching the console" is a safe assumption, so this is
    the one place a clear failure message is worth the noise: it turns
    "why did my graph build just fail" into "oh, I forgot to start Neo4j
    Desktop" before anyone has to debug it.

    Neo4j itself stays exactly as it was — not auto-launched, not managed by
    this app. Neo4j Desktop has no scriptable "start database X"
    non-interactively; there is nothing safe to automate here the way
    Phoenix's `phoenix serve` could be. This is a diagnostic, not a fix.
    """
    if graph_store.is_available():
        print(f"[boot] 🕸️  Neo4j reachable at {cfg.neo4j_uri} — "
              f"knowledge graph features are usable.")
        return

    reason = graph_store.unavailable_reason() or "unknown reason"
    print("=" * 64)
    print("⚠️  NEO4J NOT REACHABLE")
    print(f"    Tried:  {cfg.neo4j_uri}")
    print(f"    Reason: {reason}")
    print("    Knowledge graph builds and hydration will not work until this")
    print("    is fixed — everything else in the app is unaffected either way.")
    print("    If you're on Neo4j Desktop: open it and start your database,")
    print("    then either restart this app or just retry — the next graph")
    print("    build will pick it up without needing a restart.")
    print("=" * 64)


@app.on_event("startup")
async def _startup() -> None:
    """Create the chat schema and the directories the app writes into.

    Idempotent, so it is safe on every reload. Doing it here rather than at
    import time keeps `import main` side-effect-free for tests.
    """
    store.init_db()
    jobstore.init_db()
    init_tracing()
    _check_neo4j_at_boot()
    os.makedirs(cfg.upload_dir, exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(getattr(cfg, "job_staging_dir", "data/staging"), exist_ok=True)
    for root in (getattr(cfg, "ingest_allowed_roots", "data/incoming")
                 .split(os.pathsep)):
        if root.strip():
            os.makedirs(root.strip(), exist_ok=True)
    print(f"[boot] 🗄️  chat store ready at {cfg.chat_db_path} "
          f"(fts5={store.fts_enabled()})")
    print(f"[boot] 📎 uploads -> {cfg.upload_dir}, "
          f"embed model {cfg.ollama_embed_model} @ {cfg.embed_dimensions} dims")
    print(f"[boot] 🗂️  ingest roots -> {cfg.ingest_allowed_roots} "
          f"(run `python -m jobs.worker` to process queued jobs)")
    print(f"[boot] 🔭 telemetry -> "
          f"{'ON, ' + cfg.phoenix_endpoint if cfg.phoenix_tracing_enabled else 'off (default)'}")


def _snippet(text: str, max_chars: int = 220) -> str:
    """Short, sentence-aware excerpt for a live per-source card.

    Pure string slicing over content the scraper already returned — no LLM
    call, so this cannot add latency, which was the whole point of this
    feature (stream something meaningful while the graph is still working,
    at zero cost — if it didn't come free it wouldn't be worth doing).
    Backtracks to a sentence boundary where possible so the card doesn't
    end mid-word; this is genuinely user-facing text, not a debug log line.
    """
    if not text:
        return ""
    flat = " ".join(text.split())  # collapse newlines/repeated whitespace
    if len(flat) <= max_chars:
        return flat
    window = flat[:max_chars]
    for punct in (". ", "! ", "? "):
        idx = window.rfind(punct)
        if idx > max_chars * 0.4:  # reject a boundary so early it'd be a stub
            return window[: idx + 1].strip()
    idx = window.rfind(" ")  # fall back to a whole-word boundary
    return (window[:idx] if idx > 0 else window).strip() + "…"


def _source_card(context_entry: str):
    """Parses one scraper context string (see agents/web_scraper.py's
    `_result_payload`) into the small JSON shape the browser renders as a
    live card. Returns None for anything that isn't that JSON-payload shape
    — defensive on purpose, since a worker returning a plain string here
    must never crash the stream, just skip the card silently.
    """
    match = re.search(r"\{.*\}", context_entry, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "source" not in payload:
        return None
    return {
        "source": payload.get("source"),
        "title": payload.get("title") or payload.get("source"),
        "word_count": payload.get("word_count"),
        "error": payload.get("error"),
        "insufficient": bool(payload.get("insufficient_content")),
        "still_rendering": bool(payload.get("still_rendering")),
        "snippet": _snippet(payload.get("scraped_content", "")),
    }


def _stopped_without_answer(context: list) -> str:
    """Last-resort body for a user-stopped run that produced no prose.

    Only reachable when the stop landed while the supervisor's own LLM call
    was already in flight, so that invocation never saw the directive and
    returned tasks instead of an answer. Rather than show "Workflow
    terminated" — which reads like a crash for something the user chose —
    hand back the evidence itself. Deliberately no LLM call: the person just
    asked to stop waiting.
    """
    lines = []
    for entry in context or []:
        text = entry if isinstance(entry, str) else json.dumps(entry)
        text = text.strip()
        if text:
            lines.append(f"- {text[:600]}")
    body = "\n".join(lines[:25]) or "_Nothing had been gathered yet._"
    return ("⏹️ **Finished at your request, before an answer was written.** "
            "Here is the evidence gathered up to that point:\n\n" + body)


def _payload_of(entry) -> dict | None:
    """Pull the JSON payload out of a context/late entry, if there is one."""
    text = entry if isinstance(entry, str) else json.dumps(entry)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _worth_appending(payload: dict | None, known: dict) -> bool:
    """Is this late result worth showing the user at all?

    The gate exists because a deferred re-render frequently comes back with
    nothing new. Observed case: a directory page returned 99 words of nav
    chrome ending in "No record found", was deferred for a full render, and
    the full render returned the same 99 words — which were then appended
    under "Late Additions" as though they added something. They didn't; they
    just buried a correct answer under a menu listing.

    Two rejections, both cheap:
      * the scraper already flagged it as insufficient, and
      * the same source is already in context with an equal or better
        extraction, so this render is a duplicate, not an addition.
    """
    if not payload or not payload.get("word_count"):
        return False
    if payload.get("insufficient_content"):
        return False
    src = payload.get("source")
    if src and src in known and payload["word_count"] <= known[src]:
        return False
    return True


def _addendum(late: list, context: list | None = None) -> str:
    """Render late background results as an appendix to the final answer.

    Deliberately dumb: it appends rather than trying to re-open and rewrite the
    answer the critic already approved. If you want the late material woven in
    properly, that is a second supervisor+critic delta pass, not string surgery.

    Returns "" when nothing survives the quality gate, which the caller relies
    on — an empty addendum must stay empty rather than emitting a "Late
    Additions" heading with nothing useful under it.
    """
    if not late:
        return ""

    # What each source already contributed, so a re-render that adds nothing
    # can be recognised as a duplicate rather than an addition.
    known: dict = {}
    for entry in context or []:
        p = _payload_of(entry)
        if p and p.get("source") and p.get("word_count"):
            known[p["source"]] = max(known.get(p["source"], 0), p["word_count"])

    lines = []
    for entry in late:
        text = entry if isinstance(entry, str) else json.dumps(entry)
        payload = _payload_of(entry)
        if payload is not None and not _worth_appending(payload, known):
            print(f"[STREAM] 🗑️ Late result from "
                  f"{payload.get('source', '?')} added nothing new — not appended.")
            continue
        if payload and payload.get("word_count"):
            title = payload.get("title") or payload.get("source", "Source")
            src = payload.get("source", "")
            wc = payload.get("word_count")
            snippet = (payload.get("scraped_content") or "")[:400].strip()
            lines.append(f"**[{title}]({src})** — {wc} words\n\n{snippet}...")
        else:
            lines.append(f"- {text[:300]}")

    if not lines:
        return ""

    body = "\n\n".join(lines)
    return (
        "\n\n---\n\n### 🕐 Late Additions\n\n"
        "_These sources were still rendering when the answer above was finalised._\n\n"
        f"{body}"
    )


def _best_attempt(final_state: dict) -> tuple[str | None, int | None, str]:
    """Pick the highest-scoring answer the critic actually evaluated.

    The graph ends with whatever the LAST supervisor loop produced, but a
    later loop is not necessarily a better one — a critic rejection triggers
    regeneration, and that regeneration can come back worse or truncated.
    Taking the last attempt then throws away a better earlier answer, which
    is exactly the "content wiped after critic evaluation" symptom.

    Uses the critic's own score as the selection signal rather than a proxy
    like length: it is the actual quality judgement the system already
    produces. Ties go to the EARLIER attempt (max() is stable) — if a
    regeneration scored no better, the original stands rather than being
    churned for nothing.

    Returns (response, score, source) — `source` names which path was taken,
    so the caller can log it and the trace panel can show it.
    """
    attempts = [a for a in (final_state.get("response_attempts") or [])
                if a.get("response")]
    if not attempts:
        return final_state.get("final_response"), final_state.get("eval_score"), "final_state"

    best = max(attempts, key=lambda a: a.get("score", 0))
    last = attempts[-1]

    if best is last:
        return best["response"], best["score"], "final_state"
    return best["response"], best["score"], (
        f"best_of_{len(attempts)}_attempts (loop {best.get('loop')}, "
        f"score {best.get('score')} > last loop's {last.get('score')})")


def _initial_state(prompt: str, conversation_id: str, doc_blocks: list,
                   attached: list, history: list) -> dict:
    """Seed graph state for one turn.

    Retrieved excerpts are seeded into `context` rather than passed separately so
    the supervisor's FIRST decision already knows what the attachments say.
    Without that, it reliably dispatches a web search for something sitting in
    the PDF the user just uploaded.
    """
    return {
        "user_query": prompt,
        "conversation_id": conversation_id,
        "context": list(doc_blocks),
        "action_logs": [],
        "score_history": [],
        "response_attempts": [],
        "retrieval_attempts": 0,
        "dispatch_counts": {},
        "graph_traces": [],
        "attached_files": attached,
        "chat_history": history,
        "pending_tasks": [],
        "final_response": None,
        "feedback": None,
        "eval_score": 0,
        "loop_count": 0,
        "max_loops": 5,
        "critic_loop_count": 0,
        "max_critic_reviews": 3,
    }


@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    """The chat shell. All state loads client-side from the JSON API."""
    # request-first signature: the older TemplateResponse(name, {...}) form is
    # deprecated in Starlette >= 0.29 and removed in 1.x. This form works on
    # everything from 0.29 up, so it survives a FastAPI bump.
    return templates.TemplateResponse(
        request, "chat.html", {"app_title": cfg.app_title})


@app.get("/legacy", response_class=HTMLResponse)
async def legacy_form(request: Request):
    """The original single-shot form, kept for debugging the graph in isolation."""
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/chat/stream")
async def chat_stream(request: Request, prompt: str = Form(...),
                      conversation_id: str = Form(None),
                      user_id: str = Form(None),
                      model: str = Form(None),
                      critic_model: str = Form(None)):
    """Run one turn, streaming the execution trace as SSE.

    Order of operations matters here:
      1. resolve/create the conversation and bind it to the async context
      2. persist the user turn immediately, so a crash mid-run still leaves a
         readable transcript
      3. retrieve from the thread's own documents BEFORE the graph starts
      4. run the graph
      5. fold in late background scrapes
      6. persist the assistant turn AFTER (5), so a reloaded thread shows
         exactly what the user saw rather than the pre-addendum version

    user_id is a mock placeholder until real auth exists — the frontend
    generates and persists a pseudo-id client-side (static/js/chat.js) so
    traces are at least distinguishable per browser today. "anonymous" here
    is only reached by a caller that predates this (e.g. a script hitting
    the endpoint directly), so tracing degrades gracefully rather than 500ing.
    """
    user_id = user_id or "anonymous"
    if not conversation_id or not store.get_conversation(conversation_id):
        conversation_id = store.create_conversation()["id"]

    # Binds the conversation for every node inside the graph via ContextVar —
    # worker payloads are authored by the LLM and cannot be trusted to carry it.
    session.set_conversation(conversation_id)
    run_id = inflight.set_run()
    # Same out-of-band binding as the conversation above, and for the same
    # reason: Send() payloads are authored by the LLM, so a user preference
    # cannot travel through them. Values arrive from the browser and are
    # validated against the catalogue allow-list inside set_models — an
    # unknown id becomes the configured default rather than reaching Ollama.
    selected_models = llm_select.set_models(model, critic_model)

    print("\n==================================================")
    print(f"🚀 INCOMING QUERY: '{prompt}'")
    print(f"   conversation: {conversation_id}")
    print("==================================================")

    docs = [d for d in store.list_documents(conversation_id)
            if d["status"] in ("indexed", "degraded")]
    attached = [d["file_name"] for d in docs]
    history = store.recent_turns(conversation_id, cfg.chat_history_turns)

    store.add_message(
        conversation_id, "user", prompt,
        attachments=[{"doc_id": d["id"], "file_name": d["file_name"]} for d in docs])
    new_title = store.autotitle_if_default(conversation_id, prompt)

    async def event_generator():
        """Translate internal graph events into UI strings."""
        doc_blocks: list = []
        collected_logs: list = []

        def emit(message: str) -> str:
            collected_logs.append(message)
            return f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"

        # First event on the wire. The run id is generated server-side and had
        # no way to reach the browser before this; the "Finish now" control
        # needs it to name which run to stop. Emitted before any work starts so
        # the button is live for the whole turn, including the retrieval
        # pre-pass below.
        yield "data: " + json.dumps({"type": "run", "run_id": run_id}) + "\n\n"

        defaults = llm_select.catalogue()["defaults"]
        if (selected_models["primary"] != defaults["primary"]
                or selected_models["critic"] != defaults["critic"]):
            yield emit(f"🧩 Models for this turn — planner: "
                       f"{selected_models['primary']}, critic: "
                       f"{selected_models['critic']}.")

        # ---- retrieval pre-pass over this thread's full scope ----
        # "Full scope" = this conversation's own uploads PLUS any collections
        # attached to it (docstore/collections.py). Gating on `attached` alone
        # (this chat's own uploads) misses a chat that has no uploads of its
        # own but DOES have a collection attached — exactly the case the Jobs
        # dashboard exists to serve. Without this, the supervisor's first move
        # on a collection-only chat would go in blind, which is the specific
        # failure this pre-pass was built to prevent for uploads.
        scope_ids = doc_collections.scopes_for(conversation_id)
        scope_stats = store.corpus_stats_scoped(scope_ids) if scope_ids else {"chunks": 0}

        if scope_stats["chunks"]:
            label = (f"{len(attached)} attached document(s)" if attached
                     else "the attached knowledge collection(s)")
            yield emit(f"📚 Searching {label}...")
            try:
                doc_blocks, hits, graph_trace = await context_for_query(conversation_id, prompt)
                if hits:
                    files = sorted({h["file_name"] for h in hits})
                    yield emit(f"📚 {len(hits)} passage(s) from "
                               f"{', '.join(files)} pulled into context.")
                else:
                    yield emit("📚 No matching passages in the attachments.")
                if graph_trace:
                    # Ephemeral, on-demand only — never persisted (see the
                    # 'complete' event below, which deliberately does not
                    # include this). A frontend "show reasoning graph" button
                    # appears only when this event actually arrives; there is
                    # no separate availability check to keep in sync with it.
                    yield "data: " + json.dumps({
                        "type": "graph_trace", **graph_trace}) + "\n\n"
            except Exception as e:
                yield emit(f"📚 Document search failed: {type(e).__name__}: {e}")
        else:
            # This is the exact gap that caused the "went straight to web
            # search" symptom: an operator forgets the attach step (or never
            # knew which collection had their data), and nothing on this turn
            # signals that anything was skipped. The nudge costs one cheap
            # query and only fires when there is genuinely something to
            # attach — it stays silent for a chat that has no collections at
            # all, where going to the web is simply correct.
            non_empty = [c for c in doc_collections.list_collections() if c["chunks"]]
            if non_empty:
                yield emit(
                    f"📚 No documents or collections attached to this chat — "
                    f"{len(non_empty)} knowledge collection(s) exist and are "
                    f"not being searched. Open 'Knowledge collections' if your "
                    f"answer should include them.")

        initial_state = _initial_state(prompt, conversation_id, doc_blocks,
                                       attached, history)
        final_state: dict = {}
        supervisor_final_response: str | None = None
        # LangGraph's astream_events fires on_chain_end for BOTH the actual
        # graph node AND the internal Runnable wrapper around it, so a single
        # scraper/critic call's action_logs surface twice with identical
        # content (down to the millisecond timing baked into the string) —
        # looks like a real second pass, isn't one. Exact-string dedupe is
        # enough: a genuine retry always differs (new timing at minimum).
        seen_logs: set = set()

        # ------------------------------------------------------------------
        # LATENCY: answer-first streaming.
        #
        # Previously the browser saw nothing until the whole graph finished,
        # which meant the user waited through the critic's full LLM call
        # (a 120b model, purely to produce a score) before a single character
        # of the answer appeared. The critic does not rewrite the answer — it
        # scores it, and only sends it back when the score is under threshold.
        # So for the overwhelming majority of turns the answer was already
        # final and simply being withheld.
        #
        # Now: supervisor tokens are decoded and pushed as they arrive
        # (`answer_delta`), the complete answer is confirmed the moment the
        # supervisor node ends (`answer`), and the critic's verdict follows
        # separately (`critic`) to be attached to the turn. `complete` still
        # closes the stream with the authoritative payload including any late
        # background scrapes.
        #
        # If the critic scores low and routes back, the supervisor produces a
        # NEW answer, the streamer resets, and a fresh `answer` event replaces
        # what the user is reading. That replacement is the rare path; the
        # common path is that the text they started reading immediately is the
        # text they end up with.
        # ------------------------------------------------------------------
        answer_streamer = PartialJSONFieldStreamer("final_response")
        # Safety net: the module's own design says streamed deltas are "never
        # the source of truth" and get overwritten by graph state at the end
        # — correct when streaming fails but the state has the real answer.
        # This exists for the INVERSE failure: streaming succeeds (the
        # person sees a complete, well-formed answer arrive token by token)
        # but final_state["final_response"] comes back empty anyway — e.g.
        # the same JSON that streamed cleanly fails strict json.loads() at
        # the end over one unescaped character in a code block or Mermaid
        # diagram. Without this, that real, already-displayed answer gets
        # thrown away and replaced with a generic "workflow terminated"
        # message — discarding correct content in favour of a worse one.
        streamed_answer_text: list[str] = []
        streamed_any = False   # did token streaming actually work this turn?

        # ------------------------------------------------------------------
        # LATENCY (perceived): live per-source cards during multi-scrape fanout.
        #
        # When the supervisor dispatches several scraper tasks via Send(),
        # they run in parallel but the graph still waits for all of them
        # before the supervisor synthesizes an answer — one slow site holds
        # up a chat bubble that otherwise has nothing to show. Each scraper's
        # own result is known the moment THAT scraper finishes, well before
        # the others; this surfaces it immediately as a small metadata +
        # snippet card instead of leaving the user staring at "thinking…"
        # until the slowest source lands.
        #
        # Deliberately NOT an extra LLM call: the snippet is a plain string
        # slice of content the scraper already returned (see _snippet()
        # above). An LLM-summarized card would look nicer but would compete
        # with the supervisor/critic for the same Ollama instance and add
        # real latency to chase perceived latency — the opposite of the
        # point. If this weren't free, per the ask, it would not be worth
        # doing; it does not cost anything so it stays.
        #
        # Gated on "reasonable given the query": only fires when the
        # supervisor's OWN decision fanned out more than one scraper task in
        # the same loop. A single-URL scrape shows nothing new — waiting for
        # one source is the same wait either way, and a lone card would be
        # noise, not progress. Re-evaluated every time the supervisor makes
        # a fresh decision, since a later loop may or may not repeat a
        # multi-scrape fanout.
        # ------------------------------------------------------------------
        scrape_fanout_active = False

        try:
            async for event in app_graph.astream_events(
                initial_state, version="v2",
                config=run_config(conversation_id, user_id)):
                kind = event["event"]
                name = event["name"]
                ui_message = None

                # 1. Map native LangGraph events to human-readable UI logs
                if kind == "on_chain_start" and name == "supervisor":
                    # Fresh JSON object incoming — drop any partial state from
                    # a previous planning loop. streamed_answer_text mirrors
                    # this reset for the same reason: a stale answer from an
                    # earlier, superseded planning loop must not survive as
                    # the fallback if this newer loop is the one that ends
                    # up with an empty final_response.
                    answer_streamer.reset()
                    streamed_answer_text = []
                    yield "data: " + json.dumps({"type": "answer_reset"}) + "\n\n"
                    ui_message = "🧠 Supervisor is evaluating the context..."
                elif kind == "on_chain_start" and name in AGENT_REGISTRY:
                    ui_message = f"🔀 Dispatching worker: {name}..."
                elif kind == "on_chain_start" and name in ("critic", "critic_node"):
                    ui_message = "⚖️ Critic is evaluating the final response..."
                elif kind == "on_chat_model_start":
                    ui_message = "⏳ Generating reasoning..."

                if ui_message:
                    yield emit(ui_message)

                # 1b. TOKEN STREAMING — decode the supervisor's answer field as
                #     it is written. ChatOllama.invoke() streams internally and
                #     fires on_llm_new_token, so these events arrive even though
                #     supervisor_node calls .invoke() and not .astream().
                #     Scoped to the supervisor node specifically: the critic's
                #     tokens are its JSON verdict, not user-facing prose.
                if kind == "on_chat_model_stream":
                    node = (event.get("metadata") or {}).get("langgraph_node")
                    if node == "supervisor":
                        chunk = event["data"].get("chunk")
                        piece = getattr(chunk, "content", "") or ""
                        if piece:
                            delta = answer_streamer.feed(piece)
                            if delta:
                                streamed_any = True
                                streamed_answer_text.append(delta)
                                yield "data: " + json.dumps({
                                    "type": "answer_delta", "text": delta}) + "\n\n"

                # 1c. Re-check the fanout gate every time the supervisor
                #     finishes deciding — BEFORE any of this loop's scraper
                #     results can possibly land, since those workers haven't
                #     been dispatched yet at this point in the event stream.
                if kind == "on_chain_end" and name == "supervisor":
                    sup_out = event["data"].get("output")
                    if isinstance(sup_out, dict):
                        pending = sup_out.get("pending_tasks") or []
                        scraper_task_count = sum(
                            1 for t in pending if t.get("agent") == "scraper")
                        scrape_fanout_active = scraper_task_count >= 2

                        # Second, more direct source of final_response,
                        # alongside the "LangGraph" root capture below. That
                        # capture has proven unreliable — repeatedly empty
                        # even on a clean, error-free, single-pass run with
                        # a passing critic score — for a reason not yet
                        # isolated. supervisor_node's OWN return dict is
                        # where final_response actually originates (route to
                        # critic with no tasks IS the node saying "this is my
                        # answer"), so capturing it here sidesteps whatever
                        # is going wrong in the aggregate capture, rather
                        # than continuing to depend on the streamed-token
                        # recovery fallback to paper over it every time.
                        if sup_out.get("final_response"):
                            supervisor_final_response = sup_out["final_response"]

                # 2. Stream the rich action_logs a node returned
                if kind == "on_chain_end":
                    node_output = event["data"].get("output")
                    if isinstance(node_output, dict) and node_output.get("action_logs"):
                        for log in node_output["action_logs"]:
                            if log in seen_logs:
                                continue
                            seen_logs.add(log)
                            yield emit(log)
                    # Mid-run doc_retriever dispatches can also surface a
                    # graph trace, same ephemeral/on-demand contract as the
                    # pre-pass's own graph_trace event above.
                    if isinstance(node_output, dict):
                        for trace in node_output.get("graph_traces") or []:
                            yield "data: " + json.dumps({
                                "type": "graph_trace", **trace}) + "\n\n"

                # 2a. LIVE SOURCE CARD — one scraper task in an active fanout
                #     just finished. Emit its card the instant it lands, not
                #     when the whole batch does.
                if kind == "on_chain_end" and name == "scraper" and scrape_fanout_active:
                    node_output = event["data"].get("output")
                    if isinstance(node_output, dict):
                        for entry in node_output.get("context", []):
                            card = _source_card(entry)
                            if card:
                                yield "data: " + json.dumps({
                                    "type": "source_card", **card}) + "\n\n"

                # 2b. ANSWER CONFIRMED — the supervisor has finished and
                #     produced a final answer with nothing left to dispatch.
                #     Send the complete text now rather than after the critic.
                #     This is also the safety net for token streaming: if the
                #     model didn't stream (or the field never matched), this is
                #     the first the browser hears of the answer, and it renders
                #     in one go exactly as it used to.
                if kind == "on_chain_end" and name == "supervisor":
                    out = event["data"].get("output")
                    if isinstance(out, dict):
                        answer = out.get("final_response")
                        if answer and not out.get("pending_tasks"):
                            yield "data: " + json.dumps({
                                "type": "answer",
                                "text": answer,
                                "streamed": streamed_any,
                            }) + "\n\n"

                # 2c. CRITIC VERDICT — emitted as its own event so the UI can
                #     attach the score to a turn the user is already reading.
                if kind == "on_chain_end" and name in ("critic", "critic_node"):
                    out = event["data"].get("output")
                    if isinstance(out, dict) and out.get("feedback") is not None:
                        yield "data: " + json.dumps({
                            "type": "critic",
                            "score": out.get("eval_score"),
                            "feedback": out.get("feedback"),
                        }) + "\n\n"

                # 3. Capture final state but do NOT emit "complete" yet — there
                #    may still be background renders to fold in.
                if kind == "on_chain_end" and name == "LangGraph":
                    out = event["data"].get("output", {})
                    if isinstance(out, dict):
                        final_state = out
        except Exception as e:
            print(f"\n🔥 GRAPH EXECUTION FAILED: {e}")
            # workflow.llm.ModelChainError carries one line per attempted model.
            # Without draining them the trace shows N identical "generating..."
            # lines and a single error, with no sign that N DIFFERENT models were
            # tried or why each one failed — which is what made ordinary failover
            # look like the models were mixing at random.
            for line in getattr(e, "logs", []):
                yield emit(line)
            yield emit(f"🔥 Graph execution failed: {type(e).__name__}: {e}")

        # Read BEFORE section 4, whose `finally` calls inflight.cleanup(run_id)
        # and deletes the run entry — read it after that and it is always False.
        stopped_by_user = inflight.stop_requested(run_id)
        if stopped_by_user:
            yield emit("⏹️ Finished early at your request — answering from the "
                       "context gathered so far. The critic did not run.")

        final_response, best_score, selection_source = _best_attempt(final_state)
        if selection_source != "final_state":
            msg = f"🏆 Selected the best-scoring attempt: {selection_source}."
            print(f"[STREAM] {msg}")
            yield emit(msg)
        context = list(final_state.get("context") or doc_blocks)

        late_addendum = ""

        # ---- 4. Post-graph: fold in deferred background renders ----
        try:
            pending = inflight.pending_urls(run_id)
            if pending:
                yield emit(f"🕐 Answer ready. Waiting up to "
                           f"{int(cfg.scrape_late_wait_seconds)}s for "
                           f"{len(pending)} slow source(s) to finish rendering...")
                late = await inflight.wait_all(
                    run_id, timeout=cfg.scrape_late_wait_seconds)
            else:
                late = inflight.drain(run_id)

            if late:
                context.extend(str(x) for x in late)
                # HELD, NOT APPENDED. Appending here is what lost the answer:
                # `final_response` can legitimately be None at this point (the
                # "LangGraph" root capture comes back empty even on clean
                # single-pass runs — see the note at the supervisor capture
                # above), and `(final_response or "") + addendum` made it
                # TRUTHY. The `if not final_response:` recovery below then
                # never fired, so supervisor_final_response and the streamed
                # token buffer — both written specifically to rescue this case
                # — were skipped, and the turn rendered as nothing but a "Late
                # Additions" block. The addendum is appended at the very end
                # instead, once a real answer has actually been resolved.
                late_addendum = _addendum(late, context)
                if late_addendum:
                    yield emit(f"📥 {len(late)} late result(s) folded into the answer.")
                else:
                    yield emit(f"🗑️ {len(late)} late result(s) added nothing "
                               f"new and were not appended.")

            still = inflight.pending_urls(run_id)
            if still:
                yield emit(f"⚠️ {len(still)} source(s) did not finish in time "
                           f"and were dropped.")
        except Exception as e:
            print(f"[STREAM] ⚠️ Late-result handling failed: {e}")
        finally:
            inflight.cleanup(run_id)

        if not final_response:
            if supervisor_final_response:
                # The direct capture, not a reconstruction — strictly better
                # than the streamed-text recovery below when available, since
                # this IS the exact value the node returned, not text
                # rebuilt from individual tokens.
                final_response = supervisor_final_response
                print("[STREAM] ⚠️ final_state had no final_response; used "
                      "the direct supervisor-node capture instead.")
            else:
                recovered = "".join(streamed_answer_text).strip()
                if recovered:
                    # The graph's own state came back empty, but a complete
                    # answer was already streamed to and shown for this exact
                    # turn — almost always because the same JSON that streamed
                    # cleanly failed strict validation at the end (one unescaped
                    # character in a code block or diagram is enough). Losing
                    # already-correct, already-displayed content to a generic
                    # error message would be a strictly worse outcome than
                    # using it, so it becomes the answer instead of being
                    # discarded.
                    final_response = recovered
                    print("[STREAM] ⚠️ final_state had no final_response; "
                          "recovered it from the streamed token buffer instead.")
                elif stopped_by_user:
                    final_response = _stopped_without_answer(context)
                else:
                    final_response = (
                        "⚠️ **Workflow terminated** without producing a final answer. "
                        "The execution trace and retrieved evidence have the detail.")

        # Explicit low-confidence signal: route_from_critic() only ever exits
        # without looping again for three reasons — the score passed, the
        # loop cap was hit, or the score plateaued (workflow/routing.py).
        # A final eval_score still below threshold here means it was one of
        # the latter two, not the first — the answer above is the best the
        # system produced, but it was never actually approved. Silently
        # returning it looking identical to a passing answer would hide
        # that distinction from the person reading it.
        # The SELECTED attempt's score, not the last loop's — those differ
        # whenever _best_attempt() picked an earlier answer, and the
        # disclaimer must describe the answer actually being shown.
        final_eval_score = best_score if best_score is not None else final_state.get("eval_score", 0)
        if (final_eval_score and final_eval_score < cfg.critic_pass_threshold
                and "Workflow terminated" not in final_response):
            final_response = (
                f"⚠️ *Low confidence (critic score {final_eval_score}/100 — "
                f"stopped without reaching the {cfg.critic_pass_threshold} "
                f"threshold, either from repeated attempts or a plateaued score). "
                f"Treat the answer below as a best effort, not a verified one.*\n\n"
                f"{final_response}")

        # The low-confidence banner above is gated on a truthy eval_score, so a
        # stopped run (critic never ran, score 0) correctly skips it — but it
        # still needs its own marker. An answer that no critic reviewed must
        # not be presented as if one had.
        if stopped_by_user and not final_response.startswith("⏹️"):
            final_response = (
                "⏹️ *Finished early at your request. The critic did not review "
                "this answer, and any queued research was dropped.*\n\n"
                f"{final_response}")

        # Appended LAST, after every recovery path and banner has had its say,
        # so it can only ever supplement a resolved answer — never stand in for
        # one. The guard is belt-and-braces: if nothing recovered an answer,
        # a "Late Additions" appendix on its own is worse than the explicit
        # failure message it would be hiding.
        if late_addendum and final_response and final_response.strip():
            final_response += late_addendum

        feedback = final_state.get("feedback")
        action_logs = list(final_state.get("action_logs") or collected_logs)
        if stopped_by_user:
            # Persisted with the turn so a reloaded thread — and the telemetry
            # rollups — can tell a user-stopped run from a passing one. A
            # dedicated column would be cleaner; action_logs is already a JSON
            # list and needs no migration.
            action_logs.append("⏹️ Finished early at the user's request (critic skipped).")

        message_id = store.add_message(
            conversation_id, "assistant", final_response,
            context=context, action_logs=action_logs, feedback=feedback)

        print("🎉 WORKFLOW EXECUTION COMPLETED.")

        yield "data: " + json.dumps({
            "type": "complete",
            "final_response": final_response,
            "stopped_by_user": stopped_by_user,
            "feedback": feedback,
            "context": context,
            "action_logs": action_logs,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "title": new_title,
        }) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/models")
async def list_models():
    """Selectable models plus this deployment's defaults.

    The browser persists a choice in localStorage and sends it with every
    turn; this endpoint is what lets it render labels and detect that a stored
    choice is no longer offered. Defaults are returned rather than hardcoded
    client-side so changing .env changes the UI without a frontend edit.
    """
    return llm_select.catalogue()


@app.post("/chat/stop")
async def chat_stop(run_id: str = Form(...)):
    """Ask an in-flight run to finish at its next routing decision.

    Deliberately not a 404 on an unknown id. The race is normal, not
    exceptional: the turn can finish in the moment between the user deciding
    to press the button and the request landing. `ok: false` says so without
    lighting up an error in the browser console.

    Nothing is cancelled here. The current worker runs to completion, the
    supervisor gets one synthesis pass, and the graph ends before the critic
    — see workflow/routing.py. A hard mid-worker abort is a separate,
    harder feature; this one cannot leave the graph in a partial state.
    """
    ok = inflight.request_stop(run_id)
    return {"ok": ok, "run_id": run_id,
            "detail": "Stop requested." if ok else
                      "No such active run — it has probably already finished."}


@app.post("/chat")
async def chat_once(prompt: str = Form(...), conversation_id: str = Form(None),
                    user_id: str = Form(None)):
    """Non-streaming single-shot turn. Kept for scripted/CLI callers.

    Returns JSON rather than HTML — the UI uses /chat/stream, so rendering a
    template here would only duplicate the transcript logic.
    """
    user_id = user_id or "anonymous"
    if not conversation_id or not store.get_conversation(conversation_id):
        conversation_id = store.create_conversation()["id"]

    session.set_conversation(conversation_id)
    run_id = inflight.set_run()

    docs = [d for d in store.list_documents(conversation_id)
            if d["status"] in ("indexed", "degraded")]
    attached = [d["file_name"] for d in docs]
    history = store.recent_turns(conversation_id, cfg.chat_history_turns)

    store.add_message(conversation_id, "user", prompt,
                      attachments=[{"doc_id": d["id"], "file_name": d["file_name"]}
                                   for d in docs])
    store.autotitle_if_default(conversation_id, prompt)

    doc_blocks: list = []
    if attached:
        try:
            doc_blocks, _hits, _graph_trace = await context_for_query(conversation_id, prompt)
        except Exception as e:
            print(f"[CHAT] ⚠️ Document search failed: {e}")

    try:
        final_state = await app_graph.ainvoke(
            _initial_state(prompt, conversation_id, doc_blocks, attached, history),
            config=run_config(conversation_id, user_id))
        # Same selection as the streaming endpoint. This one matters more,
        # not less: there is no frontend guard on this path at all, so a
        # worse final loop would be the only answer the caller ever sees.
        final_output, _best_score, selection_source = _best_attempt(final_state)
        if selection_source != "final_state":
            print(f"[CHAT] 🏆 Selected the best-scoring attempt: {selection_source}.")
        context = list(final_state.get("context") or doc_blocks)

        if inflight.pending_count(run_id):
            late = await inflight.wait_all(run_id, timeout=cfg.scrape_late_wait_seconds)
        else:
            late = inflight.drain(run_id)
        if late:
            final_output = (final_output or "") + _addendum(late)
            context.extend(str(x) for x in late)
    except Exception as e:
        print(f"\n🔥 GRAPH EXECUTION FAILED: {e}")
        # Same as the streaming path: surface the per-model attempt log before
        # the exception swallows it. There is no SSE channel here, so it goes to
        # the server log rather than the client.
        for line in getattr(e, "logs", []):
            print(f"[LLM] {line}")
        raise HTTPException(500, f"Workflow execution failed: {e}") from e
    finally:
        inflight.cleanup(run_id)

    if not final_output:
        final_output = "⚠️ Workflow terminated without a final answer."

    message_id = store.add_message(
        conversation_id, "assistant", final_output, context=context,
        action_logs=list(final_state.get("action_logs") or []),
        feedback=final_state.get("feedback"))

    return {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "final_response": final_output,
        "feedback": final_state.get("feedback"),
        "context": context,
    }


@app.get("/api/downloads")
async def list_downloads():
    """Returns a list of all downloaded files."""
    if not os.path.exists(DOWNLOAD_DIR):
        return {"files": []}

    files = []
    for filename in os.listdir(DOWNLOAD_DIR):
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.isfile(filepath):
            safe_name = urllib.parse.quote(filename)
            files.append({"name": filename, "url": f"/static/downloads/{safe_name}"})
    return {"files": files}


@app.delete("/api/downloads/{filename}")
async def delete_download(filename: str):
    """Deletes a specific file from the server."""
    safe_filename = urllib.parse.unquote(filename)
    filepath = os.path.join(DOWNLOAD_DIR, safe_filename)

    # Security check to prevent directory traversal attacks
    if not os.path.abspath(filepath).startswith(os.path.abspath(DOWNLOAD_DIR)):
        raise HTTPException(status_code=403, detail="Unauthorized file path")

    if os.path.exists(filepath):
        os.remove(filepath)
        return {"status": "success", "message": f"{safe_filename} deleted."}

    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    uvicorn.run(app, host=cfg.app_host, port=cfg.app_port)
