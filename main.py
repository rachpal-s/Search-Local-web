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
from docstore import session, store
from docstore.retrieve import context_for_query
from routers import conversations, jobs as jobs_router, uploads
from jobs import store as jobstore
from workflow import app_graph, inflight
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


@app.on_event("startup")
async def _startup() -> None:
    """Create the chat schema and the directories the app writes into.

    Idempotent, so it is safe on every reload. Doing it here rather than at
    import time keeps `import main` side-effect-free for tests.
    """
    store.init_db()
    jobstore.init_db()
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


def _addendum(late: list) -> str:
    """Render late background results as an appendix to the final answer.

    Deliberately dumb: it appends rather than trying to re-open and rewrite the
    answer the critic already approved. If you want the late material woven in
    properly, that is a second supervisor+critic delta pass, not string surgery.
    """
    if not late:
        return ""
    lines = []
    for entry in late:
        text = entry if isinstance(entry, str) else json.dumps(entry)
        payload = None
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
        if payload and payload.get("word_count"):
            title = payload.get("title") or payload.get("source", "Source")
            src = payload.get("source", "")
            wc = payload.get("word_count")
            snippet = (payload.get("scraped_content") or "")[:400].strip()
            lines.append(f"**[{title}]({src})** — {wc} words\n\n{snippet}...")
        else:
            lines.append(f"- {text[:300]}")
    body = "\n\n".join(lines)
    return (
        "\n\n---\n\n### 🕐 Late Additions\n\n"
        "_These sources were still rendering when the answer above was finalised._\n\n"
        f"{body}"
    )


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
                      conversation_id: str = Form(None)):
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
    """
    if not conversation_id or not store.get_conversation(conversation_id):
        conversation_id = store.create_conversation()["id"]

    # Binds the conversation for every node inside the graph via ContextVar —
    # worker payloads are authored by the LLM and cannot be trusted to carry it.
    session.set_conversation(conversation_id)
    run_id = inflight.set_run()

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
                doc_blocks, hits = await context_for_query(conversation_id, prompt)
                if hits:
                    files = sorted({h["file_name"] for h in hits})
                    yield emit(f"📚 {len(hits)} passage(s) from "
                               f"{', '.join(files)} pulled into context.")
                else:
                    yield emit("📚 No matching passages in the attachments.")
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
            async for event in app_graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event["name"]
                ui_message = None

                # 1. Map native LangGraph events to human-readable UI logs
                if kind == "on_chain_start" and name == "supervisor":
                    # Fresh JSON object incoming — drop any partial state from
                    # a previous planning loop.
                    answer_streamer.reset()
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

                # 2. Stream the rich action_logs a node returned
                if kind == "on_chain_end":
                    node_output = event["data"].get("output")
                    if isinstance(node_output, dict) and node_output.get("action_logs"):
                        for log in node_output["action_logs"]:
                            if log in seen_logs:
                                continue
                            seen_logs.add(log)
                            yield emit(log)

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
            yield emit(f"🔥 Graph execution failed: {type(e).__name__}: {e}")

        final_response = final_state.get("final_response")
        context = list(final_state.get("context") or doc_blocks)

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
                yield emit(f"📥 {len(late)} late result(s) folded into the answer.")
                final_response = (final_response or "") + _addendum(late)
                context.extend(str(x) for x in late)

            still = inflight.pending_urls(run_id)
            if still:
                yield emit(f"⚠️ {len(still)} source(s) did not finish in time "
                           f"and were dropped.")
        except Exception as e:
            print(f"[STREAM] ⚠️ Late-result handling failed: {e}")
        finally:
            inflight.cleanup(run_id)

        if not final_response:
            final_response = (
                "⚠️ **Workflow terminated** without producing a final answer. "
                "The execution trace and retrieved evidence have the detail.")

        feedback = final_state.get("feedback")
        action_logs = list(final_state.get("action_logs") or collected_logs)

        message_id = store.add_message(
            conversation_id, "assistant", final_response,
            context=context, action_logs=action_logs, feedback=feedback)

        print("🎉 WORKFLOW EXECUTION COMPLETED.")

        yield "data: " + json.dumps({
            "type": "complete",
            "final_response": final_response,
            "feedback": feedback,
            "context": context,
            "action_logs": action_logs,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "title": new_title,
        }) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat")
async def chat_once(prompt: str = Form(...), conversation_id: str = Form(None)):
    """Non-streaming single-shot turn. Kept for scripted/CLI callers.

    Returns JSON rather than HTML — the UI uses /chat/stream, so rendering a
    template here would only duplicate the transcript logic.
    """
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
            doc_blocks, _hits = await context_for_query(conversation_id, prompt)
        except Exception as e:
            print(f"[CHAT] ⚠️ Document search failed: {e}")

    try:
        final_state = await app_graph.ainvoke(
            _initial_state(prompt, conversation_id, doc_blocks, attached, history))
        final_output = final_state.get("final_response")
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
    uvicorn.run(app, host=cfg.app_host, port=9001)
