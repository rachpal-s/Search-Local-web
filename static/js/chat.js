/* ============================================================================
   chat.js — client for the agent console.

   Responsibilities, in dependency order:
     1. thread lifecycle   list / create / select / rename / delete
     2. attachments        upload, then poll ingestion status per file
     3. turn submission    POST /chat/stream, parse SSE, render the turn
     4. provenance         show which agents ran and which documents were cited

   No framework on purpose. The server owns all state; this file owns the DOM.
   ========================================================================== */

(() => {
  "use strict";

  // ── element handles ──────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const el = {
    shell: $("shell"), rail: $("rail"), railOpen: $("rail-open"),
    railClose: $("rail-close"), newChat: $("new-chat"),
    newChatBtn: $("new-chat-btn"),
    convo: $("convo"),
    threadSearch: $("thread-search"), threadList: $("thread-list"),
    convoTitle: $("convo-title"), renameBtn: $("rename-btn"),
    deleteBtn: $("delete-btn"), corpusBadge: $("corpus-badge"),
    transcript: $("transcript"), welcome: $("welcome"), starters: $("starters"),
    composer: $("composer"), composerWrap: document.querySelector(".composer-wrap"),
    prompt: $("prompt"), sendBtn: $("send-btn"), attachBtn: $("attach-btn"), pasteBtn: $("paste-btn"),
    stopBtn: $("stop-btn"),
    fileInput: $("file-input"), attachments: $("attachments"),
    trace: $("trace"), traceToggle: $("trace-toggle"), traceClose: $("trace-close"),
    traceStream: $("trace-stream"), tracePulse: $("trace-pulse"),
    traceLangsmithLink: $("trace-langsmith-link"),
    traceFeedback: $("trace-feedback"), traceContext: $("trace-context"),
    traceDocs: $("trace-docs"),
    collectionsBtn: $("collections-btn"), collectionsSheet: $("collections-sheet"),
    collectionsClose: $("collections-close"), collectionsBody: $("collections-body"),
    collectionsCount: $("collections-count"),
    collSearchInput: $("coll-search-input"), collSearchBtn: $("coll-search-btn"),
    collSearchResults: $("coll-search-results"),
    scopeBanner: $("scope-banner"), scopeBannerText: $("scope-banner-text"),
    scopeBannerBtn: $("scope-banner-btn"), scopeBannerClose: $("scope-banner-close"),
    downloadsBtn: $("downloads-btn"), downloadsSheet: $("downloads-sheet"),
    downloadsClose: $("downloads-close"), downloadsBody: $("downloads-body"),
    assetSearch: $("asset-search"), scrim: $("scrim"), toast: $("toast"),
  };
  if (navigator.clipboard && navigator.clipboard.read) {
    el.pasteBtn.hidden = false;
  }

  // ── state ────────────────────────────────────────────────────────────
  // Mock user identity, ahead of real auth. Generated once per browser and
  // persisted in localStorage — not tied to any account, just enough for
  // traces to be distinguishable per session today. When real login exists,
  // this is a one-line swap (read the authenticated id instead) and nothing
  // else in the tracing plumbing changes, since the backend already treats
  // user_id as an opaque string.
  function getOrCreateUserId() {
    const KEY = "chat_pseudo_user_id";
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = "u_" + (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}_${Math.random()}`);
      localStorage.setItem(KEY, id);
    }
    return id;
  }

  const state = {
    conversationId: null,
    userId: getOrCreateUserId(),
    threads: [],
    docs: [],              // documents attached to the active thread
    collections: [],       // knowledge collections this thread may also search
    availableCollections: [],  // every collection that exists, for the scope banner
    dismissedScopeBanner: new Set(),  // thread ids where the banner was closed this session
    busy: false,           // a turn is in flight
    runId: null,           // server run id for the in-flight turn ("Finish now")
    pollTimer: null,
    logs: [],
    seenAgents: new Set(),
  };

  const SETTLED = new Set(["indexed", "failed", "quarantined", "degraded"]);
  const MAX_TRACE_LINES = 400;

  // ── small utilities ──────────────────────────────────────────────────

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const md = (s) => {
    try { return marked.parse(String(s ?? "")); }
    catch { return `<p>${esc(s)}</p>`; }
  };

  function toast(message, kind = "info", ms = 3200) {
    el.toast.textContent = message;
    el.toast.dataset.kind = kind;
    el.toast.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { el.toast.hidden = true; }, ms);
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* keep */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.status === 204 ? null : res.json();
  }

  const bytes = (n) => {
    if (!n) return "0 B";
    const u = ["B", "KB", "MB", "GB"];
    const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), u.length - 1);
    return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
  };

  function dayBucket(iso) {
    const d = new Date(iso);
    if (Number.isNaN(+d)) return "Earlier";
    const today = new Date(); today.setHours(0, 0, 0, 0);
    const diff = Math.floor((today - new Date(d).setHours(0, 0, 0, 0)) / 86400000);
    if (diff <= 0) return "Today";
    if (diff === 1) return "Yesterday";
    if (diff < 7) return "Previous 7 days";
    if (diff < 30) return "Previous 30 days";
    return "Earlier";
  }

  // ── panes ────────────────────────────────────────────────────────────

  const isNarrow = () => window.matchMedia("(max-width: 860px)").matches;

  function setRail(open) {
    el.shell.dataset.rail = open ? "open" : "closed";
  }
  function setTrace(open) {
    el.shell.dataset.trace = open ? "open" : "closed";
    el.trace.hidden = !open;
    el.traceToggle.setAttribute("aria-pressed", String(open));
  }

  el.railOpen.addEventListener("click", () => setRail(true));
  el.railClose.addEventListener("click", () => setRail(false));
  el.traceToggle.addEventListener("click", () =>
    setTrace(el.shell.dataset.trace !== "open"));
  el.traceClose.addEventListener("click", () => setTrace(false));

  // ── trace panel ──────────────────────────────────────────────────────

  function traceReset() {
    state.logs = [];
    el.traceStream.innerHTML = '<p class="trace-idle">Starting graph…</p>';
  }

  function traceLog(message) {
    state.logs.push(message);
    if (state.logs.length > MAX_TRACE_LINES) state.logs.shift();
    const span = document.createElement("span");
    span.className = "trace-line";
    span.textContent = message;
    if (el.traceStream.querySelector(".trace-idle")) el.traceStream.innerHTML = "";
    el.traceStream.appendChild(span);
    el.traceStream.scrollTop = el.traceStream.scrollHeight;
  }

  function renderTraceContext(context) {
    if (!context || !context.length) {
      el.traceContext.innerHTML = '<p class="muted">No worker payloads captured.</p>';
      return;
    }
    el.traceContext.innerHTML = context
      .map((c) => `<div class="ctx-item">${esc(typeof c === "string" ? c : JSON.stringify(c, null, 2))}</div>`)
      .join("");
  }

  function renderTraceDocs() {
    if (!state.docs.length) {
      el.traceDocs.innerHTML = '<p class="muted">Nothing attached to this chat.</p>';
      return;
    }
    el.traceDocs.innerHTML = state.docs.map((d) => {
      const cls = (d.data_classification || "internal").toLowerCase();
      const meta = d.status === "indexed"
        ? `${d.chunk_count} chunks`
        : esc(d.reason_code || d.status);
      return `<div class="doc-row">
        <span class="doc-name" title="${esc(d.file_name)}">${esc(d.file_name)}</span>
        <span class="tag ${esc(cls)}">${esc(cls)}</span>
        <span class="doc-meta">${meta}</span>
      </div>`;
    }).join("");
  }

  // ── thread rail ──────────────────────────────────────────────────────

  async function loadThreads(query = "") {
    try {
      const q = query ? `?q=${encodeURIComponent(query)}` : "";
      const data = await api(`/api/conversations${q}`);
      state.threads = data.conversations || [];
      renderThreads();
    } catch (e) {
      el.threadList.innerHTML = `<p class="rail-empty">Couldn't load chats. ${esc(e.message)}</p>`;
    }
  }

  function renderThreads() {
    if (!state.threads.length) {
      el.threadList.innerHTML = '<p class="rail-empty">No chats yet.</p>';
      return;
    }
    let html = "";
    let bucket = null;
    for (const t of state.threads) {
      const b = dayBucket(t.updated_at);
      if (b !== bucket) { html += `<div class="thread-day">${b}</div>`; bucket = b; }
      const active = t.id === state.conversationId;
      const clip = t.doc_count ? `<span class="thread-clip">${t.doc_count}📎</span>` : "";
      html += `<button class="thread" data-id="${esc(t.id)}"
                 aria-current="${active}" title="${esc(t.title)}">
                 <span class="thread-name">${esc(t.title || "New chat")}</span>${clip}
               </button>`;
    }
    el.threadList.innerHTML = html;
  }

  el.threadList.addEventListener("click", (e) => {
    const btn = e.target.closest(".thread");
    if (!btn) return;
    selectThread(btn.dataset.id);
    if (isNarrow()) setRail(false);
  });

  let searchTimer;
  el.threadSearch.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadThreads(el.threadSearch.value.trim()), 180);
  });

  // ── thread lifecycle ─────────────────────────────────────────────────

  async function newChat() {
    try {
      const conv = await api("/api/conversations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New chat" }),
      });
      await loadThreads();
      await selectThread(conv.id);
      el.prompt.focus();
    } catch (e) {
      toast(`Couldn't start a chat: ${e.message}`, "error");
    }
  }

  async function updateTraceLink(conversationId) {
    if (!conversationId) { el.traceLangsmithLink.hidden = true; return; }
    try {
      const { url } = await api(`/api/conversations/${conversationId}/trace-url`);
      if (url) {
        el.traceLangsmithLink.href = url;
        el.traceLangsmithLink.hidden = false;
      } else {
        el.traceLangsmithLink.hidden = true;   // telemetry disabled server-side
      }
    } catch {
      el.traceLangsmithLink.hidden = true;   // decorative; a failure here is silent
    }
  }

  async function selectThread(id) {
    if (state.busy) { toast("Wait for the current answer to finish."); return; }
    stopPolling();
    try {
      const data = await api(`/api/conversations/${id}`);
      state.conversationId = id;
      state.docs = data.documents || [];
      location.hash = id;
      updateTraceLink(id);

      el.convoTitle.textContent = data.conversation.title || "New chat";
      renderCorpusBadge(data.corpus);
      renderThreads();
      renderTranscript(data.messages || []);
      renderAttachmentChips();
      renderTraceDocs();
      renderTraceContext(null);
      el.traceFeedback.innerHTML = '<p class="muted">No evaluation yet.</p>';
      el.traceStream.innerHTML = '<p class="trace-idle">Idle. Send a message to watch the graph run.</p>';
      if (state.docs.some((d) => !SETTLED.has(d.status))) startPolling();
      // Attached collections are per-thread, so refresh the rail badge on open.
      api(`/api/conversations/${id}/collections`)
        .then((c) => {
          state.collections = c.attached || [];
          state.availableCollections = c.available || [];
          updateCollectionsCount();
          updateScopeBanner();
        })
        .catch(() => { /* the badge is decorative; a failure is not worth a toast */ });
    } catch (e) {
      toast(`Couldn't open that chat: ${e.message}`, "error");
    }
  }

  function renderCorpusBadge(corpus) {
    const n = corpus?.documents || 0;
    if (!n) { el.corpusBadge.hidden = true; return; }
    el.corpusBadge.hidden = false;
    el.corpusBadge.textContent =
      `📎 ${n} doc${n > 1 ? "s" : ""} · ${corpus.chunks} chunks`;
  }

  el.newChat.addEventListener("click", newChat);
  el.newChatBtn.addEventListener("click", newChat);
  el.renameBtn.addEventListener("click", () => {
    if (!state.conversationId) return;
    el.convoTitle.contentEditable = "true";
    el.convoTitle.focus();
    document.getSelection().selectAllChildren(el.convoTitle);
  });

  el.convoTitle.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); el.convoTitle.blur(); }
    if (e.key === "Escape") { el.convoTitle.blur(); }
  });

  el.convoTitle.addEventListener("blur", async () => {
    if (el.convoTitle.contentEditable !== "true") return;
    el.convoTitle.contentEditable = "false";
    const title = el.convoTitle.textContent.trim().slice(0, 500);
    if (!title || !state.conversationId) return;
    try {
      await api(`/api/conversations/${state.conversationId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      await loadThreads(el.threadSearch.value.trim());
    } catch (e) {
      toast(`Rename failed: ${e.message}`, "error");
    }
  });

  el.deleteBtn.addEventListener("click", async () => {
    if (!state.conversationId) return;
    const name = el.convoTitle.textContent.trim() || "this chat";
    if (!confirm(`Delete "${name}"? Its messages and attached documents go with it.`)) return;
    try {
      await api(`/api/conversations/${state.conversationId}`, { method: "DELETE" });
      state.conversationId = null;
      await loadThreads();
      if (state.threads.length) await selectThread(state.threads[0].id);
      else await newChat();
      toast("Chat deleted.");
    } catch (e) {
      toast(`Delete failed: ${e.message}`, "error");
    }
  });

  // ── transcript rendering ─────────────────────────────────────────────

  function renderTranscript(messages) {
    el.transcript.innerHTML = "";
    el.convo.dataset.hasMessages = messages.length ? "true" : "false";
    if (!messages.length) {
      el.transcript.appendChild(el.welcome);
      el.welcome.hidden = false;
      return;
    }
    el.welcome.hidden = true;
    for (const m of messages) {
      if (m.role === "user") {
        el.transcript.appendChild(userTurn(m.content, m.attachments));
      } else {
        el.transcript.appendChild(assistantTurn({
          content: m.content,
          context: m.context,
          actionLogs: m.action_logs,
          feedback: m.feedback,
        }));
      }
    }
    scrollToEnd();
  }

  function userTurn(content, attachments) {
    const turn = document.createElement("article");
    turn.className = "turn user";
    const files = (attachments || []).length
      ? `<div class="turn-files">${attachments.map((a) =>
        `<span class="turn-file">📎 ${esc(a.file_name || a.name || "file")}</span>`).join("")}</div>`
      : "";
    turn.innerHTML = `<div><div class="bubble"></div>${files}</div>`;
    turn.querySelector(".bubble").textContent = content;
    return turn;
  }

  /* The provenance strip is what makes this transcript an *agent* transcript:
     the answer alone hides that six workers and a critic produced it. */
  function provenanceStrip({ context, actionLogs, feedback }) {
    const agents = new Set();
    for (const line of actionLogs || []) {
      const m = String(line).match(/worker:\s*([a-z_]+)|\b(scraper|search|extractor|doc_retriever|youtube_downloader|mermaid_generator)\b/i);
      if (m) agents.add((m[1] || m[2]).toLowerCase());
    }
    const docs = new Map();
    for (const c of context || []) {
      const s = typeof c === "string" ? c : "";
      if (!s.startsWith("[UPLOADED DOCUMENT")) continue;
      const f = s.match(/^file:\s*(.+)$/m);
      if (f) docs.set(f[1].trim(), (docs.get(f[1].trim()) || 0) + 1);
    }

    if (!agents.size && !docs.size && !(context || []).length) return null;

    const wrap = document.createElement("div");
    wrap.className = "provenance";

    const chips = [];
    for (const [name, n] of docs) {
      chips.push(`<span class="chip doc">📄 ${esc(name)}${n > 1 ? ` ×${n}` : ""}</span>`);
    }
    for (const a of agents) chips.push(`<span class="chip">${esc(a)}</span>`);
    if (!chips.length) chips.push(`<span class="chip">${(context || []).length} payload(s)</span>`);

    wrap.innerHTML = `
      <div class="provenance-row">
        <span class="prov-label">Evidence</span>
        ${chips.join("")}
        <button class="prov-more" type="button">Inspect</button>
      </div>
      <div class="prov-detail" hidden></div>`;

    const detail = wrap.querySelector(".prov-detail");
    const btn = wrap.querySelector(".prov-more");
    btn.addEventListener("click", () => {
      const show = detail.hidden;
      if (show && !detail.dataset.filled) {
        const parts = [];
        if (feedback) parts.push(`── critic ──\n${feedback}`);
        for (const c of context || []) {
          parts.push(typeof c === "string" ? c : JSON.stringify(c, null, 2));
        }
        detail.textContent = parts.join("\n\n") || "Nothing recorded.";
        detail.dataset.filled = "1";
      }
      detail.hidden = !show;
      btn.textContent = show ? "Hide" : "Inspect";
    });
    return wrap;
  }

  function graphTraceMarkup(hasData) {
    // Shared between the live (pending) turn and the finished turn, so the
    // "show reasoning graph" button — and the data behind it — survives the
    // pendingNode.replaceWith(turn) swap in finishTurn(), instead of
    // vanishing the instant the answer completes. Losing it there would
    // leave only the few seconds of streaming to ever click it, which
    // defeats the point of an on-demand, review-afterward feature.
    return `<button class="graph-trace-toggle" type="button" ${hasData ? "" : "hidden"}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="4" cy="4" r="2"/><circle cx="12" cy="4" r="2"/>
          <circle cx="8" cy="12" r="2"/>
          <path d="M5.5 5.2 7 10.4M10.5 5.2 9 10.4M6 4h4"/>
        </svg>
        Show reasoning graph
      </button>
      <div class="graph-trace-panel" hidden></div>`;
  }

  function assistantTurn({ content, context, actionLogs, feedback, graphTrace }) {
    const turn = document.createElement("article");
    turn.className = "turn assistant";
    const hasGraph = !!(graphTrace && graphTrace.collections && graphTrace.collections.length);
    if (hasGraph) {
      turn.insertAdjacentHTML("beforeend", graphTraceMarkup(true));
      turn._graphTrace = graphTrace;
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble markdown";
    bubble.innerHTML = md(content);
    turn.appendChild(bubble);
    const prov = provenanceStrip({ context, actionLogs, feedback });
    if (prov) turn.appendChild(prov);
    return turn;
  }

  function pendingTurn() {
    const turn = document.createElement("article");
    turn.className = "turn assistant";
    // .source-cards sits ABOVE the bubble and starts empty/hidden — it only
    // appears once the first "source_card" SSE event lands, which itself
    // only fires when the supervisor fanned out multiple scraper tasks (see
    // main.py's scrape_fanout_active gate). A single-source turn never shows
    // this at all, matching the old behaviour exactly.
    //
    // .graph-trace-toggle follows the identical pattern for the on-demand
    // reasoning-graph button: hidden until the first "graph_trace" SSE
    // event arrives (main.py only ever sends one when hydration actually
    // found graph-linked entities), so a turn with no graph involvement
    // shows nothing extra at all — no disabled button, no empty state.
    turn.innerHTML = `<div class="source-cards" hidden></div>
      ${graphTraceMarkup(false)}
      <div class="bubble">
      <div class="thinking">
        <span class="dots"><i></i><i></i><i></i></span>
        <span class="thinking-step">Supervisor is planning…</span>
      </div></div>`;
    turn._graphTrace = { collections: [] };   // accumulates across multiple events
    return turn;
  }

  function sourceCard({ source, title, word_count, error, insufficient, still_rendering, snippet }) {
    // One card per completed scraper result, appended live as each source
    // finishes — not waiting for the whole fanout batch. Purely metadata +
    // a pre-truncated text slice already computed server-side (_snippet in
    // main.py); no rendering work here beyond escaping, so this never
    // becomes the thing that slows down the stream it's meant to speed up.
    const card = document.createElement("div");
    card.className = "source-card";
    let domain = source || "";
    try { domain = new URL(source).hostname.replace(/^www\./, ""); } catch { }

    if (error) {
      card.classList.add("source-card--error");
      card.innerHTML = `
        <div class="source-card-head">
          <span class="source-card-title">${esc(domain)}</span>
          <span class="source-card-tag bad">failed</span>
        </div>
        <p class="source-card-snippet muted">${esc(error).slice(0, 140)}</p>`;
      return card;
    }

    const tag = still_rendering
      ? '<span class="source-card-tag warn">still rendering</span>'
      : insufficient
        ? '<span class="source-card-tag warn">thin</span>'
        : `<span class="source-card-tag">${word_count ?? "?"} words</span>`;

    card.innerHTML = `
      <div class="source-card-head">
        <span class="source-card-title">${esc(title || domain)}</span>
        ${tag}
      </div>
      ${snippet ? `<p class="source-card-snippet">${esc(snippet)}</p>` : ""}
      <span class="source-card-domain">${esc(domain)}</span>`;
    return card;
  }

  function scrollToEnd() {
    requestAnimationFrame(() => {
      el.transcript.scrollTop = el.transcript.scrollHeight;
    });
  }

  // ── attachments ──────────────────────────────────────────────────────

  el.attachBtn.addEventListener("click", () => {
    if (!state.conversationId) { toast("Start a chat first."); return; }
    el.fileInput.click();
  });

  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files.length) uploadFiles(el.fileInput.files);
    el.fileInput.value = "";
  });

  // Drag and drop onto the composer area
  ["dragenter", "dragover"].forEach((ev) =>
    el.composerWrap.addEventListener(ev, (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      el.composerWrap.dataset.drag = "true";
    }));

  ["dragleave", "drop"].forEach((ev) =>
    el.composerWrap.addEventListener(ev, (e) => {
      if (ev === "dragleave" && el.composerWrap.contains(e.relatedTarget)) return;
      el.composerWrap.dataset.drag = "false";
    }));

  el.composerWrap.addEventListener("drop", (e) => {
    if (!e.dataTransfer?.files?.length) return;
    e.preventDefault();
    uploadFiles(e.dataTransfer.files);
  });

  async function uploadFiles(fileList) {
    if (!state.conversationId) { toast("Start a chat first."); return; }

    const form = new FormData();
    form.append("conversation_id", state.conversationId);
    for (const f of fileList) form.append("files", f);

    // Optimistic chips so the user sees something the instant they drop.
    for (const f of fileList) {
      state.docs.push({
        id: `tmp:${f.name}:${f.size}`, file_name: f.name,
        size_bytes: f.size, status: "uploading", chunk_count: 0
      });
    }
    renderAttachmentChips();

    try {
      const data = await api("/api/uploads", { method: "POST", body: form });
      state.docs = state.docs.filter((d) => !String(d.id).startsWith("tmp:"));
      for (const a of data.accepted || []) {
        state.docs.push({
          id: a.doc_id, file_name: a.file_name,
          size_bytes: a.size_bytes, status: a.status,
          chunk_count: 0, child_count: a.child_count || 0,
          parent_doc_id: a.parent_doc_id || null,
        });
      }
      for (const r of data.rejected || []) {
        toast(`${r.file_name}: ${r.reason}`, "error", 5000);
      }
      renderAttachmentChips();
      renderCorpusBadge(data.corpus);
      startPolling();
    } catch (e) {
      state.docs = state.docs.filter((d) => !String(d.id).startsWith("tmp:"));
      renderAttachmentChips();
      toast(`Upload failed: ${e.message}`, "error", 5000);
    }
  }

  // ── clipboard pasting ────────────────────────────────────────────────

  // 1. Listen for standard Ctrl-V / Cmd-V on the textarea
  el.prompt.addEventListener("paste", (e) => {
    if (!e.clipboardData || !e.clipboardData.files.length) return;

    // If there are files in the clipboard (e.g. image screenshots), upload them
    uploadFiles(e.clipboardData.files);
  });

  // 2. Listen for clicks on the new Paste button
  el.pasteBtn.addEventListener("click", async () => {
    if (!state.conversationId) { toast("Start a chat first."); return; }

    if (!navigator.clipboard || !navigator.clipboard.read) {
      toast("Clipboard button unavailable. Please use Ctrl-V to paste.", "error", 4000);
      return;
    }

    try {
      // Prompt the user for clipboard read permissions
      const clipboardItems = await navigator.clipboard.read();
      const filesToUpload = [];

      for (const item of clipboardItems) {
        // Find supported types (usually images or text payloads in clipboard)
        for (const type of item.types) {
          if (type.startsWith("image/") || type.startsWith("text/") || type.includes("pdf")) {
            const blob = await item.getType(type);

            // Generate a filename since clipboard blobs don't have native filenames
            const extension = type.split("/")[1]?.split(";")[0] || "bin";
            const fileName = `clipboard-${Date.now()}.${extension}`;

            const file = new File([blob], fileName, { type });
            filesToUpload.push(file);
          }
        }
      }

      if (filesToUpload.length > 0) {
        uploadFiles(filesToUpload);
      } else {
        toast("No readable files found in clipboard.", "info");
      }
    } catch (err) {
      // This will fire if the user denies clipboard permission or if the API is unsupported
      toast(`Clipboard access failed: ${err.message}`, "error");
    }
  });
  // Clipboard pasting end here ------------------------------------
  const STATE_LABEL = {
    uploading: "uploading", expanding: "unpacking", pending: "queued",
    extracting: "reading",
    chunking: "chunking", embedding: "embedding", indexed: "ready",
    degraded: "partial", failed: "failed", quarantined: "unsupported",
  };

  function renderAttachmentChips() {
    if (!state.docs.length) {
      el.attachments.hidden = true;
      el.attachments.innerHTML = "";
      return;
    }
    el.attachments.hidden = false;
    el.attachments.innerHTML = state.docs.map((d) => {
      const busy = !SETTLED.has(d.status);
      const label = STATE_LABEL[d.status] || d.status;
      // An archive row is a manifest, not content — it has no chunks of its
      // own, so show how many documents came out of it instead of "0 chunks".
      const detail = d.child_count
        ? `${d.child_count} file${d.child_count === 1 ? "" : "s"}`
        : d.status === "indexed" ? `${d.chunk_count} chunks`
          : (d.reason_code ? String(d.reason_code).split(":")[0] : label);
      return `<span class="att" data-state="${esc(d.status)}" data-busy="${busy}"
                title="${esc(d.file_name)} · ${bytes(d.size_bytes)} · ${esc(detail)}">
        <span class="att-name">${esc(d.file_name)}</span>
        <span class="att-state">${esc(detail)}</span>
        <button class="att-x" type="button" data-doc="${esc(d.id)}"
                aria-label="Remove ${esc(d.file_name)}">✕</button>
      </span>`;
    }).join("");
  }

  el.attachments.addEventListener("click", async (e) => {
    const btn = e.target.closest(".att-x");
    if (!btn) return;
    const id = btn.dataset.doc;
    if (String(id).startsWith("tmp:")) return;
    try {
      await api(`/api/uploads/doc/${id}`, { method: "DELETE" });
      state.docs = state.docs.filter((d) => d.id !== id);
      renderAttachmentChips();
      renderTraceDocs();
      await refreshDocs();
    } catch (err) {
      toast(`Couldn't remove that file: ${err.message}`, "error");
    }
  });

  /* Poll rather than stream ingestion status: a second SSE channel alongside
     the answer stream would double the connection budget for a payload that is
     four fields wide and changes a handful of times per file. */
  function startPolling() {
    stopPolling();
    state.pollTimer = setInterval(refreshDocs, 1500);
  }
  function stopPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = null;
  }

  async function refreshDocs() {
    if (!state.conversationId) return stopPolling();
    try {
      const data = await api(`/api/uploads/${state.conversationId}`);
      state.docs = data.documents || [];
      renderAttachmentChips();
      renderCorpusBadge(data.corpus);
      renderTraceDocs();
      if (data.all_settled) {
        stopPolling();
        await loadThreads(el.threadSearch.value.trim());
      }
    } catch {
      stopPolling();
    }
  }

  // ── composer ─────────────────────────────────────────────────────────

  function autogrow() {
    el.prompt.style.height = "auto";
    el.prompt.style.height = `${Math.min(el.prompt.scrollHeight, 208)}px`;
  }
  el.prompt.addEventListener("input", autogrow);

  el.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      el.composer.requestSubmit();
    }
  });

  el.starters.addEventListener("click", (e) => {
    const b = e.target.closest(".starter");
    if (!b) return;
    el.prompt.value = b.dataset.prompt;
    autogrow();
    el.prompt.focus();
  });

  el.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    submitTurn();
  });

  /* "Finish now": ask the server to end the graph at its next routing
     decision and answer from the context it already has.

     Deliberately does NOT abort the fetch. Aborting would stop the browser
     reading while the server kept working — the user would lose the answer
     AND still pay for it. Instead the stream stays open and closes with a
     normal "complete" event, so the turn is rendered, persisted and
     reloadable exactly like any other. The only difference is the banner and
     that the critic never ran.

     The button disables itself immediately rather than waiting for the POST:
     a second press cannot do anything the first didn't, and a still-live
     button through the (possibly several seconds of) remaining synthesis
     reads as a control that isn't working. */
  el.stopBtn.addEventListener("click", async () => {
    if (!state.runId) return;
    el.stopBtn.disabled = true;
    traceLog("⏹️ Finish-now requested — waiting for the current step to land...");
    try {
      const res = await fetch("/chat/stop", {
        method: "POST",
        body: new URLSearchParams({ run_id: state.runId }),
      });
      const data = await res.json();
      if (!data.ok) traceLog(`⏹️ ${data.detail}`);
    } catch (err) {
      // Re-enable: the request never reached the server, so a retry is
      // meaningful here, unlike the success path.
      el.stopBtn.disabled = false;
      traceLog(`❌ Could not request stop: ${err.message}`);
    }
  });

  async function submitTurn() {
    const prompt = el.prompt.value.trim();
    if (!prompt || state.busy) return;
    if (!state.conversationId) { await newChat(); }

    const unsettled = state.docs.filter((d) => !SETTLED.has(d.status));
    if (unsettled.length &&
      !confirm(`${unsettled.length} file(s) are still processing and won't be searchable yet. Send anyway?`)) {
      return;
    }

    state.busy = true;
    // Cleared here, set when the server's "run" event lands. Anything the
    // stop button does before that is a no-op, which is correct — there is
    // no run to stop yet.
    state.runId = null;
    el.stopBtn.disabled = false;
    el.stopBtn.hidden = false;
    el.sendBtn.disabled = true;
    el.welcome.hidden = true;
    el.convo.dataset.hasMessages = "true";
    if (el.welcome.parentNode === el.transcript) el.transcript.removeChild(el.welcome);

    const attached = state.docs
      .filter((d) => d.status === "indexed" || d.status === "degraded")
      .map((d) => ({ doc_id: d.id, file_name: d.file_name }));

    el.transcript.appendChild(userTurn(prompt, attached));
    const pending = pendingTurn();
    el.transcript.appendChild(pending);
    scrollToEnd();

    el.prompt.value = "";
    autogrow();
    traceReset();
    el.tracePulse.dataset.state = "running";
    renderTraceContext(null);

    const step = pending.querySelector(".thinking-step");
    const form = new FormData();
    form.append("prompt", prompt);
    form.append("conversation_id", state.conversationId);
    form.append("user_id", state.userId);

    try {
      const res = await fetch("/chat/stream", { method: "POST", body: form });
      if (!res.ok || !res.body) throw new Error(`Stream failed (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;

      // ----------------------------------------------------------------
      // LATENCY: progressive answer rendering.
      //
      // The server now sends the answer in three escalating forms:
      //   answer_delta -> decoded tokens as the supervisor writes them
      //   answer       -> the confirmed full text, before the critic runs
      //   complete     -> authoritative payload incl. late background scrapes
      //
      // Each supersedes the last, so we always overwrite rather than append
      // across types. `liveText` accumulates only the token deltas; the
      // moment a fuller form arrives it wins, which keeps this correct even
      // if token streaming is unavailable and `answer` is the first thing
      // we hear.
      // ----------------------------------------------------------------
      let liveText = "";          // accumulated answer_delta text
      let streaming = false;      // has the bubble switched out of "thinking"?
      let criticData = null;      // critic verdict, if it lands before complete
      let renderQueued = false;   // rAF throttle guard

      const bubble = () => pending.querySelector(".bubble");

      function paint() {
        renderQueued = false;
        bubble().innerHTML = md(liveText);
        scrollToEnd();
      }

      function renderLive() {
        // First token replaces the thinking indicator with a real bubble.
        if (!streaming) {
          streaming = true;
          const b = bubble();
          b.classList.add("markdown");
          b.innerHTML = "";
        }
        // Markdown parsing re-parses the WHOLE answer each call, so painting
        // on every token is quadratic in answer length — on a long report the
        // tail arrives visibly slower than the head, which is precisely the
        // impression this feature exists to remove. Coalescing to one paint
        // per animation frame keeps it smooth regardless of token rate, and
        // matches the screen's refresh anyway.
        if (!renderQueued) {
          renderQueued = true;
          requestAnimationFrame(paint);
        }
      }

      while (!done) {
        const { value, done: finished } = await reader.read();
        if (finished) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); }
          catch { continue; }

          if (data.type === "run") {
            state.runId = data.run_id;
          } else if (data.type === "log") {
            traceLog(data.message);
            // Only drive the thinking-step caption while still thinking —
            // once prose is on screen, replacing it with a status line
            // would yank the text the user is mid-sentence on.
            if (step && !streaming) step.textContent = String(data.message).slice(0, 90);
          } else if (data.type === "answer_reset") {
            // New supervisor loop starting. Deliberately does NOT clear
            // liveText — this also fires when critic REJECTS an answer and
            // the graph loops back for regeneration, and the just-streamed
            // content at that moment is often a good, complete answer, not
            // garbage. Clearing it there erased good content the instant
            // critic finished evaluating — worse than the cosmetic problem
            // (a concatenated multi-attempt blob during live preview) this
            // was meant to fix. The live preview is cosmetic either way;
            // "complete" always supersedes it with the real data.final_response.
            streaming = false;
          } else if (data.type === "answer_delta") {
            liveText += data.text;
            renderLive();
          } else if (data.type === "answer") {
            // Confirmed full text for THIS supervisor loop — but not
            // necessarily the best one. This event fires on every loop that
            // produces a final_response with no pending tasks, which
            // includes regeneration attempts after a critic rejection and
            // the truncated attempt when max_loops is hit. An unconditional
            // `liveText = data.text` here is what was still wiping good
            // content: a later, shorter, worse attempt replacing a complete
            // earlier one the user was already reading.
            //
            // Length is a crude proxy for completeness, but it's the only
            // signal available client-side, and the failure it prevents
            // (a full answer replaced by a truncated stub) is far worse
            // than the one it risks (keeping a longer answer when a genuinely
            // better shorter one arrived). "complete" still has the final
            // say either way — it carries the backend's own resolved
            // final_response, which is the actual source of truth.
            if (!liveText || data.text.length >= liveText.length) {
              liveText = data.text;
              renderLive();
            } else {
              traceLog(`↩︎ Kept the longer previous answer (${liveText.length} chars) `
                + `over a shorter regeneration (${data.text.length} chars).`);
            }
          } else if (data.type === "source_card") {
            // Live per-source card — see main.py's scrape_fanout_active gate
            // for when this fires. Reveal the (initially hidden) container on
            // the first card and append; never replaces or reorders earlier
            // cards, so a fast source doesn't jump around when a slower one
            // lands after it.
            const wrap = pending.querySelector(".source-cards");
            wrap.hidden = false;
            wrap.appendChild(sourceCard(data));
            scrollToEnd();
          } else if (data.type === "critic") {
            // Arrives while the user is already reading. Attach the verdict
            // to the trace panel immediately; the provenance strip picks it
            // up when the turn is finalised.
            criticData = data;
            el.traceFeedback.innerHTML = data.feedback
              ? md(data.feedback)
              : '<p class="muted">No evaluation recorded.</p>';
            traceLog(`⚖️ Critic score: ${data.score}/100`);
          } else if (data.type === "graph_trace") {
            // Ephemeral by design — lives only on this DOM node for this
            // turn, never sent to the server for storage (see main.py's
            // add_message call, which deliberately excludes it). Reload the
            // page or reopen this thread and the button is simply gone,
            // which is the agreed behaviour, not a bug to fix later.
            const cols = data.collections || [];
            pending._graphTrace.collections.push(...cols);
            const btn = pending.querySelector(".graph-trace-toggle");
            if (cols.length && btn.hidden) btn.hidden = false;
          } else if (data.type === "complete") {
            done = true;
            finishTurn(pending, data, liveText);
          }
        }
      }
      if (!done) {
        // Stream cut off. If prose already landed, keep it — a partial
        // answer the user can read beats replacing it with an error.
        if (!streaming) {
          bubble().innerHTML =
            '<p class="muted">The stream ended without a final answer. Check the trace.</p>';
        }
        el.tracePulse.dataset.state = "error";
      }
    } catch (err) {
      el.tracePulse.dataset.state = "error";
      traceLog(`❌ ${err.message}`);
      pending.querySelector(".bubble").innerHTML =
        `<p><strong>Connection error.</strong> ${esc(err.message)}</p>`;
    } finally {
      state.busy = false;
      state.runId = null;
      el.stopBtn.hidden = true;
      el.stopBtn.disabled = false;
      el.sendBtn.disabled = false;
      el.prompt.focus();
    }
  }

  // ── reasoning graph (on-demand, ephemeral) ───────────────────────────

  /* Hand-rolled SVG, deliberately no charting library. This runs in the
     deployed app's own frontend for real end users — not a capability
     available only in an authoring/assistant context — and this deployment
     looks offline/restricted-network (local Neo4j, local file paths), so a
     CDN-loaded dependency is a real risk, not a hypothetical one. A single
     ring layout is enough at the node counts this ever ships (capped
     server-side at config.graph_trace_max_nodes, default 15) and needs no
     physics simulation to look clean. */

  el.transcript.addEventListener("click", (e) => {
    const btn = e.target.closest(".graph-trace-toggle");
    if (!btn) return;
    const turn = btn.closest(".turn");
    const panel = turn.querySelector(".graph-trace-panel");
    if (!panel.hidden) {
      panel.hidden = true;
      btn.classList.remove("is-open");
      return;
    }
    if (!panel.dataset.rendered) {
      renderGraphTrace(panel, turn._graphTrace);
      panel.dataset.rendered = "1";
    }
    panel.hidden = false;
    btn.classList.add("is-open");
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  const GRAPH_PALETTE = {
    PERSON: "#2563eb", ORG: "#059669", GPE: "#d97706", LOC: "#d97706",
    NORP: "#db2777", DATE: "#7c3aed", DEFAULT: "#64748b",
  };

  function renderGraphTrace(container, trace) {
    const collections = (trace && trace.collections) || [];
    const nodeMap = new Map();     // lowercase name -> {name, label}
    const edges = [];              // {a, b, weight} — a/b are lowercase keys
    let anyStale = false;

    for (const c of collections) {
      if (c.stale) anyStale = true;
      for (const f of c.facts || []) {
        const ek = f.entity.toLowerCase(), rk = f.related.toLowerCase();
        if (!nodeMap.has(ek)) nodeMap.set(ek, { name: f.entity, label: f.entity_label });
        if (!nodeMap.has(rk)) nodeMap.set(rk, { name: f.related, label: f.related_label });
        edges.push({ a: ek, b: rk, weight: f.weight || 1 });
      }
    }

    const nodes = [...nodeMap.entries()].map(([key, v]) => ({ key, ...v }));
    if (!nodes.length) {
      container.innerHTML = '<p class="muted">No graph data for this answer.</p>';
      return;
    }

    const W = 420, H = 380, R = 140, CX = W / 2, CY = H / 2 + 10;
    const pos = new Map();
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
      pos.set(node.key, { x: CX + R * Math.cos(angle), y: CY + R * Math.sin(angle), angle });
    });

    const maxWeight = Math.max(...edges.map((e) => e.weight), 1);
    const edgeSvg = edges.map((e) => {
      const p1 = pos.get(e.a), p2 = pos.get(e.b);
      if (!p1 || !p2) return "";
      const opacity = (0.25 + 0.55 * (e.weight / maxWeight)).toFixed(2);
      return `<line x1="${p1.x.toFixed(1)}" y1="${p1.y.toFixed(1)}" ` +
        `x2="${p2.x.toFixed(1)}" y2="${p2.y.toFixed(1)}" ` +
        `stroke="#94a3b8" stroke-width="1.5" opacity="${opacity}"/>`;
    }).join("");

    const nodeSvg = nodes.map((node) => {
      const p = pos.get(node.key);
      const color = GRAPH_PALETTE[node.label] || GRAPH_PALETTE.DEFAULT;
      const label = node.name.length > 18 ? `${node.name.slice(0, 17)}…` : node.name;
      const lx = p.x + Math.cos(p.angle) * 16;
      const ly = p.y + Math.sin(p.angle) * 16;
      const anchor = Math.cos(p.angle) > 0.3 ? "start"
        : Math.cos(p.angle) < -0.3 ? "end" : "middle";
      return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="7" ` +
        `fill="${color}" stroke="#fff" stroke-width="2">` +
        `<title>${esc(node.name)} (${esc(node.label)})</title></circle>` +
        `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10" fill="#334155" ` +
        `text-anchor="${anchor}" dominant-baseline="middle">${esc(label)}</text>`;
    }).join("");

    const names = collections.map((c) => c.collection_name);
    const caption = names.length === 1
      ? `From knowledge graph: ${esc(names[0])}`
      : `From ${names.length} knowledge graph collection(s)`;

    container.innerHTML = `
      <div class="graph-trace-caption">
        ${caption}
        ${anyStale ? '<span class="graph-trace-stale">may be out of date</span>' : ""}
      </div>
      <svg viewBox="0 0 ${W} ${H}" class="graph-trace-svg" role="img"
           aria-label="Reasoning graph showing entities related to this answer">
        ${edgeSvg}${nodeSvg}
      </svg>
      <p class="graph-trace-hint">Shows what the knowledge graph surfaced as related
        for this question — not necessarily every fact the answer relied on.</p>`;
  }

  function finishTurn(pendingNode, data, liveTextAtFinish) {
    el.tracePulse.dataset.state = "idle";
    traceLog("🎉 Run complete.");

    // Same guard as the "answer" handler, applied at the point where the
    // pending turn is actually replaced — this is where the post-critic
    // wipe was visible: a complete answer on screen, replaced by a shorter
    // final_response the moment "complete" arrived.
    //
    // Backend now resolves final_response through a three-tier fallback
    // (final_state -> direct supervisor capture -> streamed-token recovery,
    // see main.py), so it is usually right. But when it lands shorter than
    // content the user is already reading, that is the truncation case, not
    // an improvement — keep what is on screen.
    let content = data.final_response;
    if (liveTextAtFinish && content && content.length < liveTextAtFinish.length) {
      traceLog(`↩︎ Final response (${content.length} chars) was shorter than the `
        + `streamed answer (${liveTextAtFinish.length} chars) — kept the longer one.`);
      content = liveTextAtFinish;
    } else if (!content && liveTextAtFinish) {
      content = liveTextAtFinish;
    }

    const turn = assistantTurn({
      content: content || "_No answer was produced. The trace has the detail._",
      context: data.context || [],
      actionLogs: data.action_logs || state.logs,
      feedback: data.feedback,
      graphTrace: pendingNode._graphTrace,
    });
    pendingNode.replaceWith(turn);
    scrollToEnd();

    el.traceFeedback.innerHTML = data.feedback
      ? md(data.feedback)
      : '<p class="muted">No evaluation recorded.</p>';
    renderTraceContext(data.context);

    if (data.title) {
      el.convoTitle.textContent = data.title;
    }
    loadThreads(el.threadSearch.value.trim());
  }

  // ── knowledge collections ────────────────────────────────────────────

  /* Collections are corpora built by batch jobs. They live outside any one
     chat, so attaching one is an explicit act: retrieval scope is this
     conversation PLUS whatever it has been given, never anything implicit. */

  async function renderCollections() {
    if (!state.conversationId) {
      el.collectionsBody.innerHTML =
        '<p class="muted">Start a chat first.</p>';
      return;
    }
    el.collectionsBody.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const data = await api(`/api/conversations/${state.conversationId}/collections`);
      const attachedIds = new Set(data.attached.map((c) => c.id));
      state.collections = data.attached;
      state.availableCollections = data.available;
      updateCollectionsCount();
      updateScopeBanner();

      if (!data.available.length) {
        el.collectionsBody.innerHTML =
          '<p class="muted">No collections yet. Build one from the ' +
          '<a href="/jobs">Jobs</a> page by ingesting a folder.</p>';
        return;
      }

      el.collectionsBody.innerHTML = data.available.map((c) => {
        const on = attachedIds.has(c.id);
        const g = c.graph || { status: "none" };
        // Only render anything when there IS a graph — a collection with
        // none looks exactly like it did before this feature existed. This
        // is the "made aware where relevant, invisible where not" placement
        // the graph feature needs: right where someone is deciding whether
        // to search a collection, not a separate page they'd have to check.
        const graphTag = (g.status === "ready" || g.status === "stale")
          ? `<span class="coll-graph-tag" data-status="${esc(g.stale ? "stale" : g.status)}"
               title="${g.stale ? "Graph may be out of date for this collection" : "Knowledge graph available for this collection"}">
               ${g.stale ? "graph · stale" : "graph"}
             </span>` : "";
        return `<div class="coll-row" data-id="${esc(c.id)}">
          <label class="coll-toggle">
            <input type="checkbox" ${on ? "checked" : ""} data-id="${esc(c.id)}">
            <span class="coll-name">${esc(c.name)}</span>
            ${graphTag}
          </label>
          <span class="coll-meta">${c.documents.toLocaleString()} docs ·
            ${c.chunks.toLocaleString()} chunks</span>
          ${c.description ? `<p class="coll-desc">${esc(c.description)}</p>` : ""}
        </div>`;
      }).join("");
    } catch (e) {
      el.collectionsBody.innerHTML =
        `<p class="muted">Couldn't load collections. ${esc(e.message)}</p>`;
    }
  }

  // The gap this closes: a chat can have zero collections attached with no
  // signal anywhere that anything was skipped, so a query silently falls back
  // to the web. This banner sits right above the composer — where the person
  // is already looking — rather than in the rail, which is easy to miss
  // especially on mobile where the rail starts collapsed.
  function updateScopeBanner() {
    const nonEmpty = (state.availableCollections || []).filter((c) => c.documents > 0);
    const attachedCount = (state.collections || []).length;
    const dismissed = state.conversationId
      && state.dismissedScopeBanner.has(state.conversationId);
    if (attachedCount > 0 || nonEmpty.length === 0 || dismissed) {
      el.scopeBanner.hidden = true;
      return;
    }
    el.scopeBanner.hidden = false;
    el.scopeBannerText.textContent =
      `No knowledge collections attached to this chat — ${nonEmpty.length} ` +
      `available. Answers won't include them until you attach one.`;
  }

  el.scopeBannerClose.addEventListener("click", () => {
    if (state.conversationId) state.dismissedScopeBanner.add(state.conversationId);
    el.scopeBanner.hidden = true;
  });

  el.scopeBannerBtn.addEventListener("click", () => {
    el.collectionsSheet.hidden = false;
    el.scrim.hidden = false;
    renderCollections();
    el.collSearchInput.focus();
  });

  // ── cross-collection discovery search ──────────────────────────────

  /* Solves the other half of the problem: even once someone opens the sheet,
     they may not know WHICH of several collections has their data. This runs
     the query against every non-empty collection at once and shows counts +
     a snippet, so the choice of what to attach is informed rather than a
     guess. Read-only — nothing is attached until the person clicks Attach. */

  async function runCollectionDiscovery() {
    const q = el.collSearchInput.value.trim();
    if (!q) { el.collSearchResults.innerHTML = ""; return; }

    el.collSearchResults.innerHTML = '<p class="muted">Searching…</p>';
    try {
      const data = await api(`/api/collections/search?q=${encodeURIComponent(q)}`);
      if (!data.results.length) {
        el.collSearchResults.innerHTML =
          '<p class="muted">No matches in any collection.</p>';
        return;
      }
      const attachedIds = new Set((state.collections || []).map((c) => c.id));
      el.collSearchResults.innerHTML = data.results.map((r) => {
        const c = r.collection;
        const on = attachedIds.has(c.id);
        const snippet = r.snippets[0]
          ? `<p class="coll-result-snippet">${esc(r.snippets[0].file_name)}: `
            + `${esc(r.snippets[0].text)}…</p>` : "";
        return `<div class="coll-result-row">
          <div class="coll-result-head">
            <span class="coll-name">${esc(c.name)}</span>
            <span class="coll-meta">${r.hit_count} match${r.hit_count === 1 ? "" : "es"}</span>
            <button class="coll-attach-btn" data-id="${esc(c.id)}" ${on ? "disabled" : ""}>
              ${on ? "Attached" : "Attach"}
            </button>
          </div>
          ${snippet}
        </div>`;
      }).join("");
    } catch (e) {
      el.collSearchResults.innerHTML =
        `<p class="muted">Search failed. ${esc(e.message)}</p>`;
    }
  }

  let collSearchTimer;
  el.collSearchInput.addEventListener("input", () => {
    clearTimeout(collSearchTimer);
    collSearchTimer = setTimeout(runCollectionDiscovery, 400);
  });
  el.collSearchBtn.addEventListener("click", runCollectionDiscovery);

  el.collSearchResults.addEventListener("click", async (e) => {
    const btn = e.target.closest(".coll-attach-btn");
    if (!btn || btn.disabled || !state.conversationId) return;
    const id = btn.dataset.id;
    btn.disabled = true;
    btn.textContent = "…";
    try {
      const data = await api(
        `/api/conversations/${state.conversationId}/collections/${id}`,
        { method: "POST" });
      state.collections = data.attached || [];
      updateCollectionsCount();
      updateScopeBanner();
      btn.textContent = "Attached";
      toast("Attached. This chat can now search that collection.");
      renderCollections();   // syncs the checklist below the search box
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Attach";
      toast(`Couldn't attach: ${err.message}`, "error");
    }
  });

  function updateCollectionsCount() {
    const n = (state.collections || []).length;
    el.collectionsCount.hidden = n === 0;
    el.collectionsCount.textContent = String(n);
  }

  el.collectionsBody.addEventListener("change", async (e) => {
    const box = e.target.closest('input[type="checkbox"]');
    if (!box || !state.conversationId) return;
    const id = box.dataset.id;
    const url = `/api/conversations/${state.conversationId}/collections/${id}`;
    box.disabled = true;
    try {
      const data = await api(url, { method: box.checked ? "POST" : "DELETE" });
      state.collections = data.attached || [];
      updateCollectionsCount();
      updateScopeBanner();
      toast(box.checked
        ? "Attached. This chat can now search that collection."
        : "Detached.");
    } catch (err) {
      box.checked = !box.checked;
      toast(`Couldn't update: ${err.message}`, "error");
    } finally {
      box.disabled = false;
    }
  });

  el.collectionsBtn.addEventListener("click", () => {
    el.collectionsSheet.hidden = false;
    el.scrim.hidden = false;
    renderCollections();
  });

  el.collectionsClose.addEventListener("click", closeCollections);

  function closeCollections() {
    el.collectionsSheet.hidden = true;
    el.scrim.hidden = true;
  }

  // ── downloaded assets sheet ──────────────────────────────────────────

  function closeSheet() {
    el.downloadsSheet.hidden = true;
    el.scrim.hidden = true;
  }

  el.downloadsBtn.addEventListener("click", async () => {
    el.downloadsSheet.hidden = false;
    el.scrim.hidden = false;
    el.downloadsBody.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const data = await api("/api/downloads");
      if (!data.files.length) {
        el.downloadsBody.innerHTML = '<p class="muted">No assets downloaded yet.</p>';
        return;
      }
      el.downloadsBody.innerHTML = data.files.map((f) => `
        <div class="asset-row" data-name="${esc(f.name.toLowerCase())}">
          <a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.name)}</a>
          <button class="icon-btn danger" data-del="${encodeURIComponent(f.name)}"
                  aria-label="Delete ${esc(f.name)}">✕</button>
        </div>`).join("");
    } catch (e) {
      el.downloadsBody.innerHTML = `<p class="muted">Couldn't load assets. ${esc(e.message)}</p>`;
    }
  });

  el.downloadsBody.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-del]");
    if (!btn) return;
    if (!confirm("Delete this file from the server?")) return;
    try {
      await api(`/api/downloads/${btn.dataset.del}`, { method: "DELETE" });
      btn.closest(".asset-row").remove();
    } catch (err) {
      toast(`Delete failed: ${err.message}`, "error");
    }
  });

  el.assetSearch.addEventListener("input", () => {
    const q = el.assetSearch.value.toLowerCase();
    el.downloadsBody.querySelectorAll(".asset-row").forEach((row) => {
      row.style.display = row.dataset.name.includes(q) ? "" : "none";
    });
  });

  el.downloadsClose.addEventListener("click", closeSheet);
  el.scrim.addEventListener("click", () => { closeSheet(); closeCollections(); });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!el.downloadsSheet.hidden) closeSheet();
    if (!el.collectionsSheet.hidden) closeCollections();
  });

  // ── boot ─────────────────────────────────────────────────────────────

  async function boot() {
    setRail(!isNarrow());
    setTrace(false);
    // `marked` comes from a CDN. On an offline box, behind a strict proxy, or
    // on an air-gapped deploy that script 403s and `marked` is undefined —
    // unguarded, this line threw and aborted boot() before loadThreads(),
    // leaving the whole app dead rather than merely unstyled. md() already
    // falls back to escaped plain text, so degrading here is safe.
    try {
      marked.setOptions({ breaks: true, gfm: true });
    } catch {
      console.warn("marked unavailable — falling back to plain-text rendering.");
    }

    // One allowlist, owned by the server (docstore/filetypes.py). Hard-coding
    // it here is how chat upload and batch ingest drift apart.
    try {
      const types = await api("/api/filetypes");
      el.fileInput.setAttribute("accept", types.accept);
    } catch { /* the server enforces the real limit regardless */ }

    await loadThreads();
    const hashed = location.hash.replace("#", "");
    if (hashed && state.threads.some((t) => t.id === hashed)) {
      await selectThread(hashed);
    } else if (state.threads.length) {
      await selectThread(state.threads[0].id);
    } else {
      await newChat();
    }
    el.prompt.focus();
  }

  window.addEventListener("beforeunload", stopPolling);
  boot();
})();
