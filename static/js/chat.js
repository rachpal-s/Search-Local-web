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
    fileInput: $("file-input"), attachments: $("attachments"),
    trace: $("trace"), traceToggle: $("trace-toggle"), traceClose: $("trace-close"),
    traceStream: $("trace-stream"), tracePulse: $("trace-pulse"),
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
  const state = {
    conversationId: null,
    threads: [],
    docs: [],              // documents attached to the active thread
    collections: [],       // knowledge collections this thread may also search
    availableCollections: [],  // every collection that exists, for the scope banner
    dismissedScopeBanner: new Set(),  // thread ids where the banner was closed this session
    busy: false,           // a turn is in flight
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

  async function selectThread(id) {
    if (state.busy) { toast("Wait for the current answer to finish."); return; }
    stopPolling();
    try {
      const data = await api(`/api/conversations/${id}`);
      state.conversationId = id;
      state.docs = data.documents || [];
      location.hash = id;

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

  function assistantTurn({ content, context, actionLogs, feedback }) {
    const turn = document.createElement("article");
    turn.className = "turn assistant";
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
    turn.innerHTML = `<div class="source-cards" hidden></div>
      <div class="bubble">
      <div class="thinking">
        <span class="dots"><i></i><i></i><i></i></span>
        <span class="thinking-step">Supervisor is planning…</span>
      </div></div>`;
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

          if (data.type === "log") {
            traceLog(data.message);
            // Only drive the thinking-step caption while still thinking —
            // once prose is on screen, replacing it with a status line
            // would yank the text the user is mid-sentence on.
            if (step && !streaming) step.textContent = String(data.message).slice(0, 90);
          } else if (data.type === "answer_delta") {
            liveText += data.text;
            renderLive();
          } else if (data.type === "answer") {
            // Confirmed full text. Supersedes anything streamed so far —
            // and covers the no-streaming case, where this is the first
            // answer content to arrive.
            liveText = data.text;
            renderLive();
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
          } else if (data.type === "complete") {
            done = true;
            finishTurn(pending, data);
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
      el.sendBtn.disabled = false;
      el.prompt.focus();
    }
  }

  function finishTurn(pendingNode, data) {
    el.tracePulse.dataset.state = "idle";
    traceLog("🎉 Run complete.");

    const turn = assistantTurn({
      content: data.final_response || "_No answer was produced. The trace has the detail._",
      context: data.context || [],
      actionLogs: data.action_logs || state.logs,
      feedback: data.feedback,
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
        return `<div class="coll-row" data-id="${esc(c.id)}">
          <label class="coll-toggle">
            <input type="checkbox" ${on ? "checked" : ""} data-id="${esc(c.id)}">
            <span class="coll-name">${esc(c.name)}</span>
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
