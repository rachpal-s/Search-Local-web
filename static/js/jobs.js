/* jobs.js — ingestion jobs dashboard.
 *
 * Two live channels, deliberately different:
 *   the LIST polls every 6s — it only needs to be roughly current, and a poll
 *   survives a mobile tab being backgrounded, which an SSE connection does not.
 *   the OPEN JOB uses SSE — it is being watched, so it gets per-second state
 *   plus the event feed, and closes itself on a terminal status rather than
 *   reconnecting forever.
 */
(() => {
  "use strict";

  const PHASES = [
    ["expand",   "Expand archives"],
    ["discover", "Discover files"],
    ["ingest",   "Ingest documents"],
    ["resolve",  "Resolve entities"],
    ["write",    "Write to graph"],
    ["verify",   "Verify"],
  ];

  const $ = (id) => document.getElementById(id);
  const isMobile = () => window.matchMedia("(max-width: 700px)").matches;

  const state = {
    jobs: [], selected: null, stream: null,
    page: 0, size: 25, total: 0, status: "", seen: new Set(),
  };

  // ── utils ──────────────────────────────────────────────────────────

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const num = (n) => (n || 0).toLocaleString();

  function rel(iso) {
    if (!iso) return "—";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `${Math.floor(s)}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  }

  function dur(a, b) {
    if (!a) return "—";
    const s = Math.max(0, ((b ? new Date(b) : new Date()) - new Date(a)) / 1000);
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }

  function toast(msg, isErr = false) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.toggle("err", isErr);
    t.hidden = false;
    clearTimeout(t._t);
    t._t = setTimeout(() => { t.hidden = true; }, 4200);
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" }, ...opts,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const d = body.detail;
      throw new Error(typeof d === "string" ? d : (d ? JSON.stringify(d) : `HTTP ${res.status}`));
    }
    return body;
  }

  // ── list ───────────────────────────────────────────────────────────

  async function loadJobs() {
    const q = new URLSearchParams({ limit: state.size, offset: state.page * state.size });
    if (state.status) q.set("status", state.status);
    try {
      const data = await api(`/api/jobs?${q}`);
      state.jobs = data.jobs;
      state.total = data.total;
      renderJobs();
      renderPager();
    } catch (e) {
      $("jobs-list").innerHTML = `<p class="muted">Couldn't load jobs. ${esc(e.message)}</p>`;
    }
  }

  function renderJobs() {
    const box = $("jobs-list");
    if (!state.jobs.length) {
      box.innerHTML = state.status
        ? `<p class="muted">No ${esc(state.status)} jobs.</p>`
        : `<p class="muted">No jobs yet. Start one to ingest a folder into a collection.</p>`;
      return;
    }

    box.innerHTML = state.jobs.map((j) => {
      const live = ["running", "claimed"].includes(j.status);
      const bits = [
        rel(j.created_at),
        j.collection_name ? `→ ${j.collection_name}` : "",
        `${j.folders.length} folder${j.folders.length === 1 ? "" : "s"}`,
        j.docs_ok ? `${num(j.docs_ok)} indexed` : "",
        j.docs_quarantined ? `${num(j.docs_quarantined)} quarantined` : "",
        j.docs_failed ? `${num(j.docs_failed)} failed` : "",
        j.docs_skipped ? `${num(j.docs_skipped)} skipped` : "",
        live && j.current_phase ? `· ${j.current_phase}` : "",
      ].filter(Boolean);

      return `<button class="job-row ${j.job_id === state.selected ? "is-selected" : ""}"
                data-id="${esc(j.job_id)}">
        <span class="job-name">${esc(j.name)}</span>
        <span class="badge" data-status="${esc(j.status)}">${esc(j.status)}</span>
        <span class="job-meta">${bits.map((b) => `<span>${esc(b)}</span>`).join("")}</span>
        ${live || j.progress_pct > 0
          ? `<span class="job-bar"><span style="width:${j.progress_pct}%"></span></span>` : ""}
      </button>`;
    }).join("");

    box.querySelectorAll(".job-row").forEach((r) =>
      r.addEventListener("click", () => select(r.dataset.id)));
  }

  function renderPager() {
    const start = state.page * state.size;
    const end = Math.min(start + state.jobs.length, state.total);
    $("page-info").textContent = state.total ? `${start + 1}–${end} of ${state.total}` : "";
    $("prev-page").disabled = state.page === 0;
    $("next-page").disabled = end >= state.total;
  }

  async function loadWorkerStatus() {
    const banner = $("worker-banner");
    const text = $("worker-banner-text");
    try {
      const { active, workers } = await api("/api/jobs/workers");
      if (active > 0) {
        // A healthy queue is the common case — don't occupy space for it.
        banner.hidden = true;
        return;
      }
      const anyQueued = state.jobs.some((j) => j.status === "queued")
        || (await api("/api/jobs?status=queued&limit=1")).total > 0;
      banner.hidden = false;
      banner.dataset.state = "down";
      text.innerHTML = anyQueued
        ? "No worker is running — queued jobs will not start. " +
          "Run <code>python -m jobs.worker</code> on the server."
        : "No worker is running. Jobs you start will stay queued until " +
          "<code>python -m jobs.worker</code> is running on the server.";
    } catch {
      banner.hidden = true;   // don't let a status-check failure block the page
    }
  }

  // ── collections panel ─────────────────────────────────────────────

  const GRAPH_LABEL = {
    none: "No graph", building: "Building…", ready: "Graph ready",
    stale: "Graph stale", failed: "Build failed",
  };

  let pendingGraphCollection = null;   // {id, name} awaiting confirm

  async function loadCollections() {
    const box = $("collections-list");
    try {
      const { collections } = await api("/api/collections");
      if (!collections.length) {
        box.innerHTML = '<p class="muted">No collections yet. Run an ' +
          'ingestion job to create one.</p>';
        return;
      }
      box.innerHTML = collections.map((c) => {
        const g = c.graph || { status: "none", stale: false };
        const label = GRAPH_LABEL[g.stale ? "stale" : g.status] || "No graph";
        const canBuild = g.status === "none" || g.status === "failed";
        const canRebuild = g.status === "ready" || g.status === "stale";
        const building = g.status === "building";
        return `<div class="coll-panel-row">
          <div class="coll-panel-main">
            <span class="coll-panel-name">${esc(c.name)}</span>
            <span class="coll-panel-meta">${num(c.documents)} docs · ${num(c.chunks)} chunks</span>
          </div>
          <span class="graph-badge" data-status="${esc(g.stale ? "stale" : g.status)}">
            ${esc(label)}${g.stats && g.stats.nodes ? ` · ${num(g.stats.nodes)} entities` : ""}
          </span>
          ${canBuild ? `<button class="btn small" data-graph-build="${esc(c.id)}" data-graph-name="${esc(c.name)}">Build graph</button>` : ""}
          ${canRebuild ? `<button class="btn small" data-graph-build="${esc(c.id)}" data-graph-name="${esc(c.name)}">Rebuild</button>` : ""}
          ${building ? '<button class="btn small" disabled>Building…</button>' : ""}
        </div>`;
      }).join("");
    } catch (e) {
      box.innerHTML = `<p class="muted">Couldn't load collections. ${esc(e.message)}</p>`;
    }
  }

  $("collections-list").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-graph-build]");
    if (!btn) return;
    pendingGraphCollection = { id: btn.dataset.graphBuild, name: btn.dataset.graphName };
    $("graph-confirm-text").textContent =
      `Build a knowledge graph for "${pendingGraphCollection.name}"? This helps ` +
      `answers connect related facts across documents in this collection — ` +
      `for example, recognising that two differently-worded mentions refer to ` +
      `the same person, company, or thing.`;
    $("graph-confirm").hidden = false;
    $("scrim").hidden = false;
  });

  function closeGraphConfirm() {
    $("graph-confirm").hidden = true;
    $("scrim").hidden = true;
    pendingGraphCollection = null;
  }
  $("graph-confirm-close").addEventListener("click", closeGraphConfirm);
  $("graph-confirm-cancel").addEventListener("click", closeGraphConfirm);

  $("graph-confirm-start").addEventListener("click", async () => {
    if (!pendingGraphCollection) return;
    const { id, name } = pendingGraphCollection;
    $("graph-confirm-start").disabled = true;
    try {
      const job = await api(`/api/collections/${id}/graph/build`, { method: "POST" });
      closeGraphConfirm();
      toast(`Building the graph for "${name}". Track progress in job history below.`);
      await loadCollections();
      await loadJobs();
      select(job.job_id);
    } catch (e) {
      toast(e.message, true);
    } finally {
      $("graph-confirm-start").disabled = false;
    }
  });

  async function loadStats() {
    try {
      const s = await api("/api/jobs/stats");
      $("stat-running").textContent = (s.by_status.running || 0) + (s.by_status.claimed || 0);
      $("stat-queued").textContent = s.by_status.queued || 0;
      $("stat-collections").textContent = num(s.collections);
      $("stat-docs").textContent = num(s.docs_ingested);
      $("stat-chunks").textContent = num(s.chunks_created);
      $("stat-failed").textContent = s.by_status.failed || 0;
    } catch { /* the strip is informational; a failed refresh isn't worth a toast */ }
  }

  // ── detail ─────────────────────────────────────────────────────────

  function select(id) {
    if (state.stream) { state.stream.close(); state.stream = null; }
    state.selected = id;
    state.seen = new Set();

    $("doc-list").hidden = true;
    $("doc-list").innerHTML = "";
    $("toggle-docs").textContent = "Show";

    const panel = $("detail");
    panel.hidden = false;
    if (isMobile()) {
      panel.dataset.mobile = "open";
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    $("log").innerHTML = "";
    renderJobs();

    api(`/api/jobs/${id}`).then(renderDetail).catch((e) => toast(e.message, true));
    openStream(id);
  }

  function closeDetail() {
    if (state.stream) { state.stream.close(); state.stream = null; }
    state.selected = null;
    const panel = $("detail");
    panel.hidden = true;
    delete panel.dataset.mobile;
    renderJobs();
  }

  function openStream(id) {
    const es = new EventSource(`/api/jobs/${id}/stream`);
    state.stream = es;
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "job") {
        renderDetail(msg.job);
        const i = state.jobs.findIndex((j) => j.job_id === msg.job.job_id);
        if (i >= 0) { state.jobs[i] = msg.job; renderJobs(); }
      } else if (msg.type === "events") {
        appendLog(msg.events);
      } else if (msg.type === "done") {
        es.close();
        state.stream = null;
        loadStats();
        loadJobs();
        loadCollections();
      }
    };
    // A dropped SSE connection isn't worth shouting about — the 6s list poll
    // keeps the page truthful either way.
    es.onerror = () => { es.close(); state.stream = null; };
  }

  function renderDetail(j) {
    if (j.job_id !== state.selected) return;

    $("detail-title").textContent = j.name;
    $("detail-meta").innerHTML = `
      <div><dt>Status</dt><dd><span class="badge" data-status="${esc(j.status)}">${esc(j.status)}</span></dd></div>
      <div><dt>Collection</dt><dd>${esc(j.collection_name || "—")}</dd></div>
      <div><dt>Started</dt><dd>${j.started_at ? rel(j.started_at) : "not yet"}</dd></div>
      <div><dt>Elapsed</dt><dd>${dur(j.started_at, j.finished_at)}</dd></div>
      <div><dt>Discovered</dt><dd>${num(j.files_discovered)}</dd></div>
      <div><dt>Indexed</dt><dd>${num(j.docs_ok)}</dd></div>
      <div><dt>Quarantined</dt><dd>${num(j.docs_quarantined)}</dd></div>
      <div><dt>Failed</dt><dd>${num(j.docs_failed)}</dd></div>
      <div><dt>Skipped</dt><dd>${num(j.docs_skipped)}</dd></div>
      <div><dt>Chunks</dt><dd>${num(j.chunks_total)}</dd></div>
      ${j.error ? `<div style="grid-column:1/-1"><dt>Error</dt><dd>${esc(j.error)}</dd></div>` : ""}`;

    // The folders a job ran against are set once at creation and never shown
    // again unless rendered explicitly here — the operator needs this to
    // sanity-check *what* built a collection, especially weeks later when a
    // re-run's folder list has drifted from what they remember configuring.
    $("detail-folders").innerHTML = j.folders.map((f) => {
      const filters = [];
      if (f.include && f.include.length) filters.push(`only ${f.include.join(", ")}`);
      if (f.exclude && f.exclude.length) filters.push(`excluding ${f.exclude.join(", ")}`);
      if (f.classification_hint) filters.push(`floor: ${f.classification_hint}`);
      return `<li>
        <code class="folder-path">${esc(f.path)}</code>
        ${f.recursive === false ? '<span class="folder-flag">this folder only</span>' : ""}
        ${filters.length ? `<span class="folder-filters">${esc(filters.join(" · "))}</span>` : ""}
      </li>`;
    }).join("") || '<li class="muted">No folders recorded.</li>';

    $("bar-fill").style.width = `${j.progress_pct}%`;
    $("bar-track").setAttribute("aria-valuenow", String(Math.round(j.progress_pct)));
    $("bar-caption").textContent =
      `${j.progress_pct.toFixed(1)}%${j.current_phase ? ` · ${j.current_phase}` : ""}`;

    const byPhase = Object.fromEntries(j.phases.map((p) => [p.phase, p]));
    $("phase-rail").innerHTML = PHASES.filter(([k]) => byPhase[k]).map(([k, label]) => {
      const p = byPhase[k];
      const count = p.items_total
        ? `${num(p.items_done)}/${num(p.items_total)}`
        : (p.status === "succeeded" ? "✓" : "");
      return `<li class="phase" data-status="${esc(p.status)}">
        <span class="dot"></span>
        <span>${esc(label)}</span>
        <span class="phase-count">${esc(count)}</span>
      </li>`;
    }).join("");

    const live = !["succeeded", "failed", "cancelled"].includes(j.status);
    $("cancel-btn").disabled = !live;
    $("rerun-btn").disabled = live;
    $("rerun-force-btn").disabled = live;
  }

  // Answers "what did THIS job actually produce" — the folder list above shows
  // intent, this shows outcome. Loaded on demand rather than with the rest of
  // the detail panel: a job over a large folder can produce thousands of rows,
  // and most visits to a job are just to check progress.
  async function toggleDocuments() {
    const box = $("doc-list");
    const btn = $("toggle-docs");
    if (!box.hidden) {
      box.hidden = true;
      btn.textContent = "Show";
      return;
    }
    if (!state.selected) return;

    btn.textContent = "Hide";
    box.hidden = false;
    box.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const data = await api(`/api/jobs/${state.selected}/documents`);
      if (!data.documents.length) {
        box.innerHTML = '<p class="muted">No documents recorded for this job yet.</p>';
        return;
      }
      box.innerHTML = data.documents.map((d) => `
        <div class="doc-row">
          <span class="doc-name" title="${esc(d.file_name)}">${esc(d.file_name)}</span>
          <span class="badge" data-status="${esc(d.status)}">${esc(d.status)}</span>
          <span class="doc-meta">${d.chunk_count ? `${num(d.chunk_count)} chunks` : ""}
            ${d.reason_code ? esc(d.reason_code) : ""}</span>
        </div>`).join("");
    } catch (e) {
      box.innerHTML = `<p class="muted">Couldn't load documents. ${esc(e.message)}</p>`;
    }
  }

  function appendLog(events) {
    const box = $("log");
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const html = events.filter((e) => !state.seen.has(e.event_id)).map((e) => {
      state.seen.add(e.event_id);
      return `<div class="log-line ${esc(e.level)}">` +
        `<span class="log-ts">${esc(new Date(e.ts).toLocaleTimeString())}</span>` +
        `${esc(e.message)}</div>`;
    }).join("");
    if (!html) return;
    box.insertAdjacentHTML("beforeend", html);
    while (box.childElementCount > 600) box.removeChild(box.firstElementChild);
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  // ── new job ────────────────────────────────────────────────────────

  function addFolder(spec = {}) {
    const accept = (window.FILETYPES && window.FILETYPES.accept) || "";
    $("folders").insertAdjacentHTML("beforeend", `
      <div class="folder">
        <input type="text" class="field path" placeholder="data/incoming/policies"
               value="${esc(spec.path || "")}">
        <input type="text" class="field inc" placeholder="All types"
               title="Comma-separated extensions. Blank means everything supported: ${esc(accept)}"
               value="${esc((spec.include || []).join(","))}">
        <input type="text" class="field exc" placeholder="Exclude e.g. f2, f3"
               title="Comma-separated subfolder names to skip (with everything under them), e.g. 'f2, f3'. Wildcards like *.tmp also work."
               value="${esc((spec.exclude || []).join(","))}">
        <button class="icon-btn rm" aria-label="Remove folder">✕</button>
        <p class="preview"></p>
      </div>`);

    const row = $("folders").lastElementChild;
    row.querySelector(".rm").addEventListener("click", () => row.remove());

    const path = row.querySelector(".path");
    const inc = row.querySelector(".inc");
    const exc = row.querySelector(".exc");
    const out = row.querySelector(".preview");
    let timer;

    async function preview() {
      const p = path.value.trim();
      if (!p) { out.textContent = ""; return; }
      out.classList.remove("err");
      out.textContent = "Checking…";
      try {
        const q = new URLSearchParams({ path: p });
        if (inc.value.trim()) q.set("include", inc.value.trim());
        if (exc.value.trim()) q.set("exclude", exc.value.trim());
        const r = await api(`/api/folders/preview?${q}`);
        const top = Object.entries(r.by_extension).slice(0, 5)
          .map(([k, v]) => `${k} ${v}`).join("  ");
        out.textContent =
          `${num(r.files_ingestible)} of ${num(r.files_seen)} files ingestible` +
          (r.files_excluded ? ` · ${num(r.files_excluded)} excluded by folder rules` : "") +
          (r.archives ? ` · ${r.archives} archive(s)` : "") + (top ? ` · ${top}` : "");
      } catch (e) {
        out.classList.add("err");
        out.textContent = e.message;
      }
    }

    path.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(preview, 500); });
    inc.addEventListener("change", preview);
    exc.addEventListener("change", preview);
    if (spec.path) preview();
  }

  function folders() {
    return [...document.querySelectorAll(".folder")].map((row) => {
      const path = row.querySelector(".path").value.trim();
      if (!path) return null;
      const inc = row.querySelector(".inc").value.trim();
      const exc = row.querySelector(".exc").value.trim();
      return {
        path, recursive: true,
        include: inc ? inc.split(",").map((s) => s.trim()).filter(Boolean) : [],
        exclude: exc ? exc.split(",").map((s) => s.trim()).filter(Boolean) : [],
      };
    }).filter(Boolean);
  }

  function openSheet() {
    $("scrim").hidden = false;
    $("new-job").hidden = false;
    $("new-job-error").hidden = true;
    if (!document.querySelector(".folder")) addFolder();
    $("job-name").focus();
  }

  function closeSheet() {
    $("scrim").hidden = true;
    $("new-job").hidden = true;
  }

  function syncCollectionName() {
    // The name box only matters when creating a new collection; showing it
    // beside a chosen existing one implies it would rename that collection.
    $("collection-name").hidden = Boolean($("collection").value);
  }

  async function startJob() {
    const err = $("new-job-error");
    const fail = (m) => { err.textContent = m; err.hidden = false; };

    const name = $("job-name").value.trim();
    if (!name) return fail("Give the job a name so you can find it later.");

    const setId = $("folder-set").value;
    const dirs = folders();
    if (!dirs.length && !setId) return fail("Add at least one folder, or pick a folder set.");

    const collectionId = $("collection").value;
    const collectionName = $("collection-name").value.trim();
    if (!collectionId && !collectionName && !name) {
      return fail("Name the collection this job should build.");
    }

    $("start-job").disabled = true;
    try {
      const job = await api("/api/jobs", {
        method: "POST",
        body: JSON.stringify({
          name,
          collection_id: collectionId || null,
          collection_name: collectionName || name,
          set_id: setId || null,
          folders: dirs,
          options: {
            workers: parseInt($("workers").value, 10) || 0,
            embed_concurrency: parseInt($("embed-conc").value, 10) || 4,
            enrich_mode: $("enrich").value,
            max_files: parseInt($("max-files").value, 10) || 0,
            force: $("opt-force").checked,
            skip_images: $("opt-skip-images").checked,
          },
        }),
      });
      closeSheet();
      toast(`Queued "${job.name}". The worker picks it up within a few seconds.`);
      await loadJobs();
      await loadStats();
      await loadWorkerStatus();
      select(job.job_id);
    } catch (e) {
      fail(e.message);
    } finally {
      $("start-job").disabled = false;
    }
  }

  async function saveSet() {
    const dirs = folders();
    if (!dirs.length) return toast("Add a folder before saving the set.", true);
    const name = prompt("Name this folder set:", $("job-name").value.trim() || "My folders");
    if (!name) return;
    try {
      await api("/api/folder-sets", {
        method: "POST",
        body: JSON.stringify({ name, description: "", folders: dirs }),
      });
      toast(`Saved "${name}". Reload to see it in the list.`);
    } catch (e) { toast(e.message, true); }
  }

  async function loadSet(setId) {
    if (!setId) return;
    try {
      const { sets } = await api("/api/folder-sets");
      const fs = sets.find((s) => s.set_id === setId);
      if (!fs) return;
      $("folders").innerHTML = "";
      fs.folders.forEach(addFolder);
      if (!$("job-name").value.trim()) $("job-name").value = fs.name;
    } catch (e) { toast(e.message, true); }
  }

  // ── actions ────────────────────────────────────────────────────────

  async function cancelJob() {
    if (!state.selected) return;
    if (!confirm("Cancel this job? Documents already indexed stay in the collection, "
      + "and re-running resumes from where this stops.")) return;
    try {
      await api(`/api/jobs/${state.selected}/cancel`, { method: "POST" });
      toast("Cancelling at the next document boundary.");
    } catch (e) { toast(e.message, true); }
  }

  async function rerun(force) {
    if (!state.selected) return;
    if (force && !confirm("Rebuilding reprocesses every document, ignoring what is "
      + "already indexed. Continue?")) return;
    try {
      const job = await api(`/api/jobs/${state.selected}/rerun?force=${force}`, { method: "POST" });
      toast(`Queued "${job.name}".`);
      await loadJobs();
      select(job.job_id);
    } catch (e) { toast(e.message, true); }
  }

  // ── boot ───────────────────────────────────────────────────────────

  document.addEventListener("DOMContentLoaded", () => {
    $("new-job-btn").addEventListener("click", openSheet);
    $("new-job-close").addEventListener("click", closeSheet);
    $("scrim").addEventListener("click", closeSheet);
    $("add-folder").addEventListener("click", () => addFolder());
    $("start-job").addEventListener("click", startJob);
    $("save-set").addEventListener("click", saveSet);
    $("folder-set").addEventListener("change", (e) => loadSet(e.target.value));
    $("collection").addEventListener("change", syncCollectionName);

    $("detail-close").addEventListener("click", closeDetail);
    $("toggle-docs").addEventListener("click", toggleDocuments);
    $("cancel-btn").addEventListener("click", cancelJob);
    $("rerun-btn").addEventListener("click", () => rerun(false));
    $("rerun-force-btn").addEventListener("click", () => rerun(true));

    $("refresh-btn").addEventListener("click", () => { loadJobs(); loadStats(); });
    $("status-filter").addEventListener("change", (e) => {
      state.status = e.target.value;
      state.page = 0;
      loadJobs();
    });
    $("prev-page").addEventListener("click", () => { state.page = Math.max(0, state.page - 1); loadJobs(); });
    $("next-page").addEventListener("click", () => { state.page += 1; loadJobs(); });

    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (!$("new-job").hidden) closeSheet();
      else if (!$("detail").hidden) closeDetail();
    });

    syncCollectionName();
    loadJobs();
    loadStats();
    loadWorkerStatus();
    loadCollections();

    // Stops while the tab is hidden so a backgrounded mobile tab isn't polling
    // all afternoon.
    setInterval(() => {
      if (document.visibilityState === "visible") {
        loadJobs(); loadStats(); loadWorkerStatus(); loadCollections();
      }
    }, 6000);
  });
})();
