(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function traceDetailUrl(traceId) {
    if (!traceId || !window.PHOENIX_ENDPOINT || !window.PHOENIX_PROJECT) return null;
    // {endpoint}/projects/{project}/traces/{trace_id} — confirmed pattern
    // (Arize community support), but the project segment there uses
    // Phoenix's INTERNAL project id, not necessarily the plain name used
    // here. cfg.phoenix_project already works for querying spans in this
    // app, so it's used as-is — genuinely unverified for the URL path
    // specifically. If this 404s in your Phoenix UI, that's the signal an
    // extra name->id lookup step is needed, not a sign the trace_id is wrong.
    const base = window.PHOENIX_ENDPOINT.replace(/\/$/, "");
    return `${base}/projects/${encodeURIComponent(window.PHOENIX_PROJECT)}/traces/${encodeURIComponent(traceId)}`;
  }

  async function load() {
    const hours = $("hours").value;
    const box = $("conv-table");
    box.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const res = await fetch(`/api/telemetry/conversations?hours=${hours}`);
      const data = await res.json();

      const banner = $("status-banner");
      if (!data.available) {
        banner.hidden = false;
        banner.innerHTML = `<span class="worker-dot"></span> ${esc(data.reason || "Telemetry unavailable.")}`;
        box.innerHTML = "";
        return;
      }
      banner.hidden = true;

      // This line IS the viability check.
      const note = $("viability-note");
      if (data.total_root_spans) {
        const pct = Math.round(100 * data.matched_to_conversation / data.total_root_spans);
        note.textContent = `${data.matched_to_conversation} of ${data.total_root_spans} root spans `
          + `matched to a conversation_id (${pct}%).`
          + (data.matched_to_conversation === 0
            ? " conversation_id is NOT reaching spans yet — the metadata attribute name "
              + "guessed in routers/telemetry.py needs checking against what Phoenix actually stored."
            : "");
      } else {
        note.textContent = data.note || "";
      }

      if (!data.conversations.length) {
        box.innerHTML = '<p class="muted">No conversations found in this window.</p>';
        return;
      }

      box.innerHTML = `<table class="telemetry-table">
        <thead><tr><th>Conversation</th><th>Turn</th><th>Latency</th>
          <th>Spans</th><th>Breakdown</th><th></th><th></th></tr></thead>
        <tbody>${data.conversations.map((c) => c.turns.map((t, i) => {
          const traceUrl = traceDetailUrl(t.trace_id);
          return `
          <tr class="${t.had_error ? "has-error" : ""}">
            <td>${i === 0 ? `<code>${esc(c.conversation_id.slice(0, 12))}…</code> (${c.turn_count})` : ""}</td>
            <td>${t.start_time ? new Date(t.start_time).toLocaleString() : "—"}</td>
            <td>${t.latency_ms != null ? Math.round(t.latency_ms) + " ms" : "—"}</td>
            <td>${t.span_count}</td>
            <td>${t.span_breakdown.map((b) =>
              `<span class="span-chip">${esc(b.name)} ×${b.count} (${Math.round(b.total_ms)}ms)</span>`
            ).join(" ") || "—"}</td>
            <td>${traceUrl ? `<a href="${esc(traceUrl)}" target="_blank" title="Open this exact turn in Phoenix">trace →</a>` : "—"}</td>
            <td>${i === 0 ? `<a href="/#${esc(c.conversation_id)}" target="_blank">open chat →</a>` : ""}</td>
          </tr>`;
        }).join("")).join("")}
        </tbody></table>`;
    } catch (e) {
      box.innerHTML = `<p class="muted">Failed to load: ${esc(e.message)}</p>`;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("refresh-btn").addEventListener("click", load);
    $("hours").addEventListener("change", load);
    load();
  });
})();
