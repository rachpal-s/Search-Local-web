"""routers/telemetry.py — a small dashboard on top of Phoenix's stored traces,
mapped to this app's own conversation_id.

Why this exists rather than just using Phoenix's own UI: Phoenix shows generic
span trees, built for any LLM app. It has no idea this app has a
supervisor/critic loop — it can't show "average critic score" or "how often
did a conversation hit max_loops" without you building exactly this.

This is explicitly a FIRST CUT to check viability, not a finished feature.
Two things are genuinely unverified until tested against a real Phoenix
instance:

  1. Whether conversation_id actually reaches spans as a queryable attribute
     at all. tracing_context() (OpenInference's using_session/using_user)
     is still reverted after the answer-loss bug — the ONLY path
     conversation_id has into Phoenix right now is LangChain's own
     `config={"metadata": {...}}`, passed via run_config() in main.py,
     which was never reverted. Whether Phoenix's LangChain instrumentor
     surfaces that as a span attribute is exactly what this page will show.
  2. The exact column/attribute naming get_spans_dataframe() returns —
     written defensively (checks for several plausible names) rather than
     assuming one, specifically because guessing wrong on Phoenix API
     details has already cost real debugging time earlier in this build.

Fails open at every layer: no phoenix package, no reachable server, no
spans yet, wrong column names — every one of these returns an empty,
explained result, never a 500.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import get_settings

cfg = get_settings()
router = APIRouter(tags=["telemetry"])
templates = Jinja2Templates(directory="templates")

# Attribute/column names OpenInference's LangChain instrumentor MIGHT use for
# LangChain's own config={"metadata": {...}}. Checked in order; first match
# wins. This list is exactly the uncertainty named in the module docstring —
# expect to trim it to one entry once tested against a real instance.
_CONVERSATION_ID_CANDIDATES = [
    "attributes.metadata.conversation_id",
    "attributes.metadata",          # sometimes a dict/JSON blob, handled specially below
    "metadata.conversation_id",
    "attributes.session.id",        # if using_session() ever gets reintroduced
    "attributes.tag.tags",          # our tags list includes "conversation:<id>"
]


def _ascii_safe(value: str) -> str:
    return value.encode("ascii", errors="ignore").decode("ascii")


_project_id_cache: dict[str, str] = {}


def _resolve_project_id(project_name: str) -> str | None:
    """Phoenix's trace-detail URL needs the internal GraphQL node id, not the
    plain project name — confirmed the hard way: 'Unknown node: Web Pulse'.
    This is Arize's own documented resolution query (community support
    example), done as a raw POST rather than through phoenix.client.Client,
    since that SDK's surface for this specifically wasn't confirmed and a
    plain HTTP call is one less thing to guess wrong about.

    Cached per project name for the process lifetime — this practically
    never changes, and paginating all projects on every dashboard load
    would be wasteful.
    """
    if project_name in _project_id_cache:
        return _project_id_cache[project_name]
    if not cfg.phoenix_tracing_enabled:
        return None

    import json
    import urllib.request

    query = """
    query ($after: String = null) {
      projects(after: $after) {
        edges { project: node { id name } }
        pageInfo { hasNextPage endCursor }
      }
    }
    """
    after = None
    try:
        for _ in range(20):   # hard cap — never loop forever on a paging bug
            body = json.dumps({"query": query, "variables": {"after": after}}).encode()
            req = urllib.request.Request(
                _ascii_safe(cfg.phoenix_endpoint.rstrip("/")) + "/graphql",
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())

            conn = data.get("data", {}).get("projects", {})
            for edge in conn.get("edges", []):
                node = edge.get("project", {})
                if node.get("name") == project_name:
                    _project_id_cache[project_name] = node["id"]
                    return node["id"]

            page = conn.get("pageInfo", {})
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
    except Exception as e:  # noqa: BLE001
        print(f"[telemetry] ⚠️ Could not resolve Phoenix project id for "
              f"'{project_name}': {e}")
        return None

    print(f"[telemetry] ⚠️ No Phoenix project named '{project_name}' found — "
          f"per-trace links will be disabled until this matches a real project.")
    return None


def _get_phoenix_client():
    """Returns a connected client or None. Never raises."""
    if not cfg.phoenix_tracing_enabled:
        return None
    try:
        from phoenix.client import Client
        return Client(base_url=_ascii_safe(cfg.phoenix_endpoint))
    except Exception as e:  # noqa: BLE001
        print(f"[telemetry] ⚠️ Could not create Phoenix client: {e}")
        return None


def _extract_conversation_id(row: dict) -> str | None:
    """Best-effort extraction across whichever attribute shape actually
    shows up — see the module docstring for why this isn't just one lookup."""
    import json

    for key in _CONVERSATION_ID_CANDIDATES:
        val = row.get(key)
        if val is None:
            continue
        if key.endswith("conversation_id") and isinstance(val, str) and val:
            return val
        if key.endswith("metadata") and val:
            try:
                meta = json.loads(val) if isinstance(val, str) else val
                if isinstance(meta, dict) and meta.get("conversation_id"):
                    return meta["conversation_id"]
            except Exception:  # noqa: BLE001
                pass
        if key.endswith("tags") and val:
            tags = val if isinstance(val, list) else [val]
            for t in tags:
                if isinstance(t, str) and t.startswith("conversation:"):
                    return t.split(":", 1)[1]
    return None


def fetch_conversation_rollups(hours: int = 24, limit: int = 500) -> dict:
    """Every root span (turn) kept individually — not averaged — plus a
    child-span latency breakdown per turn (supervisor/critic/scraper/etc.),
    joined by trace_id: all spans in one trace share the root's trace_id,
    which is how a turn's children get matched back to it.
    """
    client = _get_phoenix_client()
    if client is None:
        return {"available": False, "reason": "Phoenix tracing is disabled or "
                "the phoenix package is not installed.", "conversations": []}

    try:
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
        roots = client.spans.get_spans_dataframe(
            project_identifier=cfg.phoenix_project,
            start_time=start, limit=limit, root_spans_only=True)
        all_spans = client.spans.get_spans_dataframe(
            project_identifier=cfg.phoenix_project,
            start_time=start, limit=limit * 20, root_spans_only=False)
    except Exception as e:  # noqa: BLE001
        return {"available": False,
               "reason": f"Could not query Phoenix: {type(e).__name__}: {e}",
               "conversations": []}

    if roots is None or len(roots) == 0:
        return {"available": True, "conversations": [],
               "note": "Connected to Phoenix, but no spans found in this "
                       "window. Send a chat message with tracing on, then retry."}

    root_rows = roots.to_dict("records")
    child_rows = all_spans.to_dict("records") if all_spans is not None else []
    root_span_ids = {r.get("context.span_id") or r.get("span_id") for r in root_rows}

    # trace_id -> [child span dicts], for the per-turn breakdown. Root spans
    # themselves show up again here too (root_spans_only=False includes
    # everything) — excluded by span_id, or they'd double-count as a
    # spurious "unknown"/root-named entry in every turn's breakdown.
    children_by_trace: dict[str, list[dict]] = {}
    for row in child_rows:
        span_id = row.get("context.span_id") or row.get("span_id")
        if span_id in root_span_ids:
            continue
        tid = row.get("context.trace_id") or row.get("trace_id")
        if not tid:
            continue
        name = row.get("name") or row.get("attributes.name") or "unknown"
        st, et = row.get("start_time"), row.get("end_time")
        ms = None
        if st is not None and et is not None:
            try:
                ms = round((et - st).total_seconds() * 1000, 1)
            except Exception:  # noqa: BLE001
                pass
        children_by_trace.setdefault(tid, []).append({
            "name": name, "span_kind": row.get("span_kind"), "latency_ms": ms,
        })

    unmatched = 0
    by_conv: dict[str, list[dict]] = {}

    for row in root_rows:
        conv_id = _extract_conversation_id(row)
        if not conv_id:
            unmatched += 1
            continue

        tid = row.get("context.trace_id") or row.get("trace_id")
        start_time, end_time = row.get("start_time"), row.get("end_time")
        latency_ms = None
        if start_time is not None and end_time is not None:
            try:
                latency_ms = round((end_time - start_time).total_seconds() * 1000, 1)
            except Exception:  # noqa: BLE001
                pass

        status = row.get("status_code") or row.get("attributes.status_code")
        spans = children_by_trace.get(tid, [])
        # Aggregate children by name — one line per node type (supervisor,
        # critic, scraper, ...) rather than every individual call, since a
        # multi-loop turn can have many scraper spans and the useful
        # question is usually "how much total time did scraping cost",
        # not each individual call.
        by_name: dict[str, dict] = {}
        for s in spans:
            b = by_name.setdefault(s["name"], {"name": s["name"], "count": 0, "total_ms": 0.0})
            b["count"] += 1
            if s["latency_ms"] is not None:
                b["total_ms"] += s["latency_ms"]
        span_breakdown = sorted(
            [{"name": k, "count": v["count"], "total_ms": round(v["total_ms"], 0)}
             for k, v in by_name.items()],
            key=lambda x: -x["total_ms"])

        by_conv.setdefault(conv_id, []).append({
            "trace_id": tid,
            "start_time": start_time.isoformat() if start_time is not None else None,
            "latency_ms": latency_ms,
            "had_error": bool(status and "ERROR" in str(status).upper()),
            "span_count": len(spans),
            "span_breakdown": span_breakdown,
        })

    conversations = []
    for conv_id, turns in by_conv.items():
        turns.sort(key=lambda t: t["start_time"] or "", reverse=True)
        conversations.append({
            "conversation_id": conv_id,
            "turn_count": len(turns),
            "turns": turns,
            "last_seen": turns[0]["start_time"] if turns else None,
        })
    conversations.sort(key=lambda c: c["last_seen"] or "", reverse=True)

    return {
        "available": True,
        "conversations": conversations,
        "total_root_spans": len(root_rows),
        "matched_to_conversation": len(root_rows) - unmatched,
        "unmatched": unmatched,
    }


@router.get("/telemetry", response_class=HTMLResponse)
async def telemetry_page(request: Request):
    return templates.TemplateResponse(request, "telemetry.html", {
        "phoenix_endpoint": cfg.phoenix_endpoint,
        # The RESOLVED internal id, not the raw name — Phoenix's trace URL
        # rejects the plain project name ("Unknown node: <name>"). None on
        # any failure, so telemetry.js's traceDetailUrl() correctly hides
        # the link rather than emitting a URL already known to 404.
        "phoenix_project_id": _resolve_project_id(cfg.phoenix_project),
        "phoenix_enabled": cfg.phoenix_tracing_enabled,
    })


@router.get("/api/telemetry/conversations")
async def telemetry_conversations(hours: int = Query(24, ge=1, le=720),
                                  limit: int = Query(500, ge=1, le=5000)):
    return fetch_conversation_rollups(hours=hours, limit=limit)


@router.get("/api/telemetry/status")
async def telemetry_status():
    """Cheap standalone check — is tracing on, is Phoenix reachable — without
    pulling any span data. Useful to hit first when debugging."""
    client = _get_phoenix_client()
    return {
        "phoenix_tracing_enabled": cfg.phoenix_tracing_enabled,
        "phoenix_endpoint": cfg.phoenix_endpoint,
        "phoenix_client_created": client is not None,
    }
