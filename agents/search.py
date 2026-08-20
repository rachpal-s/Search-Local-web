"""agents/search.py — Agent 2: Search Engine.

Queries a local SearXNG instance and returns the top result URLs.
"""
import json
from typing import Any, Dict
import os
import httpx
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from workflow.state import SearchTaskState

# opentelemetry-api (distinct from opentelemetry-sdk) is designed to be
# always importable and a safe no-op when no TracerProvider has been
# configured — with Phoenix tracing off (the default), start_as_current_span()
# below creates an inert span and this function behaves exactly as it did
# before observability.py existed. opentelemetry-sdk is an unconditional line
# in requirements.txt now, and opentelemetry-api installs transitively with
# it, so this import should always succeed in a correctly deployed instance —
# but "correctly deployed" has not been a safe assumption anywhere else in
# this codebase's history, so the ImportError fallback stays anyway. A search
# call degrading to untraced-but-working is a far better failure mode than
# search.py — and therefore the whole graph — failing to import at all.
try:
    from opentelemetry import trace
    _tracer = trace.get_tracer(__name__)
except ImportError:  # pragma: no cover
    class _NullSpan:
        def set_attribute(self, *_a, **_kw): pass
        def __enter__(self): return self
        def __exit__(self, *_a): return False

    class _NullTracer:
        def start_as_current_span(self, *_a, **_kw): return _NullSpan()

    _tracer = _NullTracer()

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")
MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 15.0


async def _fetch_search_results(query: str) -> list[str]:
    """The raw HTTP call, isolated in its own function.

    This is the one call site in the app that bypasses LangChain entirely —
    a plain httpx request, invisible to Phoenix's LangChain auto-instrumentor
    no matter how tracing is configured elsewhere (that instrumentor only
    sees LangChain's own callback/run system, which this call never touches).
    The explicit span below is what gives it visibility at all — timing, the
    query, and (via set_attribute) the result count — as a proper child span
    under whatever LangGraph node span is currently active.
    """
    with _tracer.start_as_current_span("searxng_http_call") as span:
        span.set_attribute("search.query", query)
        results = await _do_fetch(query)
        span.set_attribute("search.result_count", len(results))
        return results


async def _do_fetch(query: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            SEARXNG_URL, params={"q": query, "format": "json"},
            timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return [item.get("url") for item in data.get("results", [])
               if item.get("url")][:MAX_RESULTS]


async def search_node(state: SearchTaskState) -> Dict[str, Any]:
    """Query the local SearXNG instance and return the top result URLs."""
    query = state["query"]
    print(f"\n[SEARCH] 🔍 Executing search query: '{query}'")
    logs = [f"🔍 Searching: '{query}'"]

    try:
        print(f"[SEARCH] ⏳ Requesting search results from local SearXNG ({SEARXNG_URL})...")
        urls = await _fetch_search_results(query)
        result = {"subject": query, "results": urls}
        success_log = f"✅ Search successful. Retrieved {len(urls)} top results."
        logs.append(success_log) # Add to UI logs
        print(f"[SEARCH] {success_log}")

    except Exception as e:
        error_log = f"❌ Failed to fetch from SearXNG: {e}"
        logs.append(error_log) # Add to UI logs
        print(f"[SEARCH] {error_log}")
        result = {
            "subject": query,
            "error": f"Failed to connect to local SearXNG container: {str(e)}",
        }

    return {"context": [f"Search Engine Result: {json.dumps(result)}"], "action_logs": logs}
