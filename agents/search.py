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

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")
MAX_RESULTS = 5
REQUEST_TIMEOUT_SECONDS = 15.0


async def search_node(state: SearchTaskState) -> Dict[str, Any]:
    """Query the local SearXNG instance and return the top result URLs."""
    query = state["query"]
    print(f"\n[SEARCH] 🔍 Executing search query: '{query}'")
    logs = [f"🔍 Searching: '{query}'"]

    params = {"q": query, "format": "json"}

    try:
        print(f"[SEARCH] ⏳ Requesting search results from local SearXNG ({SEARXNG_URL})...")
        async with httpx.AsyncClient() as client:
            response = await client.get(SEARXNG_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()

            urls = [item.get("url") for item in data.get("results", []) if item.get("url")][:MAX_RESULTS]
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
