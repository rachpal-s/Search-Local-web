"""agents/extractor.py — Agent 1: Extractor Engine.

Extracts URLs embedded directly in raw user text (no network calls).
"""
import json
import re
from typing import Any, Dict

from workflow.state import ExtractTaskState

URL_PATTERN = r'(https?://[^\s]+)'


def extractor_node(state: ExtractTaskState) -> Dict[str, Any]:
    """Extract embedded reference URLs from user text."""
    text = state["text"]
    print(f"\n[EXTRACTOR] 🚀 Extracting URLs from text: '{text[:60]}...'")
    logs = [f"\n[EXTRACTOR] 🚀 Extracting URLs from text: '{text[:60]}...'"]

    urls = re.findall(URL_PATTERN, text)

    result = {
        "subject": text,
        "references": urls,
    }
    success_msg = f"[EXTRACTOR] ✅ Done. Found {len(urls)} reference URLs."
    logs.append(success_msg)
    print(success_msg)
    return {"context": [f"Extractor Engine Result: {json.dumps(result)}"], "action_logs": logs}
