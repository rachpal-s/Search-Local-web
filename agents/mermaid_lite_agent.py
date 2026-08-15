"""agents/mermaid_lite_agent.py — Flowchart-only Mermaid renderer (in-process).

Sibling to agents/mermaid_agent.py, not a replacement. That one shells out to
`mmdc` (Node + Chromium) and covers the full Mermaid spec; this one parses a
flowchart subset directly in Python via agents/lib/mermaid_lite.py and has no
external process to install, find on PATH, or fail silently on Windows.

The registry description is what actually steers the supervisor between the
two at runtime — see workflow/registry.py. This node's job is just: render,
save, report back in the same shape the supervisor already expects from the
mmdc-based agent, so no prompt or routing changes are needed beyond the
registry entry itself.
"""
import os
import re
import uuid
from typing import Any, Dict

from agents.lib.mermaid_lite import render_mermaid_to_svg
from workflow.state import MermaidTaskState

DOWNLOAD_DIR = "static/downloads"


def mermaid_lite_node(state: MermaidTaskState) -> Dict[str, Any]:
    """Renders a Mermaid flowchart script to SVG in-process and returns the link."""
    script = state.get("script", "")
    logs = ["[MERMAID_LITE] 🎨 Rendering flowchart in-process (no Node/mmdc)..."]
    print("\n[MERMAID_LITE] 🎨 Rendering flowchart in-process...")

    script = re.sub(r"^```(?:mermaid)?\s*", "", script.strip(), flags=re.IGNORECASE)
    script = re.sub(r"\s*```$", "", script.strip())

    if not script:
        error_msg = "[MERMAID_LITE] ❌ Error: No script provided."
        logs.append(error_msg)
        print(error_msg)
        return {"context": ["Mermaid Lite Agent Error: No script provided."], "action_logs": logs}

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    svg_filename = f"flowchart_{file_id}.svg"
    svg_path = os.path.join(DOWNLOAD_DIR, svg_filename)
    web_path = f"/static/downloads/{svg_filename}"

    try:
        svg = render_mermaid_to_svg(script)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        logs.append(f"[MERMAID_LITE] 💾 Saved to {svg_path}")

        success_message = (
            f"✅ SUCCESS: Flowchart rendered. You MUST provide this exact image "
            f"link to the user: ![Flowchart]({web_path})"
        )
        logs.append("[MERMAID_LITE] ✅ Diagram generated successfully.")
        print("[MERMAID_LITE] ✅ Diagram generated successfully.")
        return {"context": [success_message], "action_logs": logs}

    except ValueError as e:
        # Almost always means the script used syntax this parser doesn't cover
        # (sequenceDiagram, classDiagram, subgraph, etc) rather than a bug —
        # surfaced to the supervisor so it knows to retry with mermaid_generator.
        err_msg = f"[MERMAID_LITE] ⚠️ Unsupported syntax: {e}"
        logs.append(err_msg)
        print(err_msg)
        return {
            "context": [
                f"Mermaid Lite Agent Error: {e}. This renderer only supports "
                f"flowchart/graph syntax — retry with 'mermaid_generator' for "
                f"sequence, class, state, ER, or gantt diagrams."
            ],
            "action_logs": logs,
        }
    except Exception as e:
        err_msg = f"[MERMAID_LITE] ❌ Unexpected Error: {str(e)}"
        logs.append(err_msg)
        print(err_msg)
        return {"context": [f"Mermaid Lite Agent Error: {str(e)}"], "action_logs": logs}
