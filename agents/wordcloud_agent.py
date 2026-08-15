"""agents/wordcloud_agent.py — Word-cloud renderer (in-process).

Same family as agents/mermaid_lite_agent.py: pure Python, no external
process, renders SVG directly via agents/lib/wordcloud_lite.py. Node
shape mirrors mermaid_lite_node deliberately — same context/action_logs
return contract, same error-reporting pattern — so the supervisor prompt
and routing didn't need any changes beyond the registry entry itself.
"""
import os
import uuid
from typing import Any, Dict

from agents.lib.wordcloud_lite import render_wordcloud_to_svg
from workflow.state import WordCloudTaskState

DOWNLOAD_DIR = "static/downloads"


def wordcloud_node(state: WordCloudTaskState) -> Dict[str, Any]:
    """Renders a word-frequency cloud from free text to SVG in-process."""
    text = state.get("text", "")
    logs = ["[WORDCLOUD] 🎨 Rendering word cloud in-process..."]
    print("\n[WORDCLOUD] 🎨 Rendering word cloud in-process...")

    if not text or not text.strip():
        error_msg = "[WORDCLOUD] ❌ Error: No text provided."
        logs.append(error_msg)
        print(error_msg)
        return {"context": ["Word Cloud Agent Error: No text provided."], "action_logs": logs}

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    svg_filename = f"wordcloud_{file_id}.svg"
    svg_path = os.path.join(DOWNLOAD_DIR, svg_filename)
    web_path = f"/static/downloads/{svg_filename}"

    try:
        svg = render_wordcloud_to_svg(text)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        logs.append(f"[WORDCLOUD] 💾 Saved to {svg_path}")

        success_message = (
            f"✅ SUCCESS: Word cloud rendered. You MUST provide this exact image "
            f"link to the user: ![Word Cloud]({web_path})"
        )
        logs.append("[WORDCLOUD] ✅ Diagram generated successfully.")
        print("[WORDCLOUD] ✅ Diagram generated successfully.")
        return {"context": [success_message], "action_logs": logs}

    except ValueError as e:
        # Almost always "not enough distinct words" — too short an input, or
        # text that's nearly all stopwords. Surfaced so the supervisor knows
        # to gather more source material rather than blindly retrying.
        err_msg = f"[WORDCLOUD] ⚠️ Insufficient input: {e}"
        logs.append(err_msg)
        print(err_msg)
        return {
            "context": [
                f"Word Cloud Agent Error: {e}. Gather more source text (e.g. "
                f"via scraper/search/doc_retriever) before retrying."
            ],
            "action_logs": logs,
        }
    except Exception as e:
        err_msg = f"[WORDCLOUD] ❌ Unexpected Error: {str(e)}"
        logs.append(err_msg)
        print(err_msg)
        return {"context": [f"Word Cloud Agent Error: {str(e)}"], "action_logs": logs}
