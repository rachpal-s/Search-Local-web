"""workflow/registry.py — Dynamic registry of specialist worker agents.

Adding a new worker to the graph only requires: (1) an agent module
under agents/ exposing a `<name>_node` function, (2) a matching
TypedDict payload in workflow/state.py, and (3) an entry here. The
graph builder and supervisor prompt both consume this registry.
"""
from agents.doc_retriever import doc_retriever_node
from agents.extractor import extractor_node
from agents.search import search_node
from agents.web_scraper import scraper_node
from agents.youtube_downloader import youtube_downloader_node
from agents.mermaid_agent import mermaid_agent_node
from agents.mermaid_lite_agent import mermaid_lite_node
from agents.wordcloud_agent import wordcloud_node

AGENT_REGISTRY = {
    "doc_retriever": {
        "description": (
            "Searches documents the user uploaded to THIS conversation "
            "(hybrid semantic + keyword search over the attached files). "
            "Use this FIRST whenever the query refers to attachments, 'this "
            "document', 'the file', 'the contract/report/policy', or when "
            "attached files are listed in the prompt. Reformulate the user's "
            "wording into the terms likely to appear in the document itself. "
            "Requires payload format: {'query': '<what to look up>'}"
        ),
        "func": doc_retriever_node,
    },
    "extractor": {
        "description": "Extracts source URLs embedded directly in the text. Requires payload format: {'text': '<text>'}",
        "func": extractor_node,
    },
    "search": {
        "description": "Searches the web for top search results using SearXNG. Requires payload format: {'query': '<search term>'}",
        "func": search_node,
    },
    "scraper": {
        "description": "Scrapes content from a specific web URL. Requires payload format: {'url': '<url>'}",
        "func": scraper_node,
    },
    "youtube_downloader": {
        "description": "Searches for a music or video query on YouTube and downloads it locally to the server. Requires payload format: {'query': '<song or video description>'}",
        "func": youtube_downloader_node,
    },
    "flowchart_generator": {
        "description": (
            "PREFERRED for flowcharts, architecture diagrams, and process/pipeline "
            "diagrams — renders in-process (pure Python, no external tools), so it "
            "is fast and cannot fail from a missing Node/Chromium install. "
            "Supports ONLY Mermaid 'graph'/'flowchart' syntax: TD/TB/BT/LR/RL "
            "direction, node shapes [rect] (round) {diamond} ((circle)), and edges "
            "--> -.-> ==> --- with optional |labels|. Does NOT support "
            "sequenceDiagram, classDiagram, stateDiagram, erDiagram, gantt, "
            "subgraph blocks, or classDef/style directives — if the request needs "
            "any of those, use 'mermaid_generator' instead. If this agent returns "
            "an 'Unsupported syntax' error, retry the SAME request with "
            "'mermaid_generator' rather than reformulating. "
            "Requires payload format: {\"script\": \"graph TD\\nA-->B;\"}"
        ),
        "func": mermaid_lite_node,
    },
    "mermaid_generator": {
        "description": (
            "Full Mermaid.js spec — sequence diagrams, class diagrams, state "
            "diagrams, ER diagrams, gantt charts, subgraphs, styling directives — "
            "anything 'flowchart_generator' doesn't cover. Renders via the mmdc "
            "CLI (Node + Chromium), so it is slower and depends on that toolchain "
            "being installed. Prefer 'flowchart_generator' for plain flowcharts "
            "and architecture diagrams; use this one only when the request needs "
            "a diagram TYPE flowchart_generator doesn't support. "
            "Provide the raw Mermaid.js syntax in the 'script' payload field. "
            "Example payload: {\"script\": \"graph TD\\nA-->B;\"}"
        ),
        "func": mermaid_agent_node,
    },
    "wordcloud_generator": {
        "description": (
            "Renders a word-frequency cloud (SVG, in-process, no external "
            "dependencies) from a block of text — a scraped article, aggregated "
            "search results, an uploaded document excerpt, a transcript. Good "
            "for a quick visual summary of dominant themes/keywords. Needs at "
            "least ~40-50 words of running English prose to produce a "
            "meaningful cloud — short input raises an error asking for more "
            "text; gather more via scraper/search/doc_retriever first if that "
            "happens. NOT for structured or relational data (use "
            "'flowchart_generator' for that) — this only does word frequency. "
            "Requires payload format: {\"text\": \"<text to visualize>\"}"
        ),
        "func": wordcloud_node,
    },
}
