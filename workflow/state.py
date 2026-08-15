"""workflow/state.py — Shared state schemas for the LangGraph workflow.

`AgentState` is the top-level graph state threaded through the
supervisor/critic loop. Each worker agent gets its own narrow
TypedDict describing the payload it expects via `Send(...)`.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Top-level graph state shared across supervisor/critic/worker nodes."""
    user_query: str
    context: Annotated[List[str], operator.add]
    # action_logs needs the same reducer as context: worker nodes run in
    # parallel under Send(), so without one the last writer silently wins and
    # the execution trace loses every branch but one.
    action_logs: Annotated[List[str], operator.add]
    pending_tasks: List[Dict[str, Any]]
    final_response: Optional[str]
    feedback: Optional[str]
    eval_score: int
    loop_count: int
    max_loops: int
    critic_loop_count: int
    max_critic_reviews: int
    # State Stagnation Trackers
    last_unique_context_count: int
    stagnation_streak: int
    # --- conversation scope (set by the /chat/stream endpoint) ---
    conversation_id: Optional[str]
    attached_files: List[str]
    chat_history: List[Dict[str, str]]


class ExtractTaskState(TypedDict):
    """Payload state for the Extractor worker."""
    text: str


class SearchTaskState(TypedDict):
    """Payload state for the Search worker."""
    query: str


class ScrapeTaskState(TypedDict):
    """Payload state for the Web Scraper worker."""
    url: str


class YoutubeTaskState(TypedDict):
    """Payload state for the YouTube Downloader worker."""
    query: str

class MermaidTaskState(TypedDict):
    """Payload state for the Mermaid Diagram Generator worker."""
    script: str


class WordCloudTaskState(TypedDict):
    """Payload state for the Word Cloud Generator worker."""
    text: str


class DocRetrieveTaskState(TypedDict):
    """Payload state for the Document Retriever worker."""
    query: str
