"""workflow/state.py — Shared state schemas for the LangGraph workflow.

`AgentState` is the top-level graph state threaded through the
supervisor/critic loop. Each worker agent gets its own narrow
TypedDict describing the payload it expects via `Send(...)`.
"""
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def merge_dispatch_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """Sum per-agent dispatch counts across parallel Send() branches.

    operator.add on dicts is a TypeError, and dict-union (|) would let the
    last writer win — losing counts from every parallel branch but one,
    which is precisely what the cap needs to see.
    """
    merged = dict(left or {})
    for agent, count in (right or {}).items():
        merged[agent] = merged.get(agent, 0) + count
    return merged


class AgentState(TypedDict):
    """Top-level graph state shared across supervisor/critic/worker nodes."""
    user_query: str
    context: Annotated[List[str], operator.add]
    # action_logs needs the same reducer as context: worker nodes run in
    # parallel under Send(), so without one the last writer silently wins and
    # the execution trace loses every branch but one.
    action_logs: Annotated[List[str], operator.add]
    # Same reducer, same reason as action_logs — doc_retriever can run more
    # than once per turn. Deliberately NOT persisted anywhere (see main.py's
    # add_message call): this is a live, on-demand visual for the turn as it
    # happens, not part of the saved conversation record. If a person wants
    # graph transparency for a specific answer, that answer needs to exist in
    # the live stream, not be reconstructed after the fact.
    graph_traces: Annotated[List[Dict[str, Any]], operator.add]
    pending_tasks: List[Dict[str, Any]]
    final_response: Optional[str]
    feedback: Optional[str]
    eval_score: int
    # Every critic score across the whole run, in order. Lets the router
    # detect a plateau (attempt N barely better than attempt N-1) and stop
    # instead of burning another full regenerate+evaluate cycle chasing an
    # improvement that isn't coming.
    score_history: Annotated[List[int], operator.add]
    # Every (score, response) pair the critic has evaluated this run, in
    # order. The graph currently ends with whatever the LAST supervisor loop
    # produced — but a later loop is not necessarily a better one. A critic
    # rejection triggers regeneration, and that regeneration can legitimately
    # come back worse or truncated; taking the last attempt then discards a
    # better earlier answer the user may already have seen streamed.
    # main.py selects the highest-scoring attempt from this list instead.
    response_attempts: Annotated[List[dict], operator.add]
    loop_count: int
    max_loops: int
    critic_loop_count: int
    max_critic_reviews: int
    # Same reducer pattern as action_logs — doc_retriever can be dispatched
    # more than once per turn (possibly in parallel via Send()), each
    # contributing +1 so the count is correct regardless of dispatch shape.
    retrieval_attempts: Annotated[int, operator.add]
    # agent name -> times dispatched this turn. Merged with a custom reducer
    # (workflow/state.py::merge_dispatch_counts) rather than operator.add,
    # since dicts need summing per key, not concatenation.
    dispatch_counts: Annotated[Dict[str, int], merge_dispatch_counts]
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
