"""workflow/routing.py — Conditional routing logic between graph nodes.

NOTE: the original file also contained `route_from_critic_OLDDD`, an
earlier APPROVED/feedback-text based router paired with
`critic_node_OLDDD`. Neither was wired into the compiled graph, so it
was dropped here as dead code (see agents/critic.py).
"""
import json

from langgraph.graph import END
from langgraph.types import Send

from config import get_settings
from workflow import inflight
from workflow.registry import AGENT_REGISTRY
from workflow.state import AgentState

# MAX_SUPERVISOR_LOOPS = 2  # loop_count > this forces a route to critic
MAX_CRITIC_LOOPS = 2      # loop_count >= this terminates the graph


def _task_key(agent_name: str, payload) -> str:
    """Stable identity for one unit of work: agent + normalised payload.

    The old cap counted dispatches per AGENT NAME, which conflated two
    completely different things. Four scraper tasks against four distinct URLs
    is fan-out — the thing Send() exists for, and the most common shape of
    query this app receives. Four mermaid_generator calls with the same script
    is repetition. Only the second is worth blocking, and only identity can
    tell them apart.
    """
    try:
        body = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        body = str(payload)
    return f"{agent_name}:{body.strip().lower()}"


def route_from_supervisor(state: AgentState):
    """Fan out pending tasks to worker agents via Send, or hand off to the critic."""
    tasks = state.get("pending_tasks", [])
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    print(f"\n[ROUTER] 🔀 Supervisor Routing evaluation. Pending tasks count: {len(tasks or [])}")

    # ── "Finish now" (cooperative stop) ────────────────────────────────
    # Set by POST /chat/stop while the user watches the trace. Checked HERE,
    # before anything else, because this is the one point where every path
    # through the graph converges: a worker always returns to the supervisor,
    # and the supervisor always consults this router before dispatching again.
    #
    # END, not "critic": the whole point of the button is to skip the critic's
    # LLM call. add_conditional_edges("supervisor", ...) in workflow/build.py
    # has no path map, so a bare END return is a valid transition and needs no
    # change there.
    #
    # supervisor_node has already seen the same flag and been told to answer
    # from existing context without scheduling work (agents/supervisor.py), so
    # state["final_response"] is normally populated by the time we get here.
    # If it somehow isn't — the flag landed mid-LLM-call, so that invocation
    # never saw the directive — main.py's fallback chain
    # (supervisor_final_response -> streamed tokens -> gathered context) still
    # produces something readable. Ending empty-handed is not a risk worth a
    # self-loop to avoid.
    if inflight.stop_requested():
        print(f"[ROUTER] ⏹️ Stop requested by the user. Dropping "
              f"{len(tasks or [])} pending task(s) and ending without the critic.")
        return END


    if loop_count > max_loops:
        print(f"[ROUTER] 🛑 Maximum loop count ({loop_count}) reached. Forcing route to CRITIC.")
        return "critic"

    if not tasks:
        print("[ROUTER] 🔀 No sub-tasks remaining. Routing to CRITIC.")
        return "critic"

    send_actions = []
    settings = get_settings()
    retrieval_attempts = state.get("retrieval_attempts", 0)
    max_retrieval_attempts = settings.max_retrieval_attempts
    dispatch_counts = state.get("dispatch_counts") or {}
    max_per_agent = settings.max_dispatches_per_agent
    # Counts dispatches decided within THIS routing call too, so a single
    # supervisor decision asking for the same agent five times is capped
    # just as a five-loop repeat would be.
    dispatched_now: dict[str, int] = {}
    # Identities decided within THIS routing call, so one decision listing the
    # same URL twice is deduped as well as one repeated across loops.
    keys_now: set[str] = set()

    for task in tasks:
        agent_name = task.get("agent")
        payload = task.get("payload")
        if agent_name not in AGENT_REGISTRY:
            print(f"[ROUTER] ⚠️ Requested agent '{agent_name}' is not in AGENT_REGISTRY.")
            inflight.note_dropped(
                str(agent_name), str(payload)[:200],
                "no such agent exists — check the available agent list and use "
                "one of those names, or answer without it")
            continue

        if agent_name == "doc_retriever" and state.get("artifact_mode"):
            print("[ROUTER] 🛑 Skipping doc_retriever — attached files are "
                    "already in context whole. Nothing to retrieve.")
            continue

        if agent_name == "doc_retriever" and retrieval_attempts >= max_retrieval_attempts:
            # Tighter than the general cap below, and kept separate — see
            # config.py. A corpus that didn't have the answer on attempt 1
            # won't have it on attempt 3; past this, re-querying the same
            # documents is wasted latency, not diligence.
            print(f"[ROUTER] 🛑 Skipping doc_retriever dispatch — already made "
                  f"{retrieval_attempts} attempt(s), cap is {max_retrieval_attempts}.")
            inflight.note_dropped(
                "doc_retriever", str(payload)[:200],
                f"retrieval cap of {max_retrieval_attempts} reached — the "
                f"attached documents have already been searched; they do not "
                f"contain more on this")
            continue

        # ── 1. Identity dedupe (replaces the old per-name count as the
        #       primary guard). This is what actually stops the mermaid case:
        #       the same script twice is refused however few dispatches have
        #       happened, while four DIFFERENT URLs all get through.
        key = _task_key(agent_name, payload)
        if inflight.was_dispatched(key) or key in keys_now:
            print(f"[ROUTER] 🛑 Skipping duplicate '{agent_name}' task — this "
                  f"exact payload has already been dispatched this turn.")
            inflight.note_dropped(
                agent_name, str(payload)[:200],
                "already dispatched this turn with an identical payload — the "
                "result is already in context; use it rather than re-requesting it")
            continue

        # ── 2. Per-agent ceiling, now a backstop rather than the main guard.
        #       It was 2, which meant a query naming four URLs could never have
        #       more than two of them scraped: the router dropped the other two
        #       silently, the supervisor re-requested them, context stopped
        #       growing, and the stagnation interceptor dumped raw JSON. The
        #       cap was doing the damage, not the model.
        already = dispatch_counts.get(agent_name, 0) + dispatched_now.get(agent_name, 0)
        if already >= max_per_agent:
            print(f"[ROUTER] 🛑 Skipping '{agent_name}' dispatch — already "
                  f"dispatched {already}x this turn, cap is {max_per_agent}.")
            inflight.note_dropped(
                agent_name, str(payload)[:200],
                f"per-agent ceiling of {max_per_agent} dispatches reached for "
                f"'{agent_name}' this turn — answer from what you already have "
                f"instead of requesting more")
            continue

        print(f"[ROUTER] 🔀 Dispatching task to specialist: '{agent_name}' with payload: {payload}")
        dispatched_now[agent_name] = dispatched_now.get(agent_name, 0) + 1
        keys_now.add(key)
        inflight.note_dispatch(key)
        send_actions.append(Send(agent_name, payload))

    if not send_actions:
        # Every requested task was filtered out (capped agents, or unknown
        # ones). An empty Send list isn't a valid transition — route to
        # critic with whatever context already exists rather than leave the
        # graph with nowhere to go.
        print("[ROUTER] 🔀 All requested tasks were capped; nothing left to "
              "dispatch. Routing to CRITIC.")
        return "critic"

    return send_actions


def route_from_critic(state: AgentState):
    """Terminate once the critic's score clears the threshold, the loop
    limit is hit, or the score has plateaued (not worth another full
    regenerate+evaluate cycle to chase)."""
    score = state.get("eval_score", 0)
    critic_loop_count = state.get("critic_loop_count", 0)
    settings = get_settings()
    pass_threshold = settings.critic_pass_threshold

    print(f"\n[ROUTER] 🔀 Critic Routing evaluation. Score: {score}/100, Iteration: {critic_loop_count}")

    # Same flag, second gate. Reached when the stop lands while the critic is
    # already mid-evaluation: the supervisor router had nothing to catch. A
    # low score here would normally loop back for a full regenerate cycle,
    # which is exactly the wait the user just asked to end.
    if inflight.stop_requested():
        print("[ROUTER] ⏹️ Stop requested by the user. Accepting the current "
              "answer without another supervisor loop.")
        return END


    if score >= pass_threshold:
        print(f"[ROUTER] 🛑 Score {score} meets threshold of {pass_threshold}. Flow terminating.")
        return END

    if critic_loop_count >= state.get("max_critic_reviews", 3):
        print("[ROUTER] 🛑 Maximum loop iteration limit hit. Terminating graph (low confidence).")
        return END

    history = state.get("score_history") or []
    if len(history) >= 2:
        improvement = history[-1] - history[-2]
        min_improvement = settings.critic_min_improvement
        if improvement < min_improvement:
            print(f"[ROUTER] 🛑 Score improved only {improvement} point(s) over the last "
                  f"attempt (need {min_improvement}+) — plateaued. Terminating graph "
                  f"(low confidence) rather than spending another loop chasing it.")
            return END

    print(f"[ROUTER] 🔄 Score {score} is below threshold. Re-routing back to SUPERVISOR.")
    return "supervisor"
