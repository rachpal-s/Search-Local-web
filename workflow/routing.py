"""workflow/routing.py — Conditional routing logic between graph nodes.

NOTE: the original file also contained `route_from_critic_OLDDD`, an
earlier APPROVED/feedback-text based router paired with
`critic_node_OLDDD`. Neither was wired into the compiled graph, so it
was dropped here as dead code (see agents/critic.py).
"""
from langgraph.graph import END
from langgraph.types import Send

from config import get_settings
from workflow.registry import AGENT_REGISTRY
from workflow.state import AgentState

# MAX_SUPERVISOR_LOOPS = 2  # loop_count > this forces a route to critic
MAX_CRITIC_LOOPS = 2      # loop_count >= this terminates the graph


def route_from_supervisor(state: AgentState):
    """Fan out pending tasks to worker agents via Send, or hand off to the critic."""
    tasks = state.get("pending_tasks", [])
    loop_count = state.get("loop_count", 0)
    max_loops = state.get("max_loops", 3)
    print(f"\n[ROUTER] 🔀 Supervisor Routing evaluation. Pending tasks count: {len(tasks or [])}")

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

    for task in tasks:
        agent_name = task.get("agent")
        payload = task.get("payload")
        if agent_name not in AGENT_REGISTRY:
            print(f"[ROUTER] ⚠️ Requested agent '{agent_name}' is not in AGENT_REGISTRY.")
            continue

        if agent_name == "doc_retriever" and retrieval_attempts >= max_retrieval_attempts:
            # Tighter than the general cap below, and kept separate — see
            # config.py. A corpus that didn't have the answer on attempt 1
            # won't have it on attempt 3; past this, re-querying the same
            # documents is wasted latency, not diligence.
            print(f"[ROUTER] 🛑 Skipping doc_retriever dispatch — already made "
                  f"{retrieval_attempts} attempt(s), cap is {max_retrieval_attempts}.")
            continue

        already = dispatch_counts.get(agent_name, 0) + dispatched_now.get(agent_name, 0)
        if already >= max_per_agent:
            # General per-agent ceiling. Observed failure this prevents: the
            # supervisor re-dispatching mermaid_generator on four consecutive
            # loops, producing four diagrams from one request and presenting
            # all of them as the answer, when the first had already succeeded.
            print(f"[ROUTER] 🛑 Skipping '{agent_name}' dispatch — already "
                  f"dispatched {already}x this turn, cap is {max_per_agent}.")
            continue

        print(f"[ROUTER] 🔀 Dispatching task to specialist: '{agent_name}' with payload: {payload}")
        dispatched_now[agent_name] = dispatched_now.get(agent_name, 0) + 1
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
