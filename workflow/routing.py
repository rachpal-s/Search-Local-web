"""workflow/routing.py — Conditional routing logic between graph nodes.

NOTE: the original file also contained `route_from_critic_OLDDD`, an
earlier APPROVED/feedback-text based router paired with
`critic_node_OLDDD`. Neither was wired into the compiled graph, so it
was dropped here as dead code (see agents/critic.py).
"""
from langgraph.graph import END
from langgraph.types import Send

from workflow.registry import AGENT_REGISTRY
from workflow.state import AgentState

CRITIC_PASS_THRESHOLD = 85
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
    for task in tasks:
        agent_name = task.get("agent")
        payload = task.get("payload")
        if agent_name in AGENT_REGISTRY:
            print(f"[ROUTER] 🔀 Dispatching task to specialist: '{agent_name}' with payload: {payload}")
            send_actions.append(Send(agent_name, payload))
        else:
            print(f"[ROUTER] ⚠️ Requested agent '{agent_name}' is not in AGENT_REGISTRY.")

    return send_actions


def route_from_critic(state: AgentState):
    """Terminate once the critic's score clears the threshold or the loop limit is hit."""
    score = state.get("eval_score", 0)
    critic_loop_count = state.get("critic_loop_count", 0)

    print(f"\n[ROUTER] 🔀 Critic Routing evaluation. Score: {score}/100, Iteration: {critic_loop_count}")

    if score >= CRITIC_PASS_THRESHOLD:
        print(f"[ROUTER] 🛑 Score {score} meets threshold of {CRITIC_PASS_THRESHOLD}. Flow terminating.")
        return END

    if critic_loop_count >= state.get("max_critic_reviews", 3):
        print("[ROUTER] 🛑 Maximum loop iteration limit hit. Terminating graph.")
        return END

    # state["critic_loop_count"] = critic_loop_count + 1
    print(f"[ROUTER] 🔄 Score {score} is below threshold. Re-routing back to SUPERVISOR.")
    return "supervisor"
