"""workflow/build.py — Assembles and compiles the LangGraph workflow.

Node wiring is data-driven off AGENT_REGISTRY, so new worker agents
never require touching this file — see workflow/registry.py.
"""
from langgraph.graph import END, START, StateGraph

from agents.critic import critic_node
from agents.supervisor import supervisor_node
from workflow.registry import AGENT_REGISTRY
from workflow.routing import route_from_critic, route_from_supervisor
from workflow.state import AgentState


def build_graph():
    """Assemble and compile the supervisor/critic/worker StateGraph."""
    # (named graph_builder, not `workflow`, to avoid shadowing this
    # package's own name for anyone reading top-to-bottom)
    graph_builder = StateGraph(AgentState)

    # Supervisor & Critic
    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("critic", critic_node)

    # Worker nodes, sourced from the dynamic registry
    for name, meta in AGENT_REGISTRY.items():
        graph_builder.add_node(name, meta["func"])
        graph_builder.add_edge(name, "supervisor")

    # Graph connections
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges("supervisor", route_from_supervisor)
    graph_builder.add_conditional_edges("critic", route_from_critic, {"supervisor": "supervisor", END: END})

    return graph_builder.compile()


app_graph = build_graph()
