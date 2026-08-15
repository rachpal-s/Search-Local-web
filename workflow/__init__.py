"""workflow — graph state, routing, registry and the compiled LangGraph.

Import order below is deliberate. `build` pulls in agents.critic and
agents.supervisor, and both of those do `from workflow import inflight` while
this module is still executing. Binding `inflight` and `state` first means the
attribute already exists on the partially-initialised package by the time that
happens, instead of relying on the submodule-import fallback to paper over it.
"""
from workflow import inflight  # noqa: F401  (must precede `build`)
from workflow.state import AgentState  # noqa: F401
from workflow.build import app_graph, build_graph  # noqa: F401

__all__ = ["app_graph", "build_graph", "inflight", "AgentState"]
