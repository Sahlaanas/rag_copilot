r"""
Assembles the LangGraph state graph:

    START -> router -> (retrieval | incident) -> validator -> (retry same node | END)

  router picks one of two branches based on state["route"]:
    - "policy_search"       -> retrieval node (general informational retrieval)
    - "incident_resolution" -> incident node (triage-oriented prompt: immediate
                                steps, escalation SLA, explicit dos/don'ts)

  Both branches feed into the same validator node. If the validator finds
  the draft ungrounded (and retries remain), the graph loops back to
  whichever node produced it -- not always retrieval -- so an incident-path
  retry stays on the incident-tuned prompt rather than silently switching
  to the general one.
"""
from langgraph.graph import END, StateGraph

from src.agents.incident_agent import incident_agent
from src.agents.retrieval_agent import retrieval_agent
from src.agents.router_agent import router_agent
from src.agents.state import AgentState
from src.agents.validator_agent import validator_agent


def _route_after_router(state: AgentState) -> str:
    return "incident" if state.get("route") == "incident_resolution" else "retrieval"


def _route_after_validator(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    return "retry_incident" if state.get("route") == "incident_resolution" else "retry_retrieval"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("incident", incident_agent)
    graph.add_node("validator", validator_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"retrieval": "retrieval", "incident": "incident"},
    )
    graph.add_edge("retrieval", "validator")
    graph.add_edge("incident", "validator")
    graph.add_conditional_edges(
        "validator",
        _route_after_validator,
        {"retry_retrieval": "retrieval", "retry_incident": "incident", "end": END},
    )

    return graph.compile()


# Compiled once at import time -- reused across requests.
compiled_graph = build_graph()