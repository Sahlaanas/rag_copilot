"""
Router node — classifies the user's query into a workflow branch.
Currently a 2-way split (policy_search vs incident_resolution); add more
branches by extending the prompt's allowed labels and adding a
corresponding edge in agents/graph.py.
"""
from src.agents.state import AgentState
from src.services.llm_provider import chat_completion
from src.utils.telemetry import trace_node

_VALID_ROUTES = {"policy_search", "incident_resolution"}


@trace_node("router_agent")
def router_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    response = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's message into exactly one category.\n\n"
                    '"policy_search": the user is asking ABOUT a policy or procedure — '
                    'including questions like "how do I report an incident", "how quickly '
                    'must I report an incident", or "where do I submit X". They want to know '
                    "the RULE, not help with something happening right now.\n\n"
                    '"incident_resolution": the user is describing or reporting an ACTUAL, '
                    'CURRENT problem happening right now (e.g. "the database is down", '
                    '"I think we\'ve been breached", "our site is returning errors") and '
                    "needs help responding to it.\n\n"
                    "If in doubt whether something is a live problem or a question about the "
                    "rules, prefer policy_search.\n\n"
                    "Reply with only the category name, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )

    route = response.strip().lower().replace(" ", "_")
    if route not in _VALID_ROUTES:
        route = "policy_search"  # safe default — the broader/more general branch

    return {"route": route}