"""
Incident-resolution node. Uses the same retrieval machinery as
retrieval_agent (hybrid search + rerank, for grounding), but drafts the
answer with a triage-oriented system prompt instead of a general
informational one: immediate actionable steps, who/where to escalate to
and within what SLA, and explicit dos/don'ts — appropriate for someone
describing an active, ongoing problem rather than asking about a policy.

This is what makes router_agent's classification functionally meaningful
rather than just a recorded field: policy_search and incident_resolution
now take genuinely different paths through the graph (see agents/graph.py).
"""
from src.agents.state import AgentState
from src.services.llm_provider import chat_completion
from src.services.reranker import rerank
from src.services.vector_store import hybrid_search
from src.utils.telemetry import trace_node


@trace_node("incident_agent")
def incident_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    candidates = hybrid_search(query, top_k=10)
    top_chunks = rerank(query, candidates, top_k=5)
    context = "\n\n---\n\n".join(c["text"] for c in top_chunks) or "(no relevant documents found)"

    draft = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an incident-response triage assistant. The user is describing "
                    "or asking about an active or suspected operational/security incident. "
                    "Using ONLY the context below, give immediate, actionable guidance:\n"
                    "  1. What to do right now\n"
                    "  2. Who/where to report it, and within what time window\n"
                    "  3. Anything the user should explicitly NOT do\n\n"
                    "Be direct and concise — this may be urgent. If the context doesn't cover "
                    "the specific situation described, say so plainly and point the user to "
                    "the general incident-reporting channel/SLA from the context rather than "
                    "guessing at remediation steps you have no basis for.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,  # slightly lower than retrieval_agent's — prefer precise/safe over creative here
    )

    return {"retrieved_chunks": top_chunks, "draft_answer": draft}