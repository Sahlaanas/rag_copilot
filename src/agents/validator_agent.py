"""
Validator node — checks whether the draft answer is actually grounded in
the retrieved chunks (not hallucinated). If not, and we haven't retried
too many times already, the graph loops back to retrieval (see
agents/graph.py's conditional edge) rather than returning an ungrounded
answer.
"""
from src.agents.state import AgentState
from src.services.llm_provider import chat_completion
from src.utils.telemetry import trace_node

MAX_RETRIES = 2


@trace_node("validator_agent")
def validator_agent(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "")
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n---\n\n".join(c["text"] for c in chunks) or "(no context)"

    verdict = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker. Given a context and a draft answer, "
                    "determine if every claim in the draft is actually supported by the "
                    'context. Reply with exactly one word: "grounded" or "ungrounded".'
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nDraft answer:\n{draft}",
            },
        ],
        temperature=0.0,
    )

    is_grounded = "grounded" in verdict.strip().lower() and "ungrounded" not in verdict.strip().lower()
    retry_count = state.get("retry_count", 0)

    update: AgentState = {
        "is_grounded": is_grounded,
        "validation_notes": verdict.strip(),
        "retry_count": retry_count + (0 if is_grounded else 1),
    }

    if is_grounded or retry_count + 1 >= MAX_RETRIES:
        # Either it's good, or we've retried enough — accept what we have
        # rather than looping forever.
        update["final_answer"] = draft

    return update