"""
Retrieval node — runs hybrid search, reranks the results, and drafts an
answer grounded in the retrieved chunks. The validator node (see
validator_agent.py) checks this draft before it's returned to the user.
"""
from src.agents.state import AgentState
from src.services.llm_provider import chat_completion
from src.services.reranker import rerank
from src.services.vector_store import hybrid_search
from src.utils.telemetry import trace_node


@trace_node("retrieval_agent")
def retrieval_agent(state: AgentState) -> AgentState:
    query = state["user_query"]

    candidates = hybrid_search(query, top_k=10)
    top_chunks = rerank(query, candidates, top_k=5)

    context = "\n\n---\n\n".join(c["text"] for c in top_chunks) or "(no relevant documents found)"

    draft = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the context below. "
                    "If the context doesn't contain the answer, say you don't have "
                    "enough information — never make something up.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.2,
    )

    return {"retrieved_chunks": top_chunks, "draft_answer": draft}