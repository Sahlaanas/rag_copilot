"""
Shared state passed between every node in the LangGraph graph. Each
agent node reads what it needs from this dict and returns a partial
update — LangGraph merges updates into the running state automatically.
"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────
    user_query: str

    # ── Guardrails ───────────────────────────────────────────────────
    input_blocked: bool
    block_reason: str | None

    # ── Router ───────────────────────────────────────────────────────
    # "policy_search" | "incident_resolution" — extend as you add branches
    route: str

    # ── Retrieval ────────────────────────────────────────────────────
    retrieved_chunks: list[dict]  # [{"id", "text", "score", "rerank_score"}, ...]

    # ── Generation ───────────────────────────────────────────────────
    draft_answer: str

    # ── Validation ───────────────────────────────────────────────────
    is_grounded: bool
    validation_notes: str | None
    retry_count: int

    # ── Output ───────────────────────────────────────────────────────
    final_answer: str