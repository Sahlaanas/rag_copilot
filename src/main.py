"""
FastAPI application entrypoint.

Run locally:
    uvicorn src.main:app --reload

Flow per request (see README for the full architecture diagram):
    1. Guardrails screen the input
    2. LangGraph agent graph runs (router -> retrieval -> validator[-> retrieval...])
    3. Guardrails screen the output
    4. Response returned to the caller
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.graph import compiled_graph
from src.core.guardrails import check_input, check_output
from src.services.llm_provider import ensure_model_available


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pulls the Ollama model on startup if it isn't already present,
    # so the first real request doesn't stall on a multi-GB download.
    ensure_model_available()
    yield


app = FastAPI(
    title="Enterprise Agentic Ops & Compliance Copilot",
    description="100% free/local stack: Ollama + Chroma + LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    is_grounded: bool
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    guard_result = check_input(request.query)
    if not guard_result.allowed:
        raise HTTPException(status_code=400, detail=f"Blocked by guardrails: {guard_result.reason}")

    result_state = compiled_graph.invoke({"user_query": request.query, "retry_count": 0})

    answer = result_state.get("final_answer") or result_state.get("draft_answer", "")

    output_guard = check_output(answer)
    if not output_guard.allowed:
        raise HTTPException(
            status_code=500,
            detail=f"Response blocked by output guardrails: {output_guard.reason}",
        )

    sources = [c["id"] for c in result_state.get("retrieved_chunks", [])]

    return QueryResponse(
        answer=answer,
        route=result_state.get("route", "unknown"),
        is_grounded=result_state.get("is_grounded", False),
        sources=sources,
    )