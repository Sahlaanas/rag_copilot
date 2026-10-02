# Architecture

```mermaid
flowchart TD
    A[User Request] --> B[Input Guardrails<br/>heuristic + LLM self-check]
    B -- blocked --> Z[400 — rejected]
    B -- allowed --> C[Router Agent<br/>LangGraph node]
    C -->|policy_search| D[Retrieval Agent<br/>general informational prompt]
    C -->|incident_resolution| K[Incident Agent<br/>triage-oriented prompt]
    D --> E1[Hybrid Search<br/>BM25 + Chroma vector, RRF merge]
    K --> E2[Hybrid Search<br/>BM25 + Chroma vector, RRF merge]
    E1 --> F1[Cross-Encoder Reranker<br/>top-5]
    E2 --> F2[Cross-Encoder Reranker<br/>top-5]
    F1 --> H[Validator Agent<br/>groundedness check]
    F2 --> H
    H -- ungrounded, retries left --> D
    H -- ungrounded, retries left --> K
    H -- grounded / retries exhausted --> I[Output Guardrails]
    I -- blocked --> Z
    I -- allowed --> J[Response to User]

    style B fill:#2d2d2d,color:#fff
    style I fill:#2d2d2d,color:#fff
    style H fill:#1a3a5c,color:#fff
```

Every node above is wrapped in `@trace_node(...)` (`src/utils/telemetry.py`),
which logs each node's latency to stdout — lightweight and dependency-free,
no separate observability stack required.

A retry loops back to whichever node produced the draft (`retrieval` stays
on `retrieval`, `incident` stays on `incident`), not always the same node —
see `agents/graph.py` for the conditional edge logic.

## Demo

- [ ] Add a Loom/screen-recording link here once recorded (see checklist in README)
