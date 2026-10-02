"""
Lightweight per-node latency logging for the agent graph.

This used to also support Langfuse tracing (self-hosted, free) for full
trace spans — dropped in favor of keeping the project dependency-light
and infra-free. If you want structured tracing back later, the natural
place to add it is inside the `try` block in `wrapper()` below: wrap
`result = func(*args, **kwargs)` with whatever tracing client you pick,
without touching any of the agent node files that use this decorator.
"""
import functools
import time


def trace_node(name: str):
    """
    Decorator for LangGraph node functions. Logs latency to stdout.

    Usage:
        @trace_node("retrieval_agent")
        def retrieval_agent(state: AgentState) -> AgentState:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = func(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000
            print(f"[telemetry] {name} completed in {elapsed_ms:.1f}ms")
            return result

        return wrapper
    return decorator