"""
Cross-encoder reranker — local, free, no API key (replaces Cohere's
paid reranker). Cross-encoders score a (query, document) pair jointly,
which is more accurate than the bi-encoder similarity used for the
initial vector search, at the cost of being too slow to run over an
entire corpus — hence: cheap hybrid search first, then rerank only the
top-N candidates.
"""
from sentence_transformers import CrossEncoder

from src.core.config import settings

_reranker = CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    candidates: list of {"id", "text", ...} from hybrid_search().
    Returns the top_k candidates, reordered by cross-encoder relevance,
    each with a "rerank_score" field added.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = _reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]