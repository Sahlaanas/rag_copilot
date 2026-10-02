"""
Vector store setup — Chroma running embedded (no separate server process,
no Docker required), embeddings via a local sentence-transformers model.
Also implements hybrid search: BM25 (keyword) + vector (semantic),
merged by reciprocal rank fusion, so exact-term queries ("policy 4.2")
work as well as semantic ones ("what's the rule about remote work").
"""
import uuid

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.core.config import settings

_embedder = SentenceTransformer(settings.embedding_model)
_chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_collection = _chroma_client.get_or_create_collection(name=settings.chroma_collection)

# BM25 index is rebuilt in memory from whatever's in the collection —
# fine at the small/medium document-count scale this project targets.
_bm25_index: BM25Okapi | None = None
_bm25_doc_ids: list[str] = []
_bm25_doc_texts: list[str] = []


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple sliding-window chunker. Good enough for policy-doc-style text;
    swap for a structure-aware chunker (headers, sections) for real docs."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_document(text: str, source: str) -> int:
    """Chunks and embeds a document into Chroma. Returns number of chunks added."""
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    ids = [f"{source}-{uuid.uuid4().hex[:8]}" for _ in chunks]
    embeddings = _embedder.encode(chunks).tolist()

    _collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": source} for _ in chunks],
    )
    _rebuild_bm25_index()
    return len(chunks)


def _rebuild_bm25_index() -> None:
    global _bm25_index, _bm25_doc_ids, _bm25_doc_texts
    all_docs = _collection.get()
    _bm25_doc_ids = all_docs["ids"]
    _bm25_doc_texts = all_docs["documents"]
    tokenized = [doc.lower().split() for doc in _bm25_doc_texts]
    _bm25_index = BM25Okapi(tokenized) if tokenized else None


def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns up to top_k {"id", "text", "score"} results, merging BM25
    keyword search and vector semantic search via reciprocal rank fusion
    (RRF) — simpler and more robust than trying to normalize/weight raw
    scores from two very different scoring systems.
    """
    if _bm25_index is None:
        _rebuild_bm25_index()

    rrf_scores: dict[str, float] = {}
    k = 60  # standard RRF constant

    # ── Vector search ────────────────────────────────────────────────
    query_embedding = _embedder.encode([query]).tolist()
    vector_results = _collection.query(query_embeddings=query_embedding, n_results=min(top_k * 2, 50))
    for rank, doc_id in enumerate(vector_results["ids"][0]):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    # ── BM25 keyword search ─────────────────────────────────────────
    if _bm25_index is not None and _bm25_doc_ids:
        bm25_scores = _bm25_index.get_scores(query.lower().split())
        ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        for rank, idx in enumerate(ranked[: top_k * 2]):
            doc_id = _bm25_doc_ids[idx]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    if not rrf_scores:
        return []

    top_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
    fetched = _collection.get(ids=top_ids)
    id_to_text = dict(zip(fetched["ids"], fetched["documents"]))

    return [
        {"id": doc_id, "text": id_to_text.get(doc_id, ""), "score": rrf_scores[doc_id]}
        for doc_id in top_ids
        if doc_id in id_to_text
    ]