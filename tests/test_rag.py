"""
Tests for the RAG pipeline. Chunking is pure logic (fast, no model
needed); ingestion/search tests are marked integration since they load
real embedding models on first run.
"""
import pytest

from src.services.vector_store import _chunk_text


def test_chunk_text_respects_chunk_size():
    text = "word " * 500  # ~2500 chars
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) > 1


def test_chunk_text_empty_input():
    assert _chunk_text("") == []


def test_chunk_text_short_input_single_chunk():
    text = "Short policy text."
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    assert chunks == [text]


@pytest.mark.integration
def test_ingest_and_hybrid_search_roundtrip():
    from src.services.vector_store import hybrid_search, ingest_document

    n_chunks = ingest_document(
        "All employees must complete security training annually. "
        "Remote work requires VPN access at all times.",
        source="test-doc",
    )
    assert n_chunks >= 1

    results = hybrid_search("security training requirement", top_k=3)
    assert len(results) >= 1
    assert any("security training" in r["text"].lower() for r in results)