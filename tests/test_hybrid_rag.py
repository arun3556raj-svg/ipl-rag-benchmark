"""Sanity tests for the Hybrid RAG architecture."""

import pytest

from architectures import chunk_builder, hybrid_rag

DEMO_QUESTION = "Who scored the most runs in death overs across all IPL seasons?"


@pytest.fixture(scope="session", autouse=True)
def ensure_chunks_built():
    """Build chunks once for the whole test session if missing."""
    if not hybrid_rag.CHUNKS_PATH.exists():
        chunk_builder.materialise()


def test_chunks_file_exists_and_has_content():
    assert hybrid_rag.CHUNKS_PATH.exists()
    chunks = hybrid_rag._load_chunks()
    assert len(chunks) > 4000, f"Expected >4000 chunks, got {len(chunks)}"

    types = {c["type"] for c in chunks}
    assert "match" in types
    assert "player_season" in types


def test_match_chunk_text_is_readable():
    chunks = hybrid_rag._load_chunks()
    match_chunks = [c for c in chunks if c["type"] == "match"]
    assert len(match_chunks) > 1000
    sample = match_chunks[0]["text"]
    assert "IPL season" in sample
    assert " played " in sample


def test_player_season_chunk_text_is_readable():
    chunks = hybrid_rag._load_chunks()
    ps_chunks = [c for c in chunks if c["type"] == "player_season"]
    assert len(ps_chunks) > 3000
    sample = ps_chunks[0]["text"]
    assert "IPL season" in sample


def test_indexes_build_and_retrieval_returns_chunks():
    hybrid_rag.ensure_ready()
    result = hybrid_rag.retrieve(DEMO_QUESTION, top_k=8)
    assert len(result["chunks"]) == 8
    assert result["bm25_ms"] > 0
    assert result["vector_ms"] > 0


def test_answer_returns_full_shape():
    out = hybrid_rag.answer(DEMO_QUESTION, use_mock=True)
    for key in ("question", "answer", "retrieved_count", "retrieved_ids",
                "bm25_ms", "vector_ms", "latency_ms", "cost_usd",
                "llm_calls", "use_mock"):
        assert key in out, f"Missing key: {key}"
    assert out["retrieved_count"] > 0
    assert isinstance(out["answer"], str)
    assert len(out["answer"]) > 0
