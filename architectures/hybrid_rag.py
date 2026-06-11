"""Hybrid BM25 plus Vector RAG retrieval architecture.

Two retrieval signals over the same chunk pool:

  * **BM25**  classic sparse keyword scoring. Fast, lexical, no embeddings.
  * **Vector**  dense semantic retrieval using all-MiniLM-L6-v2 embeddings
    persisted in ChromaDB.

Top results from each retriever are merged using reciprocal rank fusion.
The merged chunks plus the question go to the LLM, which generates the
final answer.

Phase 3 uses a mock LLM so the full retrieval pipeline can be exercised
without an api key. Real DeepSeek client gets wired once available.

Initial build cost (one time): about 2 to 3 minutes on CPU for the vector
index over ~4,600 chunks. ChromaDB persists, so subsequent runs are fast.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
CHROMA_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "ipl_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_state: dict = {
    "chunks": None,          # list[dict]
    "by_id": None,           # dict[str, dict]
    "bm25": None,            # BM25Okapi instance
    "tokenized": None,       # list[list[str]]
    "embed_model": None,     # SentenceTransformer instance
    "collection": None,      # chromadb collection
}


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}. "
            "Run: uv run python -m architectures.chunk_builder"
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def _tokenize(text: str) -> list[str]:
    """Light lowercasing tokenizer for BM25. Splits on whitespace and
    strips trivial punctuation. Good enough for cricket text."""
    return [
        t.strip(".,;:!?\"'()[]")
        for t in text.lower().split()
        if t.strip(".,;:!?\"'()[]")
    ]


def _build_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized), tokenized


def _build_or_load_vector_collection(chunks: list[dict]):
    """Ensure a Chroma collection exists and contains every chunk."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    expected = len(chunks)
    current = collection.count()

    if current == expected:
        return model, collection

    # Rebuild from scratch on count mismatch. Simpler than diffing.
    if current > 0:
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    # Chroma metadata accepts only primitives. We keep only safe fields here
    # and rely on the local by_id dict for the rest.
    safe_metas = []
    for c in chunks:
        m = {"type": c["type"]}
        for k, v in c.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                m[k] = v
        safe_metas.append(m)

    # Chroma has a per-add ceiling; chunk the writes.
    BATCH = 1000
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=texts[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas=safe_metas[i:i + BATCH],
        )
    return model, collection


def ensure_ready() -> None:
    """Load chunks and build indexes if not already in memory."""
    if _state["chunks"] is not None:
        return

    chunks = _load_chunks()
    bm25, tokenized = _build_bm25(chunks)
    model, collection = _build_or_load_vector_collection(chunks)

    _state["chunks"] = chunks
    _state["by_id"] = {c["id"]: c for c in chunks}
    _state["bm25"] = bm25
    _state["tokenized"] = tokenized
    _state["embed_model"] = model
    _state["collection"] = collection


def retrieve(query: str, top_k: int = 12, per_retriever: int = 20) -> dict:
    """Run both retrievers and fuse the results.

    Returns a dict with the merged chunks plus per retriever debugging info.
    """
    ensure_ready()
    chunks = _state["chunks"]

    # BM25
    start_bm = time.perf_counter()
    scores = _state["bm25"].get_scores(_tokenize(query))
    bm25_ranked_idx = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True,
    )[:per_retriever]
    bm25_rank_ids = [chunks[i]["id"] for i in bm25_ranked_idx]
    bm_ms = (time.perf_counter() - start_bm) * 1000

    # Vector
    start_v = time.perf_counter()
    q_emb = _state["embed_model"].encode([query]).tolist()
    v_result = _state["collection"].query(
        query_embeddings=q_emb,
        n_results=per_retriever,
    )
    vec_rank_ids = v_result["ids"][0]
    v_ms = (time.perf_counter() - start_v) * 1000

    # Reciprocal rank fusion
    fused = _rrf([bm25_rank_ids, vec_rank_ids])
    merged_ids = [cid for cid, _ in fused[:top_k]]
    merged_chunks = [_state["by_id"][cid] for cid in merged_ids]

    return {
        "chunks": merged_chunks,
        "bm25_ids": bm25_rank_ids,
        "vector_ids": vec_rank_ids,
        "bm25_ms": round(bm_ms, 1),
        "vector_ms": round(v_ms, 1),
    }


def _rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Standard technique for merging rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


_ANSWER_SYSTEM = (
    "You are a cricket analyst. Using ONLY the retrieved context below, "
    "answer the question in one or two clear sentences. "
    "Cite specific numbers when they appear in the context. "
    "If the context does not contain enough information, say so briefly."
)

def format_answer(question: str, chunks: list[dict], use_mock: bool = True) -> tuple[str, float, float]:
    """Generate a natural language answer. Returns (text, latency_ms, cost_usd)."""
    if use_mock:
        return _mock_format_answer(question, chunks), 0.0, 0.0
    from architectures.llm import chat
    context = "\n\n".join(c["text"] for c in chunks[:12])
    text, ms, cost = chat(
        system=_ANSWER_SYSTEM,
        user=f"Context:\n{context}\n\nQuestion: {question}",
        temperature=0.2,
        max_tokens=300,
    )
    return text, ms, cost


def _mock_format_answer(question: str, chunks: list[dict]) -> str:
    """Mock answer: look for a player name that appears across many chunks
    and surface it. Useful baseline that exercises the retrieval pipeline
    without making things up."""
    if not chunks:
        return "No chunks retrieved."

    q = question.lower()

    # For the death overs demo question, the relevant signal is which
    # player names show up most in the retrieved set.
    name_counts: dict[str, int] = {}
    for c in chunks:
        name = (c.get("metadata") or {}).get("player_name")
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1

    top_player = max(name_counts.items(), key=lambda kv: kv[1], default=(None, 0))

    if "death over" in q and top_player[0]:
        return (
            f"Based on the retrieved chunks, {top_player[0]} appears most often "
            f"across late innings records ({top_player[1]} of {len(chunks)} "
            f"retrieved chunks). Hybrid RAG flags this as the likely leader "
            f"in death over scoring, though the precise total is not extracted."
        )

    if top_player[0]:
        return (
            f"Retrieved {len(chunks)} chunks. {top_player[0]} is the most "
            f"frequently mentioned player in this context."
        )

    sample = chunks[0]["text"]
    snippet = sample[:140] + ("..." if len(sample) > 140 else "")
    return f"Retrieved {len(chunks)} chunks. Top match: \"{snippet}\""


def answer(question: str, use_mock: bool = True, top_k: int = 12) -> dict:
    overall_start = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k)
    formatted, llm_ms, cost = format_answer(question, retrieved["chunks"], use_mock=use_mock)
    total_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "question": question,
        "answer": formatted,
        "retrieved_count": len(retrieved["chunks"]),
        "retrieved_ids": [c["id"] for c in retrieved["chunks"]],
        "bm25_ms": retrieved["bm25_ms"],
        "vector_ms": retrieved["vector_ms"],
        "latency_ms": round(total_ms, 1),
        "cost_usd": round(cost, 6),
        "llm_calls": 0 if use_mock else 1,
        "use_mock": use_mock,
    }
