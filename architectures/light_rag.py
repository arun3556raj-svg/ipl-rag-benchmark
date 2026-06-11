"""LightRAG: graph-based retrieval architecture.

Retrieval flow:
  1. Entity extraction  scan the question for known player/team/venue names.
  2. Seed nodes         each matched entity becomes a seed in the graph.
  3. Personalised PageRank (PPR)  propagate relevance from seeds through edges.
  4. Top-K subgraph     take the highest-scored nodes and their connecting edges.
  5. Serialise to text  convert the subgraph into a readable context paragraph.
  6. LLM answer         mock (or real) answer from the context.

Why PPR over simple neighbour expansion:
  Neighbour expansion explodes on well-connected nodes (MS Dhoni touches
  thousands of edges). PPR damps the walk by a teleport factor, so distant
  nodes receive exponentially less weight -- the context stays focused.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "graph.pkl"

_state: dict = {
    "G": None,           # nx.DiGraph
    "name_index": None,  # dict[str, list[str]] — lowercase name -> node ids
}


def _load_graph() -> nx.DiGraph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"graph.pkl not found at {GRAPH_PATH}. "
            "Run: uv run python -m architectures.graph_builder"
        )
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def _build_name_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """Map lowercase name fragments to node IDs for fuzzy entity matching."""
    idx: dict[str, list[str]] = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("name", "")
        if not name:
            continue
        key = name.lower()
        idx.setdefault(key, []).append(node_id)
        # also index by last word (surname matching)
        parts = key.split()
        if len(parts) > 1:
            idx.setdefault(parts[-1], []).append(node_id)
    return idx


def ensure_ready() -> None:
    if _state["G"] is not None:
        return
    G = _load_graph()
    _state["G"] = G
    _state["name_index"] = _build_name_index(G)


def extract_entities(question: str) -> list[str]:
    """Return node IDs whose name appears in the question (case-insensitive)."""
    ensure_ready()
    q = question.lower()
    idx = _state["name_index"]
    matched: list[str] = []
    seen: set[str] = set()
    for key, node_ids in idx.items():
        if key in q:
            for nid in node_ids:
                if nid not in seen:
                    matched.append(nid)
                    seen.add(nid)
    return matched


def retrieve(question: str, top_k: int = 30, alpha: float = 0.15) -> dict:
    """Run PPR from seed entities; return top-K nodes + subgraph context."""
    ensure_ready()
    G: nx.DiGraph = _state["G"]

    t0 = time.perf_counter()

    seeds = extract_entities(question)

    if seeds:
        # Personalised PageRank: high weight on seed nodes
        personalization = {nid: 1.0 / len(seeds) for nid in seeds}
        ppr = nx.pagerank(G, alpha=alpha, personalization=personalization,
                          max_iter=200, tol=1e-6)
    else:
        # No entities found — fall back to global PageRank
        ppr = nx.pagerank(G, alpha=alpha, max_iter=200, tol=1e-6)

    top_nodes = sorted(ppr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    top_ids = [nid for nid, _ in top_nodes]

    graph_ms = (time.perf_counter() - t0) * 1000

    # Build a readable subgraph context
    context = _subgraph_to_text(G, top_ids, seeds)

    return {
        "seeds": seeds,
        "top_nodes": top_ids,
        "context": context,
        "graph_ms": round(graph_ms, 1),
    }


def _subgraph_to_text(G: nx.DiGraph, node_ids: list[str],
                      seeds: list[str]) -> str:
    """Convert the top subgraph nodes + their edges into a readable paragraph."""
    node_set = set(node_ids)
    lines: list[str] = []

    # Describe seed nodes first
    for nid in seeds:
        if nid in G.nodes:
            lines.append(_describe_node(G, nid))

    # Describe edges between top nodes (avoid duplicate pairs)
    seen_pairs: set[tuple[str, str]] = set()
    for src in node_ids:
        for dst in node_ids:
            if src == dst or (src, dst) in seen_pairs:
                continue
            if G.has_edge(src, dst):
                edata = G[src][dst]
                desc = _describe_edge(G, src, dst, edata)
                if desc:
                    lines.append(desc)
                seen_pairs.add((src, dst))

    # Add top non-seed nodes that haven't been described via edges
    described = set(seeds)
    for nid in node_ids:
        if nid not in described and G.nodes[nid].get("kind") != "team":
            lines.append(_describe_node(G, nid))
            described.add(nid)
        if len(lines) > 40:
            break

    return " ".join(lines) if lines else "No relevant graph context found."


def _describe_node(G: nx.DiGraph, nid: str) -> str:
    d = G.nodes[nid]
    kind = d.get("kind")
    name = d.get("name", nid)
    if kind == "player":
        role = d.get("role", "")
        return f"{name} is a {role} player." if role else f"{name} is an IPL player."
    if kind == "team":
        return f"{name} is an IPL team."
    if kind == "venue":
        city = d.get("city", "")
        return f"{name} is a venue{' in ' + city if city else ''}."
    return f"{name}."


def _describe_edge(G: nx.DiGraph, src: str, dst: str, edata: dict) -> str:
    rel = edata.get("rel", "")
    src_name = G.nodes[src].get("name", src)
    dst_name = G.nodes[dst].get("name", dst)
    if rel == "played_for":
        seasons = edata.get("seasons", [])
        s = ", ".join(str(x) for x in sorted(set(seasons))[:5])
        return f"{src_name} played for {dst_name} in {s}."
    if rel == "won_against":
        margin = edata.get("margin")
        result = edata.get("result", "")
        m = f" by {margin} {result}" if margin else ""
        return f"{src_name} beat {dst_name}{m}."
    if rel == "pom":
        count = edata.get("count", 1)
        return f"{src_name} was player of the match for {dst_name} {count} time(s)."
    if rel == "played_in":
        season = edata.get("season", "")
        return f"{src_name} played at {dst_name}" + (f" in {season}." if season else ".")
    return ""


_ANSWER_SYSTEM = (
    "You are a cricket analyst with access to a knowledge graph of IPL players, "
    "teams, and venues. Using ONLY the graph context provided, answer the question "
    "in one or two sentences. Be specific. If the context is insufficient, say so briefly."
)

def format_answer(question: str, context: str,
                  seeds: list[str], use_mock: bool = True) -> tuple[str, float, float]:
    """Returns (text, latency_ms, cost_usd)."""
    if use_mock:
        return _mock_format_answer(question, context, seeds), 0.0, 0.0
    from architectures.llm import chat
    text, ms, cost = chat(
        system=_ANSWER_SYSTEM,
        user=f"Graph context:\n{context}\n\nQuestion: {question}",
        temperature=0.2,
        max_tokens=300,
    )
    return text, ms, cost


def _mock_format_answer(question: str, context: str,
                        seeds: list[str]) -> str:
    G: nx.DiGraph = _state["G"]
    if not seeds:
        return (
            f"No named entities found in the question. "
            f"Graph context snippet: {context[:200]}..."
        )

    seed_names = [G.nodes[s].get("name", s) for s in seeds if s in G.nodes]
    names_str = ", ".join(seed_names[:5])
    return (
        f"Graph retrieval identified {len(seeds)} entity node(s): {names_str}. "
        f"Personalised PageRank surfaced a {len(context.split())}-word context "
        f"subgraph. Context excerpt: {context[:300]}..."
    )


def answer(question: str, use_mock: bool = True, top_k: int = 30) -> dict:
    overall_start = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k)
    formatted, llm_ms, cost = format_answer(
        question, retrieved["context"], retrieved["seeds"], use_mock=use_mock
    )
    total_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "question": question,
        "answer": formatted,
        "seed_count": len(retrieved["seeds"]),
        "seed_nodes": retrieved["seeds"],
        "top_node_count": len(retrieved["top_nodes"]),
        "graph_ms": retrieved["graph_ms"],
        "latency_ms": round(total_ms, 1),
        "cost_usd": round(cost, 6),
        "llm_calls": 0 if use_mock else 1,
        "use_mock": use_mock,
    }
