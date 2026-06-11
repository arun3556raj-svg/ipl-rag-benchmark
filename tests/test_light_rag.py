"""Sanity tests for the LightRAG graph architecture."""

import pytest

from architectures import graph_builder, light_rag

DEMO_QUESTION = "How many times did MS Dhoni win player of the match for Chennai Super Kings?"
ENTITY_QUESTION = "Where does Mumbai Indians play their home matches?"


@pytest.fixture(scope="session", autouse=True)
def ensure_graph_built():
    if not light_rag.GRAPH_PATH.exists():
        graph_builder.materialise()


def test_graph_file_exists():
    assert light_rag.GRAPH_PATH.exists()


def test_graph_loads_with_expected_node_counts():
    light_rag.ensure_ready()
    G = light_rag._state["G"]
    players = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "player")
    teams   = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "team")
    venues  = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "venue")
    assert players > 500, f"Expected >500 players, got {players}"
    assert teams   > 10,  f"Expected >10 teams, got {teams}"
    assert venues  > 30,  f"Expected >30 venues, got {venues}"


def test_entity_extraction_finds_known_names():
    light_rag.ensure_ready()
    seeds = light_rag.extract_entities(DEMO_QUESTION)
    seed_names = [
        light_rag._state["G"].nodes[s].get("name", "")
        for s in seeds
    ]
    assert any("Dhoni" in n or "MS Dhoni" in n for n in seed_names), (
        f"Expected MS Dhoni in seeds, got: {seed_names}"
    )


def test_retrieve_returns_top_nodes_and_context():
    result = light_rag.retrieve(ENTITY_QUESTION, top_k=20)
    assert len(result["top_nodes"]) == 20
    assert result["graph_ms"] > 0
    assert isinstance(result["context"], str)
    assert len(result["context"]) > 20


def test_answer_returns_full_shape():
    out = light_rag.answer(DEMO_QUESTION, use_mock=True)
    for key in ("question", "answer", "seed_count", "seed_nodes",
                "top_node_count", "graph_ms", "latency_ms",
                "cost_usd", "llm_calls", "use_mock"):
        assert key in out, f"Missing key: {key}"
    assert out["seed_count"] > 0
    assert isinstance(out["answer"], str)
    assert len(out["answer"]) > 0
