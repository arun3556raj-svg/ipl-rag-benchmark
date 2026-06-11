"""Tests for the query routing classifier."""

import pytest
from architectures import query_classifier

FACTUAL_Q    = "Who scored the most runs in IPL history?"
NARRATIVE_Q  = "Describe Virat Kohli's performance in the 2016 season."
RELATIONAL_Q = "Which teams has MS Dhoni played for?"


@pytest.fixture(scope="session", autouse=True)
def warm_up():
    query_classifier.ensure_ready()


def test_model_loads():
    assert query_classifier._state["clf"] is not None
    assert query_classifier._state["le"] is not None


def test_route_returns_valid_arch():
    arch, conf = query_classifier.route(FACTUAL_Q)
    assert arch in ("text_to_sql", "hybrid_rag", "light_rag")
    assert 0.0 < conf <= 1.0


def test_factual_routes_to_text_to_sql():
    arch, _ = query_classifier.route(FACTUAL_Q)
    assert arch == "text_to_sql", f"Expected text_to_sql, got {arch}"


def test_relational_routes_to_light_rag():
    arch, _ = query_classifier.route(RELATIONAL_Q)
    assert arch == "light_rag", f"Expected light_rag, got {arch}"


def test_route_with_scores_shape():
    result = query_classifier.route_with_scores(NARRATIVE_Q)
    assert "predicted_arch" in result
    assert "confidence" in result
    assert "scores" in result
    assert len(result["scores"]) == 3
    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"
