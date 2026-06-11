"""Sanity tests for the Text2SQL architecture."""

from architectures import text_to_sql

DEMO_QUESTION = "Who scored the most runs in death overs across all IPL seasons?"


def test_database_exists():
    assert text_to_sql.DB_PATH.exists(), (
        f"Database not found at {text_to_sql.DB_PATH}. "
        "Did you copy ipl_universe.db into data/ ?"
    )


def test_schema_loads():
    schema = text_to_sql.load_schema()
    assert "deliveries" in schema
    assert "players" in schema
    assert "over_number" in schema


def test_mock_generates_sql_for_demo_question():
    sql = text_to_sql.generate_sql(DEMO_QUESTION, use_mock=True)
    assert "SELECT" in sql.upper()
    assert "deliveries" in sql
    assert "over_number" in sql


def test_demo_sql_executes_and_returns_a_row():
    sql = text_to_sql.generate_sql(DEMO_QUESTION, use_mock=True)
    result = text_to_sql.execute_sql(sql)
    assert len(result["rows"]) == 1
    assert isinstance(result["rows"][0][0], str)
    assert result["rows"][0][1] > 0


def test_answer_orchestrator_returns_full_shape():
    out = text_to_sql.answer(DEMO_QUESTION, use_mock=True)
    for key in ("question", "answer", "sql", "rows", "columns",
                "latency_ms", "sql_exec_ms", "cost_usd", "llm_calls",
                "use_mock"):
        assert key in out, f"Missing key: {key}"
    assert out["use_mock"] is True
    assert out["latency_ms"] > 0
    assert isinstance(out["answer"], str)
    assert len(out["answer"]) > 0


def test_unknown_question_does_not_crash():
    out = text_to_sql.answer("Random unanswerable cricket trivia?", use_mock=True)
    assert "answer" in out
