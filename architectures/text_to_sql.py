"""Text2SQL retrieval architecture.

Given a natural language question about IPL cricket, generate a SQL query,
execute it against ipl_universe.db, and return a natural language answer.

This is the simplest of the three architectures. No embeddings, no graph,
no retrieval index. Just schema awareness plus SQL.

Phase 2 uses canned mock SQL for known questions. The real DeepSeek client
gets wired in once the api key is available.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ipl_universe.db"
SCHEMA_PATH = ROOT / "data" / "schema.md"


def load_schema() -> str:
    """Load the plain english schema doc shipped with the database."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def generate_sql(question: str, use_mock: bool = True) -> str:
    """Produce a SQL query for the given natural language question.

    In mock mode, return canned SQL for known test questions. The real LLM
    call gets wired here once the DeepSeek api key is available.
    """
    if use_mock:
        return _mock_generate_sql(question)
    raise NotImplementedError("Real DeepSeek client not yet wired.")


def _mock_generate_sql(question: str) -> str:
    """Return canned SQL for known test questions."""
    q = question.lower()

    if "death over" in q and "most runs" in q:
        return (
            "SELECT p.player_name, SUM(d.runs_batter) AS total_runs\n"
            "FROM deliveries d\n"
            "JOIN players p ON p.player_id = d.batter_id\n"
            "WHERE d.over_number >= 15\n"
            "GROUP BY d.batter_id, p.player_name\n"
            "ORDER BY total_runs DESC\n"
            "LIMIT 1;"
        )

    if "team" in q and ("win rate" in q or "winning percentage" in q):
        return (
            "SELECT t.team_name,\n"
            "  COUNT(*) AS matches_played,\n"
            "  SUM(CASE WHEN m.winner_id = t.team_id THEN 1 ELSE 0 END) AS wins,\n"
            "  ROUND(SUM(CASE WHEN m.winner_id = t.team_id THEN 1 ELSE 0 END) "
            "* 100.0 / COUNT(*), 1) AS win_pct\n"
            "FROM teams t\n"
            "JOIN matches m ON m.team1_id = t.team_id OR m.team2_id = t.team_id\n"
            "GROUP BY t.team_id, t.team_name\n"
            "ORDER BY win_pct DESC\n"
            "LIMIT 5;"
        )

    # No canned SQL for this question. Return a query that surfaces that
    # transparently rather than fabricating something.
    return "SELECT 'mock has no canned SQL for this question' AS note;"


def execute_sql(sql: str) -> dict:
    """Run sql against the database. Return rows, column names, and timing."""
    start = time.perf_counter()
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    finally:
        con.close()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"rows": rows, "columns": cols, "elapsed_ms": elapsed_ms}


def format_answer(
    question: str,
    sql: str,
    rows: list,
    use_mock: bool = True,
) -> str:
    """Turn SQL rows into a natural language sentence.

    Mock mode formats the top row tersely. The real LLM call goes here once
    the api key is available.
    """
    if use_mock:
        return _mock_format_answer(rows)
    raise NotImplementedError("Real DeepSeek client not yet wired.")


def _mock_format_answer(rows: list) -> str:
    if not rows:
        return "No rows returned."
    row = rows[0]
    if len(row) == 2:
        try:
            return f"The top result is {row[0]} with a value of {int(row[1]):,}."
        except (TypeError, ValueError):
            return f"The top result is {row[0]} with a value of {row[1]}."
    return f"Top row: {row}"


def answer(question: str, use_mock: bool = True) -> dict:
    """Run the full Text2SQL pipeline and return a structured result."""
    overall_start = time.perf_counter()

    sql = generate_sql(question, use_mock=use_mock)
    exec_result = execute_sql(sql)
    rows = exec_result["rows"]
    formatted = format_answer(question, sql, rows, use_mock=use_mock)

    total_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "question": question,
        "answer": formatted,
        "sql": sql,
        "rows": rows,
        "columns": exec_result["columns"],
        "latency_ms": round(total_ms, 1),
        "sql_exec_ms": round(exec_result["elapsed_ms"], 1),
        "cost_usd": 0.0,
        "llm_calls": 2,
        "use_mock": use_mock,
    }
