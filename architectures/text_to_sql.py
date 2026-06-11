"""Text2SQL retrieval architecture.

Given a natural language question about IPL cricket, generate a SQL query,
execute it against ipl_universe.db, and return a natural language answer.

This is the simplest of the three architectures. No embeddings, no graph,
no retrieval index. Just schema awareness plus SQL.

Phase 2 uses canned mock SQL for known questions. The real DeepSeek client
gets wired in once the api key is available.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ipl_universe.db"
SCHEMA_PATH = ROOT / "data" / "schema.md"

_SQL_SYSTEM = """You are an expert SQL analyst for an IPL cricket database.
Given the schema and a question, write ONE correct SQLite query.
Return ONLY the raw SQL — no markdown fences, no explanation, no comments.
The query must be a SELECT statement.

HARD RULES — violating any of these produces zero rows:

1. PLAYER NAMES: The database stores abbreviated initials, e.g. "V Kohli" not
   "Virat Kohli", "RG Sharma" not "Rohit Sharma". Always filter with
   LIKE '%Surname%' — never exact equality on a full first name.

2. deliveries HAS NO match_id COLUMN. This is the most common mistake.
   To filter deliveries by season or match, you MUST join through innings:
     deliveries d
     JOIN innings i ON i.innings_id = d.innings_id
     JOIN matches m ON m.match_id = i.match_id
   Never write d.match_id or deliveries.match_id — that column does not exist.

3. WICKETS: A wicket is a row where d.is_wicket = 1. To count wickets taken
   by a bowler in a season:
     SELECT p.player_name, COUNT(*) AS wickets
     FROM deliveries d
     JOIN innings i ON i.innings_id = d.innings_id
     JOIN matches m ON m.match_id = i.match_id
     JOIN players p ON p.player_id = d.bowler_id
     WHERE d.is_wicket = 1 AND m.season = <year>
     GROUP BY d.bowler_id, p.player_name
     ORDER BY wickets DESC LIMIT 1;

4. RUNS SCORED IN A MATCH/SEASON: Aggregate d.runs_batter per batter per
   innings — not per delivery. Group by (i.match_id, d.batter_id) or
   (m.season, d.batter_id) after the deliveries→innings→matches join.

5. SEASON is an integer year (e.g. 2020), not a string."""

_ANSWER_SYSTEM = """You are a cricket analyst. Given the question, the SQL that was run,
and the result rows, write a single concise sentence answering the question.
Use specific numbers. No preamble, no markdown."""


def load_schema() -> str:
    """Load the plain english schema doc shipped with the database."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def generate_sql(question: str, use_mock: bool = True) -> tuple[str, float, float]:
    """Produce a SQL query. Returns (sql, latency_ms, cost_usd)."""
    if use_mock:
        return _mock_generate_sql(question), 0.0, 0.0
    from architectures.llm import chat
    schema = load_schema()
    sql, ms, cost = chat(
        system=_SQL_SYSTEM,
        user=f"Schema:\n{schema}\n\nQuestion: {question}",
        temperature=0.0,
        max_tokens=512,
    )
    # Strip any accidental markdown fences
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.I).strip().strip("`")
    return sql, ms, cost


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
) -> tuple[str, float, float]:
    """Turn SQL rows into a natural language sentence. Returns (text, ms, cost)."""
    if use_mock:
        return _mock_format_answer(rows), 0.0, 0.0
    from architectures.llm import chat
    rows_text = str(rows[:10]) if rows else "No rows returned."
    text, ms, cost = chat(
        system=_ANSWER_SYSTEM,
        user=f"Question: {question}\nSQL: {sql}\nRows: {rows_text}",
        temperature=0.3,
        max_tokens=200,
    )
    return text, ms, cost


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

    sql, sql_gen_ms, sql_cost = generate_sql(question, use_mock=use_mock)
    exec_result = execute_sql(sql)
    rows = exec_result["rows"]
    formatted, fmt_ms, fmt_cost = format_answer(question, sql, rows, use_mock=use_mock)

    total_ms = (time.perf_counter() - overall_start) * 1000

    return {
        "question": question,
        "answer": formatted,
        "sql": sql,
        "rows": rows,
        "columns": exec_result["columns"],
        "latency_ms": round(total_ms, 1),
        "sql_exec_ms": round(exec_result["elapsed_ms"], 1),
        "cost_usd": round(sql_cost + fmt_cost, 6),
        "llm_calls": 0 if use_mock else 2,
        "use_mock": use_mock,
    }
