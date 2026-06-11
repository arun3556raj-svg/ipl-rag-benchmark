"""Derive ground truth answers from ipl_universe.db and write ground_truth.json.

Run with:  uv run python data/build_ground_truth.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ipl_universe.db"
OUT_PATH = ROOT / "data" / "ground_truth.json"


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _one(con, sql, params=()):
    row = con.execute(sql, params).fetchone()
    return dict(row) if row else {}


def _all(con, sql, params=()):
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def build(con: sqlite3.Connection) -> list[dict]:
    questions = []

    # ------------------------------------------------------------------ #
    # TEXT2SQL  (q001 – q020)  exact numeric / ranking / aggregation       #
    # ------------------------------------------------------------------ #

    r = _one(con, """
        SELECT p.player_name, SUM(ps.runs_scored) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q001",
        "question": "Who has scored the most runs in IPL history?",
        "ground_truth": f"{r['player_name']} with {r['total']} runs.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.wickets) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q002",
        "question": "Who has taken the most wickets in IPL history?",
        "ground_truth": f"{r['player_name']} with {r['total']} wickets.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT t.team_name, COUNT(*) AS wins
        FROM matches m JOIN teams t ON t.team_id = m.winner_id
        WHERE m.winner_id IS NOT NULL
        GROUP BY m.winner_id ORDER BY wins DESC LIMIT 1
    """)
    questions.append({
        "id": "q003",
        "question": "Which team has won the most matches in IPL history?",
        "ground_truth": f"{r['team_name']} with {r['wins']} wins.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, MAX(ps.highest_score) AS hs, ps.season,
               t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        ORDER BY hs DESC LIMIT 1
    """)
    questions.append({
        "id": "q004",
        "question": "What is the highest individual innings score in IPL?",
        "ground_truth": f"{r['player_name']} scored {r['hs']} runs in {r['season']}.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT v.venue_name, v.city, COUNT(*) AS matches
        FROM matches m JOIN venues v ON v.venue_id = m.venue_id
        GROUP BY m.venue_id ORDER BY matches DESC LIMIT 1
    """)
    questions.append({
        "id": "q005",
        "question": "Which venue has hosted the most IPL matches?",
        "ground_truth": f"{r['venue_name']} ({r['city']}) with {r['matches']} matches.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.sixes) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q006",
        "question": "Who has hit the most sixes in IPL history?",
        "ground_truth": f"{r['player_name']} with {r['total']} sixes.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.runs_scored) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.season = 2016
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q007",
        "question": "Who scored the most runs in the 2016 IPL season?",
        "ground_truth": f"{r['player_name']} with {r['total']} runs in 2016.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.wickets) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.season = 2020
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q008",
        "question": "Who took the most wickets in the 2020 IPL season?",
        "ground_truth": f"{r['player_name']} with {r['total']} wickets in 2020.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.fours) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q009",
        "question": "Who has hit the most fours in IPL history?",
        "ground_truth": f"{r['player_name']} with {r['total']} fours.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.hundreds) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q010",
        "question": "Which batsman has scored the most centuries in IPL?",
        "ground_truth": f"{r['player_name']} with {r['total']} IPL centuries.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name,
               ROUND(SUM(ps.runs_scored) * 100.0 / SUM(ps.balls_faced), 2) AS sr
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.balls_faced >= 500
        GROUP BY ps.player_id ORDER BY sr DESC LIMIT 1
    """)
    questions.append({
        "id": "q011",
        "question": "Which batsman has the highest career strike rate in IPL (minimum 500 balls faced)?",
        "ground_truth": f"{r['player_name']} with a strike rate of {r['sr']}.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT COUNT(*) AS total
        FROM player_stats WHERE hundreds >= 1
    """)
    total_hundreds = _one(con, "SELECT SUM(hundreds) AS total FROM player_stats")
    questions.append({
        "id": "q012",
        "question": "How many centuries have been scored in IPL history in total?",
        "ground_truth": f"{total_hundreds['total']} centuries have been scored in IPL history.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, COUNT(*) AS pom_count
        FROM matches m JOIN players p ON p.player_id = m.player_of_match_id
        WHERE m.player_of_match_id IS NOT NULL
        GROUP BY m.player_of_match_id ORDER BY pom_count DESC LIMIT 1
    """)
    questions.append({
        "id": "q013",
        "question": "Who has won the most Player of the Match awards in IPL history?",
        "ground_truth": f"{r['player_name']} with {r['pom_count']} Player of the Match awards.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT t.team_name, SUM(ps.runs_scored) AS total, ps.season
        FROM player_stats ps
        JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        JOIN teams t ON t.team_id = pt.team_id
        GROUP BY pt.team_id, ps.season ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q014",
        "question": "Which team scored the most cumulative runs in a single IPL season?",
        "ground_truth": f"{r['team_name']} scored the most runs in {r['season']} with {r['total']} total runs.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "medium",
    })

    r2 = _one(con, """
        SELECT p.player_name,
               ROUND(AVG(ps.economy), 2) AS avg_economy,
               SUM(ps.wickets) AS wkts
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.economy > 0
        GROUP BY ps.player_id
        HAVING wkts >= 50
        ORDER BY avg_economy ASC LIMIT 1
    """)
    questions.append({
        "id": "q015",
        "question": "Which bowler has the best economy rate in IPL (minimum 50 wickets)?",
        "ground_truth": f"{r2['player_name']} with an average economy of {r2['avg_economy']} runs per over (across seasons with {r2['wkts']} total wickets).",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.runs_scored) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.season = 2019
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q016",
        "question": "Who scored the most runs in the 2019 IPL season?",
        "ground_truth": f"{r['player_name']} with {r['total']} runs in 2019.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT t.team_name, SUM(ps.wickets) AS total
        FROM player_stats ps
        JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        JOIN teams t ON t.team_id = pt.team_id
        WHERE ps.season = 2013
        GROUP BY pt.team_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q017",
        "question": "Which team took the most wickets in the 2013 IPL season?",
        "ground_truth": f"{r['team_name']} with {r['total']} wickets in 2013.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.fifties) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q018",
        "question": "Which batsman has scored the most half-centuries (fifties) in IPL?",
        "ground_truth": f"{r['player_name']} with {r['total']} fifties.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT COUNT(DISTINCT season) AS seasons
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        WHERE p.player_name LIKE '%Dhoni%'
    """)
    questions.append({
        "id": "q019",
        "question": "How many IPL seasons has MS Dhoni played in?",
        "ground_truth": f"MS Dhoni has played in {r['seasons']} IPL seasons.",
        "expected_best": "text_to_sql",
        "category": "factual",
        "difficulty": "easy",
    })

    r = _one(con, """
        SELECT p.player_name, SUM(ps.runs_scored) AS total
        FROM player_stats ps JOIN players p ON p.player_id = ps.player_id
        WHERE ps.season = 2023
        GROUP BY ps.player_id ORDER BY total DESC LIMIT 1
    """)
    questions.append({
        "id": "q020",
        "question": "Who scored the most runs in the 2023 IPL season?",
        "ground_truth": f"{r['player_name']} with {r['total']} runs in 2023.",
        "expected_best": "text_to_sql",
        "category": "temporal",
        "difficulty": "easy",
    })

    # ------------------------------------------------------------------ #
    # HYBRID RAG  (q021 – q035)  narrative / season summary / career      #
    # ------------------------------------------------------------------ #

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.season,
               ps.highest_score, ps.fifties, ps.strike_rate,
               t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Kohli%' AND ps.season = 2016
        LIMIT 1
    """)
    questions.append({
        "id": "q021",
        "question": "How did Virat Kohli perform in the 2016 IPL season?",
        "ground_truth": (
            f"Virat Kohli had an exceptional 2016 season for {r.get('team_name','RCB')}, "
            f"scoring {r['runs_scored']} runs in {r['matches']} matches "
            f"with a highest score of {r['highest_score']} and {r['fifties']} fifties "
            f"at a strike rate of {r['strike_rate']}."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.season,
               ps.highest_score, ps.sixes, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Gayle%' AND ps.season = 2012
        LIMIT 1
    """)
    questions.append({
        "id": "q022",
        "question": "How did Chris Gayle perform in the 2012 IPL season?",
        "ground_truth": (
            f"Chris Gayle was destructive in 2012 for {r.get('team_name','RCB')}, "
            f"scoring {r['runs_scored']} runs in {r['matches']} matches "
            f"with {r['sixes']} sixes and a highest score of {r['highest_score']}."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    rows = _all(con, """
        SELECT ps.season, ps.runs_scored, ps.matches, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Rohit%Sharma%'
        ORDER BY ps.runs_scored DESC LIMIT 3
    """)
    top3 = "; ".join(f"{r['season']}: {r['runs_scored']} runs for {r['team_name']}" for r in rows)
    questions.append({
        "id": "q023",
        "question": "Describe Rohit Sharma's batting career across IPL seasons.",
        "ground_truth": f"Rohit Sharma's top 3 seasons by runs: {top3}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.highest_score,
               ps.sixes, ps.strike_rate, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%de Villiers%' OR p.player_name LIKE '%AB%'
        ORDER BY ps.runs_scored DESC LIMIT 1
    """)
    questions.append({
        "id": "q024",
        "question": "What made AB de Villiers a standout IPL performer?",
        "ground_truth": (
            f"AB de Villiers' best season saw him score {r['runs_scored']} runs "
            f"in {r['matches']} matches for {r.get('team_name','RCB')} "
            f"with {r['sixes']} sixes at a strike rate of {r['strike_rate']}."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT t.team_name,
               SUM(CASE WHEN m.winner_id = pt.team_id THEN 1 ELSE 0 END) AS wins,
               COUNT(*) AS total
        FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        JOIN (SELECT DISTINCT team_id FROM teams WHERE team_name LIKE '%Sunrisers%') pt
            ON pt.team_id = t.team_id
        WHERE m.season = 2016
        GROUP BY t.team_id LIMIT 1
    """)
    # Simpler approach
    srh_2016 = _one(con, """
        SELECT COUNT(*) AS wins FROM matches m
        JOIN teams t ON t.team_id = m.winner_id
        WHERE t.team_name LIKE '%Sunrisers%' AND m.season = 2016
    """)
    srh_total = _one(con, """
        SELECT COUNT(*) AS total FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        WHERE t.team_name LIKE '%Sunrisers%' AND m.season = 2016
    """)
    questions.append({
        "id": "q025",
        "question": "How did Sunrisers Hyderabad perform in the 2016 IPL season?",
        "ground_truth": (
            f"Sunrisers Hyderabad won {srh_2016['wins']} matches "
            f"out of {srh_total['total']} in the 2016 season and won the IPL title."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    rows = _all(con, """
        SELECT ps.season, ps.runs_scored, ps.wickets, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Bumrah%'
        ORDER BY ps.season
    """)
    bumrah_summary = "; ".join(f"{r['season']}: {r['wickets']} wkts for {r['team_name']}" for r in rows[:5])
    questions.append({
        "id": "q026",
        "question": "Describe Jasprit Bumrah's bowling career in IPL.",
        "ground_truth": f"Jasprit Bumrah's wicket tally by season: {bumrah_summary}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.highest_score, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Yuvraj%' AND ps.season = 2014
        LIMIT 1
    """)
    questions.append({
        "id": "q027",
        "question": "How did Yuvraj Singh perform in the 2014 IPL season?",
        "ground_truth": (
            f"Yuvraj Singh played for {r.get('team_name','an IPL team')} in 2014, "
            f"scoring {r['runs_scored']} runs in {r['matches']} matches "
            f"with a highest score of {r['highest_score']}."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    rows = _all(con, """
        SELECT ps.season, ps.runs_scored, ps.matches, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%KL Rahul%' OR p.player_name LIKE '%Lokesh Rahul%'
        ORDER BY ps.season
    """)
    kl_summary = "; ".join(f"{r['season']}: {r['runs_scored']} runs for {r['team_name']}" for r in rows[:5])
    questions.append({
        "id": "q028",
        "question": "How has KL Rahul's IPL career evolved across seasons?",
        "ground_truth": f"KL Rahul's runs by season: {kl_summary}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    mi_2019 = _one(con, """
        SELECT COUNT(*) AS wins FROM matches m
        JOIN teams t ON t.team_id = m.winner_id
        WHERE t.team_name LIKE '%Mumbai%' AND m.season = 2019
    """)
    mi_total = _one(con, """
        SELECT COUNT(*) AS total FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        WHERE t.team_name LIKE '%Mumbai%' AND m.season = 2019
    """)
    questions.append({
        "id": "q029",
        "question": "How did Mumbai Indians perform in the 2019 IPL season?",
        "ground_truth": (
            f"Mumbai Indians won {mi_2019['wins']} of {mi_total['total']} "
            f"matches in 2019 and won the IPL title that season."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.highest_score,
               ps.sixes, t.team_name, ps.season
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Warner%' AND ps.season = 2019
        LIMIT 1
    """)
    questions.append({
        "id": "q030",
        "question": "How did David Warner perform in the 2019 IPL season?",
        "ground_truth": (
            f"David Warner scored {r['runs_scored']} runs in {r['matches']} "
            f"matches for {r.get('team_name','SRH')} in 2019 "
            f"with {r['sixes']} sixes and a highest score of {r['highest_score']}."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    csk_wins = _all(con, """
        SELECT m.season, COUNT(*) AS wins
        FROM matches m JOIN teams t ON t.team_id = m.winner_id
        WHERE t.team_name LIKE '%Chennai%'
        GROUP BY m.season ORDER BY m.season
    """)
    csk_str = "; ".join(f"{r['season']}: {r['wins']} wins" for r in csk_wins)
    questions.append({
        "id": "q031",
        "question": "Summarise Chennai Super Kings' win record across IPL seasons.",
        "ground_truth": f"Chennai Super Kings wins per season: {csk_str}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.highest_score, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Raina%' AND ps.season = 2013
        LIMIT 1
    """)
    questions.append({
        "id": "q032",
        "question": "How did Suresh Raina perform in the 2013 IPL season?",
        "ground_truth": (
            f"Suresh Raina scored {r['runs_scored']} runs in {r['matches']} matches "
            f"for {r.get('team_name','CSK')} in 2013 with a highest score of {r['highest_score']}."
            if r else "No stats found for Suresh Raina in 2013."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    rcb_wins = _all(con, """
        SELECT m.season, COUNT(*) AS wins
        FROM matches m JOIN teams t ON t.team_id = m.winner_id
        WHERE t.team_name LIKE '%Royal Challengers%'
        GROUP BY m.season ORDER BY m.season
    """)
    rcb_str = "; ".join(f"{r['season']}: {r['wins']} wins" for r in rcb_wins)
    questions.append({
        "id": "q033",
        "question": "How has Royal Challengers Bangalore performed across IPL seasons?",
        "ground_truth": f"RCB wins per season: {rcb_str}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    r = _one(con, """
        SELECT ps.runs_scored, ps.wickets, ps.matches, ps.highest_score, t.team_name
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN player_teams pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        WHERE (p.player_name LIKE '%Pandya%' AND p.player_name LIKE '%Hardik%')
              AND ps.season = 2021
        LIMIT 1
    """)
    questions.append({
        "id": "q034",
        "question": "How did Hardik Pandya perform in the 2021 IPL season?",
        "ground_truth": (
            f"Hardik Pandya scored {r['runs_scored']} runs and took {r['wickets']} wickets "
            f"in {r['matches']} matches for {r.get('team_name','MI')} in 2021."
            if r else "No stats found for Hardik Pandya in 2021."
        ),
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    kkr_wins = _all(con, """
        SELECT m.season, COUNT(*) AS wins
        FROM matches m JOIN teams t ON t.team_id = m.winner_id
        WHERE t.team_name LIKE '%Kolkata%'
        GROUP BY m.season ORDER BY m.season
    """)
    kkr_str = "; ".join(f"{r['season']}: {r['wins']} wins" for r in kkr_wins)
    questions.append({
        "id": "q035",
        "question": "Summarise Kolkata Knight Riders' performance across IPL seasons.",
        "ground_truth": f"KKR wins per season: {kkr_str}.",
        "expected_best": "hybrid_rag",
        "category": "narrative",
        "difficulty": "medium",
    })

    # ------------------------------------------------------------------ #
    # LIGHT RAG  (q036 – q050)  relational / graph traversal              #
    # ------------------------------------------------------------------ #

    teams_dhoni = _all(con, """
        SELECT DISTINCT t.team_name FROM player_teams pt
        JOIN players p ON p.player_id = pt.player_id
        JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Dhoni%'
    """)
    dhoni_teams = ", ".join(r["team_name"] for r in teams_dhoni)
    questions.append({
        "id": "q036",
        "question": "Which IPL teams has MS Dhoni played for?",
        "ground_truth": f"MS Dhoni has played for: {dhoni_teams}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    venues_rcb = _all(con, """
        SELECT v.venue_name, COUNT(*) AS cnt
        FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE t.team_name LIKE '%Royal Challengers%'
        GROUP BY v.venue_id ORDER BY cnt DESC LIMIT 3
    """)
    rcb_venues = "; ".join(f"{r['venue_name']} ({r['cnt']} matches)" for r in venues_rcb)
    questions.append({
        "id": "q037",
        "question": "Which venues has Royal Challengers Bangalore played at most often?",
        "ground_truth": f"RCB's most common venues: {rcb_venues}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    pom_chepauk = _all(con, """
        SELECT p.player_name, COUNT(*) AS cnt
        FROM matches m
        JOIN venues v ON v.venue_id = m.venue_id
        JOIN players p ON p.player_id = m.player_of_match_id
        WHERE v.venue_name LIKE '%Chidambaram%' OR v.venue_name LIKE '%Chepauk%'
        GROUP BY m.player_of_match_id ORDER BY cnt DESC LIMIT 5
    """)
    pom_str = ", ".join(f"{r['player_name']} ({r['cnt']})" for r in pom_chepauk)
    questions.append({
        "id": "q038",
        "question": "Who won the most Player of the Match awards at MA Chidambaram Stadium (Chepauk)?",
        "ground_truth": f"Top POM winners at Chepauk: {pom_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    teams_raina = _all(con, """
        SELECT DISTINCT t.team_name FROM player_teams pt
        JOIN players p ON p.player_id = pt.player_id
        JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Raina%'
    """)
    raina_teams = ", ".join(r["team_name"] for r in teams_raina)
    questions.append({
        "id": "q039",
        "question": "Which IPL teams has Suresh Raina played for?",
        "ground_truth": f"Suresh Raina has played for: {raina_teams}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    pom_wankhede = _all(con, """
        SELECT p.player_name, COUNT(*) AS cnt
        FROM matches m
        JOIN venues v ON v.venue_id = m.venue_id
        JOIN players p ON p.player_id = m.player_of_match_id
        WHERE v.venue_name LIKE '%Wankhede%'
        GROUP BY m.player_of_match_id ORDER BY cnt DESC LIMIT 5
    """)
    pom_w_str = ", ".join(f"{r['player_name']} ({r['cnt']})" for r in pom_wankhede)
    questions.append({
        "id": "q040",
        "question": "Who won the most Player of the Match awards at Wankhede Stadium?",
        "ground_truth": f"Top POM winners at Wankhede: {pom_w_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    teams_sehwag = _all(con, """
        SELECT DISTINCT t.team_name FROM player_teams pt
        JOIN players p ON p.player_id = pt.player_id
        JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Sehwag%'
    """)
    sehwag_teams = ", ".join(r["team_name"] for r in teams_sehwag) if teams_sehwag else "Not found"
    questions.append({
        "id": "q041",
        "question": "Which IPL teams has Virender Sehwag played for?",
        "ground_truth": f"Virender Sehwag played for: {sehwag_teams}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    pom_kkr = _all(con, """
        SELECT p.player_name, COUNT(*) AS cnt
        FROM matches m
        JOIN teams t ON t.team_id = m.winner_id
        JOIN players p ON p.player_id = m.player_of_match_id
        WHERE t.team_name LIKE '%Kolkata%'
        GROUP BY m.player_of_match_id ORDER BY cnt DESC LIMIT 5
    """)
    pom_kkr_str = ", ".join(f"{r['player_name']} ({r['cnt']})" for r in pom_kkr)
    questions.append({
        "id": "q042",
        "question": "Who won the most Player of the Match awards for Kolkata Knight Riders?",
        "ground_truth": f"Top KKR POM winners: {pom_kkr_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    venues_mi = _all(con, """
        SELECT v.venue_name, COUNT(*) AS cnt
        FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE t.team_name LIKE '%Mumbai%'
        GROUP BY v.venue_id ORDER BY cnt DESC LIMIT 3
    """)
    mi_venues_str = "; ".join(f"{r['venue_name']} ({r['cnt']} matches)" for r in venues_mi)
    questions.append({
        "id": "q043",
        "question": "Where does Mumbai Indians play most of their matches?",
        "ground_truth": f"Mumbai Indians play most at: {mi_venues_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    teams_sachin = _all(con, """
        SELECT DISTINCT t.team_name FROM player_teams pt
        JOIN players p ON p.player_id = pt.player_id
        JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Tendulkar%' OR p.player_name LIKE '%Sachin%'
    """)
    sachin_teams = ", ".join(r["team_name"] for r in teams_sachin) if teams_sachin else "Not found"
    questions.append({
        "id": "q044",
        "question": "Which IPL teams did Sachin Tendulkar play for?",
        "ground_truth": f"Sachin Tendulkar played for: {sachin_teams}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    shared = _all(con, """
        SELECT p.player_name
        FROM players p
        WHERE p.player_id IN (
            SELECT pt1.player_id FROM player_teams pt1
            JOIN teams t1 ON t1.team_id = pt1.team_id AND t1.team_name LIKE '%Mumbai%'
        )
        AND p.player_id IN (
            SELECT pt2.player_id FROM player_teams pt2
            JOIN teams t2 ON t2.team_id = pt2.team_id AND t2.team_name LIKE '%Chennai%'
        )
        ORDER BY p.player_name
    """)
    shared_str = ", ".join(r["player_name"] for r in shared) if shared else "None found"
    questions.append({
        "id": "q045",
        "question": "Which players have played for both Mumbai Indians and Chennai Super Kings?",
        "ground_truth": f"Players who played for both MI and CSK: {shared_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "hard",
    })

    pom_eden = _all(con, """
        SELECT p.player_name, COUNT(*) AS cnt
        FROM matches m
        JOIN venues v ON v.venue_id = m.venue_id
        JOIN players p ON p.player_id = m.player_of_match_id
        WHERE v.venue_name LIKE '%Eden%'
        GROUP BY m.player_of_match_id ORDER BY cnt DESC LIMIT 5
    """)
    pom_eden_str = ", ".join(f"{r['player_name']} ({r['cnt']})" for r in pom_eden)
    questions.append({
        "id": "q046",
        "question": "Who won the most Player of the Match awards at Eden Gardens?",
        "ground_truth": f"Top POM winners at Eden Gardens: {pom_eden_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    most_teams = _all(con, """
        SELECT p.player_name, COUNT(DISTINCT pt.team_id) AS team_count
        FROM player_teams pt JOIN players p ON p.player_id = pt.player_id
        GROUP BY pt.player_id HAVING team_count >= 4
        ORDER BY team_count DESC LIMIT 5
    """)
    multi_team_str = ", ".join(f"{r['player_name']} ({r['team_count']} teams)" for r in most_teams)
    questions.append({
        "id": "q047",
        "question": "Which players have represented the most IPL teams?",
        "ground_truth": f"Players with most IPL teams: {multi_team_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "hard",
    })

    pom_hyderabad = _all(con, """
        SELECT p.player_name, COUNT(*) AS cnt
        FROM matches m
        JOIN venues v ON v.venue_id = m.venue_id
        JOIN players p ON p.player_id = m.player_of_match_id
        WHERE v.city LIKE '%Hyderabad%'
        GROUP BY m.player_of_match_id ORDER BY cnt DESC LIMIT 5
    """)
    pom_hyd_str = ", ".join(f"{r['player_name']} ({r['cnt']})" for r in pom_hyderabad)
    questions.append({
        "id": "q048",
        "question": "Who won the most Player of the Match awards in Hyderabad?",
        "ground_truth": f"Top POM winners in Hyderabad: {pom_hyd_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    teams_gayle = _all(con, """
        SELECT DISTINCT t.team_name FROM player_teams pt
        JOIN players p ON p.player_id = pt.player_id
        JOIN teams t ON t.team_id = pt.team_id
        WHERE p.player_name LIKE '%Gayle%'
    """)
    gayle_teams = ", ".join(r["team_name"] for r in teams_gayle)
    questions.append({
        "id": "q049",
        "question": "Which IPL teams has Chris Gayle played for?",
        "ground_truth": f"Chris Gayle played for: {gayle_teams}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "easy",
    })

    # Final question: multi-hop graph question
    venues_csk = _all(con, """
        SELECT v.venue_name, v.city, COUNT(*) AS cnt
        FROM matches m
        JOIN teams t ON t.team_id IN (m.team1_id, m.team2_id)
        JOIN venues v ON v.venue_id = m.venue_id
        WHERE t.team_name LIKE '%Chennai%'
        GROUP BY v.venue_id ORDER BY cnt DESC LIMIT 5
    """)
    csk_venues_str = "; ".join(f"{r['venue_name']}, {r['city']} ({r['cnt']} matches)" for r in venues_csk)
    questions.append({
        "id": "q050",
        "question": "Which venues has Chennai Super Kings played at most frequently in IPL?",
        "ground_truth": f"CSK's most frequent venues: {csk_venues_str}.",
        "expected_best": "light_rag",
        "category": "relational",
        "difficulty": "medium",
    })

    return questions


def main():
    con = _con()
    try:
        questions = build(con)
    finally:
        con.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for q in questions:
        counts[q["expected_best"]] = counts.get(q["expected_best"], 0) + 1
    print(json.dumps({"total": len(questions), "by_arch": counts}, indent=2))


if __name__ == "__main__":
    main()
