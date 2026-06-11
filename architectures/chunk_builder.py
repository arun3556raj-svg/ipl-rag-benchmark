"""Build text chunks from ipl_universe.db for Hybrid RAG retrieval.

Two chunk types are produced:

1. **Match chunks**  one per match, ~1,243 chunks. A narrative sentence
   summarising who played whom, where, who won, and by what margin.

2. **Player season chunks**  one per (player, season), ~3,386 chunks. A
   compact stat line covering matches played, runs scored, strike rate,
   wickets, economy.

Output is a JSON file at data/chunks.json. Each chunk is a dict:

    {
        "id": "match-42",
        "type": "match",
        "text": "On 2017-04-05 Sunrisers Hyderabad beat ...",
        "metadata": {...}
    }

The chunk file is the source of truth for both the BM25 index and the
vector index. Rebuild it whenever the database changes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ipl_universe.db"
CHUNKS_PATH = ROOT / "data" / "chunks.json"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def build_match_chunks(con: sqlite3.Connection) -> list[dict]:
    """One chunk per match. Joins teams, venues, and player of match."""
    rows = con.execute(
        """
        SELECT
            m.match_id,
            m.season,
            m.date,
            m.match_type,
            m.result,
            m.result_margin,
            t1.team_name AS team1_name,
            t2.team_name AS team2_name,
            tw.team_name AS winner_name,
            v.venue_name,
            v.city,
            p.player_name AS pom_name
        FROM matches m
        LEFT JOIN teams t1 ON t1.team_id = m.team1_id
        LEFT JOIN teams t2 ON t2.team_id = m.team2_id
        LEFT JOIN teams tw ON tw.team_id = m.winner_id
        LEFT JOIN venues v ON v.venue_id = m.venue_id
        LEFT JOIN players p ON p.player_id = m.player_of_match_id
        ORDER BY m.match_id
        """
    ).fetchall()

    chunks = []
    for r in rows:
        text = _match_sentence(r)
        chunks.append({
            "id": f"match-{r['match_id']}",
            "type": "match",
            "text": text,
            "metadata": {
                "match_id": r["match_id"],
                "season": r["season"],
                "match_type": r["match_type"],
                "team1": r["team1_name"],
                "team2": r["team2_name"],
                "winner": r["winner_name"],
                "venue": r["venue_name"],
                "city": r["city"],
                "player_of_match": r["pom_name"],
            },
        })
    return chunks


def _match_sentence(r) -> str:
    date = r["date"] or "unknown date"
    team1 = r["team1_name"] or "Team A"
    team2 = r["team2_name"] or "Team B"
    venue = r["venue_name"] or "an unknown venue"
    city = r["city"] or ""
    venue_phrase = f"at {venue}" + (f" in {city}" if city else "")

    season = r["season"]
    match_type = r["match_type"] or "league"
    type_phrase = "" if match_type == "league" else f" ({match_type})"

    winner = r["winner_name"]
    result = r["result"]
    margin = r["result_margin"]
    pom = r["pom_name"]

    if winner and result and margin is not None:
        result_phrase = f"{winner} beat {team2 if winner == team1 else team1} by {margin} {result}"
    elif winner:
        result_phrase = f"{winner} won"
    else:
        result_phrase = "the match had no result"

    pom_phrase = f" {pom} was player of the match." if pom else ""

    return (
        f"In IPL season {season}, on {date}, {team1} played {team2} "
        f"{venue_phrase}{type_phrase}. {result_phrase}.{pom_phrase}"
    )


def build_player_season_chunks(con: sqlite3.Connection) -> list[dict]:
    """One chunk per (player, season) using player_stats table."""
    rows = con.execute(
        """
        SELECT
            ps.stat_id,
            ps.player_id,
            ps.season,
            ps.matches,
            ps.runs_scored,
            ps.balls_faced,
            ps.fours,
            ps.sixes,
            ps.highest_score,
            ps.fifties,
            ps.hundreds,
            ps.wickets,
            ps.economy,
            ps.batting_avg,
            ps.strike_rate,
            ps.bowling_avg,
            p.player_name,
            p.role,
            t.team_name AS team_name,
            t.short_name AS team_short
        FROM player_stats ps
        JOIN players p ON p.player_id = ps.player_id
        LEFT JOIN (
            SELECT player_id, season, MIN(team_id) AS team_id
            FROM player_teams
            GROUP BY player_id, season
        ) pt ON pt.player_id = ps.player_id AND pt.season = ps.season
        LEFT JOIN teams t ON t.team_id = pt.team_id
        ORDER BY ps.player_id, ps.season
        """
    ).fetchall()

    chunks = []
    for r in rows:
        text = _player_season_sentence(r)
        chunks.append({
            "id": f"player-{r['player_id']}-season-{r['season']}",
            "type": "player_season",
            "text": text,
            "metadata": {
                "player_id": r["player_id"],
                "player_name": r["player_name"],
                "season": r["season"],
                "team": r["team_name"],
                "role": r["role"],
                "runs_scored": r["runs_scored"],
                "wickets": r["wickets"],
            },
        })
    return chunks


def _player_season_sentence(r) -> str:
    name = r["player_name"] or "Unknown player"
    season = r["season"]
    team = r["team_name"] or "an IPL team"

    bits = [f"In the {season} IPL season, {name} played for {team}."]

    matches = r["matches"] or 0
    if matches:
        bits.append(f"He featured in {matches} matches.")

    runs = r["runs_scored"] or 0
    if runs:
        bf = r["balls_faced"] or 0
        sr = r["strike_rate"]
        sr_phrase = f" at a strike rate of {sr}" if sr else ""
        bits.append(f"He scored {runs} runs off {bf} balls{sr_phrase}.")

    fours = r["fours"] or 0
    sixes = r["sixes"] or 0
    if fours or sixes:
        bits.append(f"He hit {fours} fours and {sixes} sixes.")

    fifties = r["fifties"] or 0
    hundreds = r["hundreds"] or 0
    highest = r["highest_score"] or 0
    if fifties or hundreds or highest:
        bits.append(
            f"His highest score was {highest} with {fifties} fifties "
            f"and {hundreds} hundreds."
        )

    wickets = r["wickets"] or 0
    if wickets:
        econ = r["economy"]
        econ_phrase = f" with an economy of {econ}" if econ else ""
        bits.append(f"He took {wickets} wickets{econ_phrase}.")

    return " ".join(bits)


def build_all_chunks() -> list[dict]:
    con = _connect()
    try:
        match_chunks = build_match_chunks(con)
        player_chunks = build_player_season_chunks(con)
    finally:
        con.close()
    return match_chunks + player_chunks


def materialise(path: Path = CHUNKS_PATH) -> dict:
    """Build chunks and write to disk. Return a small summary."""
    chunks = build_all_chunks()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {"total": len(chunks)}
    for c in chunks:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    return {"path": str(path), "counts": counts}


if __name__ == "__main__":
    summary = materialise()
    print(json.dumps(summary, indent=2))
