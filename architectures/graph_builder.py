"""Build a knowledge graph from ipl_universe.db for LightRAG retrieval.

Nodes  — three types:
  player  : {player_id, player_name, role}
  team    : {team_id, team_name, short_name}
  venue   : {venue_id, venue_name, city}

Edges  — four types:
  played_for   : (player) -> (team)          season attribute
  played_in    : (team)   -> (venue)         match_id, season
  won_against  : (team)   -> (team)          match_id, season, margin, result
  pom          : (player) -> (team,venue)    player-of-match, season

The graph is serialised to data/graph.pkl (NetworkX DiGraph via pickle).
Rebuild whenever the database changes.
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ipl_universe.db"
GRAPH_PATH = ROOT / "data" / "graph.pkl"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _add_nodes(G: nx.DiGraph, con: sqlite3.Connection) -> None:
    for row in con.execute("SELECT player_id, player_name, role FROM players").fetchall():
        G.add_node(
            f"player:{row['player_id']}",
            kind="player",
            player_id=row["player_id"],
            name=row["player_name"] or "Unknown",
            role=row["role"] or "",
        )

    for row in con.execute("SELECT team_id, team_name, short_name FROM teams").fetchall():
        G.add_node(
            f"team:{row['team_id']}",
            kind="team",
            team_id=row["team_id"],
            name=row["team_name"] or "Unknown",
            short=row["short_name"] or "",
        )

    for row in con.execute("SELECT venue_id, venue_name, city FROM venues").fetchall():
        G.add_node(
            f"venue:{row['venue_id']}",
            kind="venue",
            venue_id=row["venue_id"],
            name=row["venue_name"] or "Unknown",
            city=row["city"] or "",
        )


def _add_player_team_edges(G: nx.DiGraph, con: sqlite3.Connection) -> None:
    for row in con.execute(
        "SELECT player_id, team_id, season FROM player_teams"
    ).fetchall():
        src = f"player:{row['player_id']}"
        dst = f"team:{row['team_id']}"
        if src in G and dst in G:
            # Multiple seasons -> store as a list on the edge
            if G.has_edge(src, dst):
                G[src][dst]["seasons"].append(row["season"])
            else:
                G.add_edge(src, dst, rel="played_for", seasons=[row["season"]])


def _add_match_edges(G: nx.DiGraph, con: sqlite3.Connection) -> None:
    rows = con.execute(
        """
        SELECT
            m.match_id, m.season,
            m.team1_id, m.team2_id, m.winner_id, m.venue_id,
            m.result, m.result_margin,
            m.player_of_match_id
        FROM matches m
        """
    ).fetchall()

    for r in rows:
        t1 = f"team:{r['team1_id']}" if r["team1_id"] else None
        t2 = f"team:{r['team2_id']}" if r["team2_id"] else None
        v  = f"venue:{r['venue_id']}" if r["venue_id"] else None
        w  = f"team:{r['winner_id']}" if r["winner_id"] else None
        p  = f"player:{r['player_of_match_id']}" if r["player_of_match_id"] else None

        # team -> venue  (played_in)
        if t1 and v and t1 in G and v in G:
            G.add_edge(t1, v, rel="played_in",
                       match_id=r["match_id"], season=r["season"])
        if t2 and v and t2 in G and v in G:
            G.add_edge(t2, v, rel="played_in",
                       match_id=r["match_id"], season=r["season"])

        # winner -> loser  (won_against)
        if w and t1 and t2 and w in G:
            loser = t2 if w == t1 else t1
            if loser in G:
                G.add_edge(w, loser, rel="won_against",
                           match_id=r["match_id"], season=r["season"],
                           margin=r["result_margin"], result=r["result"])

        # player -> team  (player_of_match)
        if p and w and p in G and w in G:
            if G.has_edge(p, w) and G[p][w].get("rel") == "pom":
                G[p][w]["count"] = G[p][w].get("count", 1) + 1
            else:
                G.add_edge(p, w, rel="pom",
                           season=r["season"], count=1)


def build_graph() -> nx.DiGraph:
    con = _connect()
    try:
        G = nx.DiGraph()
        _add_nodes(G, con)
        _add_player_team_edges(G, con)
        _add_match_edges(G, con)
    finally:
        con.close()
    return G


def materialise(path: Path = GRAPH_PATH) -> dict:
    G = build_graph()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    return {
        "path": str(path),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_kinds": {
            k: sum(1 for _, d in G.nodes(data=True) if d.get("kind") == k)
            for k in ("player", "team", "venue")
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(materialise(), indent=2))
