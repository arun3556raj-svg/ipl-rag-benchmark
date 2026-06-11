# IPL Universe Database Schema

Plain english documentation of the SQLite database used by every retrieval architecture in this project. The LLM reads this as system context when generating SQL.

## Overview

`ipl_universe.db` is a normalized SQLite database covering Indian Premier League cricket. **295,732 ball by ball deliveries across 1,243 matches.** Seven substantive tables.

## Tables

### deliveries
The atomic fact table. One row per ball bowled in any match.

| column | type | meaning |
|---|---|---|
| delivery_id | integer | primary key |
| innings_id | integer | foreign key to innings |
| over_number | integer | 0 indexed over within the innings (0 to 19 in T20) |
| ball_number | integer | 1 indexed ball within the over |
| batter_id | integer | foreign key to players |
| bowler_id | integer | foreign key to players |
| non_striker_id | integer | foreign key to players |
| runs_batter | integer | runs scored off the bat |
| runs_extras | integer | runs from extras |
| runs_total | integer | total runs from the ball |
| extras_type | text | wides, byes, legbyes, noballs, penalty, or null |
| is_wicket | integer | 1 if a wicket fell on this ball, else 0 |
| wicket_type | text | how the wicket fell |
| player_out_id | integer | which player got out |
| fielder1_id | integer | first fielder involved in the dismissal |
| fielder2_id | integer | second fielder involved in the dismissal |
| is_four | integer | 1 if the ball went for four |
| is_six | integer | 1 if the ball went for six |

### innings
One row per innings (usually two per match in T20).

| column | type | meaning |
|---|---|---|
| innings_id | integer | primary key |
| match_id | integer | foreign key to matches |
| innings_number | integer | 1 or 2, occasionally more for super overs |
| batting_team_id | integer | foreign key to teams |
| bowling_team_id | integer | foreign key to teams |
| total_runs | integer | total runs scored in the innings |
| total_wickets | integer | wickets that fell |
| total_overs | real | overs bowled, decimal format like 19.4 |
| extras | integer | total extras |
| target | integer | runs required to win (second innings only) |
| is_super_over | integer | 1 if super over, else 0 |

### matches
One row per match. Match level metadata.

| column | type | meaning |
|---|---|---|
| match_id | integer | primary key |
| cricsheet_id | text | original cricsheet identifier |
| season | integer | IPL season year |
| match_number | integer | sequence number within the season |
| match_type | text | league, qualifier, eliminator, final |
| date | text | match date in ISO format |
| venue_id | integer | foreign key to venues |
| team1_id | integer | foreign key to teams |
| team2_id | integer | foreign key to teams |
| toss_winner_id | integer | which team won the toss |
| toss_decision | text | bat or field |
| winner_id | integer | which team won the match |
| result | text | runs, wickets, tie, no result |
| result_margin | integer | margin of victory in that unit |
| player_of_match_id | integer | foreign key to players |
| umpire1, umpire2, tv_umpire, match_referee | text | officials |

### players
Player roster.

| column | type | meaning |
|---|---|---|
| player_id | integer | primary key |
| player_name | text | full name |
| cricsheet_id | text | original identifier |
| batting_style | text | RHB or LHB |
| bowling_style | text | RF, LBG, etc |
| role | text | batsman, bowler, all rounder, wicketkeeper |
| nationality | text | country |

### teams
Nineteen teams across the history of the league.

| column | type | meaning |
|---|---|---|
| team_id | integer | primary key |
| team_name | text | full name like Chennai Super Kings |
| short_name | text | abbreviation like CSK |
| home_city | text | home city |
| primary_color | text | hex color |

### venues
Sixty four venues used across the league.

| column | type | meaning |
|---|---|---|
| venue_id | integer | primary key |
| venue_name | text | stadium name |
| city | text | city |
| country | text | country |

### player_stats
Precomputed per player per season aggregates. Useful for fast lookups but `deliveries` is the source of truth. For exact counts on filtered queries, aggregate from `deliveries`.

Key columns: `player_id`, `season`, `matches`, `runs_scored`, `balls_faced`, `fours`, `sixes`, `highest_score`, `fifties`, `hundreds`, `wickets`, `economy`, `batting_avg`, `strike_rate`, `bowling_avg`.

## Joins

```
deliveries.batter_id    -> players.player_id
deliveries.bowler_id    -> players.player_id
deliveries.innings_id   -> innings.innings_id
innings.match_id        -> matches.match_id
innings.batting_team_id -> teams.team_id
matches.venue_id        -> venues.venue_id
matches.team1_id, team2_id, winner_id, toss_winner_id -> teams.team_id
matches.player_of_match_id -> players.player_id
```

## Quirks worth knowing

1. **over_number is 0 indexed.** First over is 0, last over in T20 is 19. Death overs are overs 15 to 19.
2. **deliveries is the source of truth.** player_stats may be stale. For exact filtered counts, aggregate from deliveries.
3. **Some tables are empty.** `curated_moments`, `head_to_head`, and `team_stats` exist as empty shells. Ignore them.
4. **Season is an integer year** like 2017 or 2024.
5. **A wicket is a row where is_wicket = 1.** Use that to count dismissals.
6. **CRITICAL: deliveries has NO match_id column.** To reach match context from a delivery, you MUST go through innings: `deliveries.innings_id → innings.innings_id → innings.match_id → matches.match_id`. Never write `deliveries.match_id` — that column does not exist and will return empty results silently.
7. **Player batting score in a match** requires aggregating `runs_batter` from deliveries grouped by `(innings_id, batter_id)`. The per-match score is `SUM(d.runs_batter)` grouped on `i.match_id, d.batter_id` after joining `deliveries d JOIN innings i ON i.innings_id = d.innings_id`.

## Example questions and queries

### Most runs in death overs across all IPL seasons

```sql
SELECT p.player_name, SUM(d.runs_batter) AS total_runs
FROM deliveries d
JOIN players p ON p.player_id = d.batter_id
WHERE d.over_number >= 15
GROUP BY d.batter_id, p.player_name
ORDER BY total_runs DESC
LIMIT 1;
```

### Team win rate

```sql
SELECT t.team_name,
  COUNT(*) AS matches_played,
  SUM(CASE WHEN m.winner_id = t.team_id THEN 1 ELSE 0 END) AS wins,
  ROUND(SUM(CASE WHEN m.winner_id = t.team_id THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS win_pct
FROM teams t
JOIN matches m ON m.team1_id = t.team_id OR m.team2_id = t.team_id
GROUP BY t.team_id, t.team_name
ORDER BY win_pct DESC;
```

### Player fifty-plus scores and team win status (IMPORTANT join pattern)

```sql
-- deliveries has NO match_id — must go through innings
SELECT
    COUNT(*) AS fifty_plus_scores,
    SUM(CASE WHEN m.winner_id = fifties.batting_team_id THEN 1 ELSE 0 END) AS wins
FROM (
    SELECT i.match_id, i.batting_team_id, SUM(d.runs_batter) AS bat_runs
    FROM deliveries d
    JOIN innings i ON i.innings_id = d.innings_id
    JOIN players p ON p.player_id = d.batter_id
    WHERE p.player_name LIKE '%Rahul%'
    GROUP BY i.match_id, i.innings_id
    HAVING SUM(d.runs_batter) >= 50
) AS fifties
JOIN matches m ON m.match_id = fifties.match_id;
```

### Most economical bowler in the powerplay across seasons

```sql
SELECT p.player_name,
  SUM(d.runs_total) AS runs_conceded,
  COUNT(*) AS balls_bowled,
  ROUND(SUM(d.runs_total) * 6.0 / COUNT(*), 2) AS economy
FROM deliveries d
JOIN players p ON p.player_id = d.bowler_id
WHERE d.over_number < 6
GROUP BY d.bowler_id, p.player_name
HAVING balls_bowled >= 600
ORDER BY economy ASC
LIMIT 5;
```
