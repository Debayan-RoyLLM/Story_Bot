"""
Cricket Planning Context

Shared cricket domain knowledge injected into all DIN-SQL planning steps
(schema linking, classification, decomposition) so the LLM understands
column disambiguation, metrics formulas, ball format, and table semantics
*before* writing SQL.
"""

CRICKET_PLANNING_CONTEXT = """
=== CRICKET DATABASE QUICK REFERENCE ===

TABLES (all in schema history2):
  fixtures__balls   — one row per ball (PRIMARY table, has player names)
  fixtures__batting  — batsman stats per ball update
  fixtures__bowling  — bowler stats per over update
  fixtures__runs     — innings totals (2 rows per match: inning 1 & 2)

KEY COLUMN DISAMBIGUATION:
  "score"   → fixtures__balls.score__runs (per-ball, 0-6)
            → fixtures__batting.score (batsman total in innings)
            → fixtures__runs.score (team innings total)
            → fixtures__bowling has NO 'score' — use 'runs'
  "ball"    → fixtures__balls.ball (over.ball decimal format — see BALL CONVERSION below)
            → fixtures__batting.ball
            → fixtures__bowling uses 'overs' NOT 'ball'
  "wickets" → fixtures__runs.wickets (wickets LOST, not in hand)
            → fixtures__bowling.wickets (taken by bowler)
            → fixtures__balls.score__is_wicket (0/1 per ball)
  "inning"  → ONLY in fixtures__runs (1=batting first, 2=chasing)
  "scoreboard" → fixtures__balls: S1=bowling team, S2=batting/chasing team

BALL FORMAT & CONVERSION (critical — LLMs often get this wrong):
  The 'ball' column is stored as DECIMAL in over.ball_in_over format:
    ball = 0.1 → 1st ball of match (over 1, ball 1) = ball number 1
    ball = 0.6 → 6th ball of match (over 1, ball 6) = ball number 6
    ball = 1.1 → 7th ball (over 2, ball 1) = ball number 7
    ball = 5.3 → over 6, ball 3 = (5 × 6) + 3 = ball number 33
    ball = 9.5 → over 10, ball 5 = (9 × 6) + 5 = ball number 59
    ball = 19.6 → last ball of T20 = (19 × 6) + 6 = ball number 120

  CONVERSION FORMULA: actual_ball_number = FLOOR(ball) * 6 + (ball % 1) * 10
  BUT you do NOT need this formula — just use COUNT(*) to count balls.

  IMPORTANT:
    - ball = 5.7 does NOT exist — after 5.6 comes 6.1
    - Each over has ONLY balls .1 through .6 (6 balls per over)
    - Do NOT treat ball as a normal decimal (5.3 is NOT between 5.2 and 5.4 in cricket)
    - Do NOT use SUM(ball) or ball/6 — ball is NOT a sequential number
    - To count balls: use COUNT(*) from fixtures__balls
    - To count overs: use CAST(COUNT(*) AS FLOAT) / 6.0
    - "After 30 balls" = COUNT(*) > 30, NOT ball > 5.0
    - "First 5 overs" = WHERE ball <= 4.6 (overs 1-5 → ball 0.1 to 4.6)
    - "Powerplay" (overs 1-6) = WHERE ball <= 5.6
    - "Death overs" (overs 16-20) = WHERE ball >= 15.1

PLAYER NAMES: ONLY in fixtures__balls (batsman__fullname, bowler__fullname)
TEAM NAMES: team__name in fixtures__balls; team_id is NUMERIC, never compare with strings

VALID JOINS:
  fixtures__balls.fixture_id = fixtures__batting.fixture_id
  fixtures__balls.ball        = fixtures__batting.ball
  fixtures__balls.fixture_id = fixtures__bowling.fixture_id
  fixtures__balls.ball        = fixtures__bowling.overs
  fixtures__balls.fixture_id = fixtures__runs.fixture_id

CRICKET METRICS (compute from fixtures__balls unless noted):
  Strike Rate = CAST(SUM(score__runs) AS FLOAT) / NULLIF(COUNT(*), 0) * 100
  Economy     = CAST(SUM(score__runs) AS FLOAT) * 6.0 / NULLIF(COUNT(*), 0)
  Run Rate    = CAST(SUM(score__runs) AS FLOAT) * 6.0 / NULLIF(COUNT(*), 0)
  Batting Avg = CAST(SUM(score__runs) AS FLOAT) / NULLIF(SUM(CAST(score__out AS INT)), 0)
  Bowling SR  = CAST(COUNT(*) AS FLOAT) / NULLIF(SUM(CAST(score__out AS INT)), 0)
  Dot Ball %  = CAST(SUM(CASE WHEN score__runs = 0 THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) * 100
  Boundary %  = CAST(SUM(CASE WHEN score__runs IN (4, 6) THEN 1 ELSE 0 END) AS FLOAT) / NULLIF(COUNT(*), 0) * 100

WICKETS SEMANTICS:
  "5 wickets in hand" → fixtures__runs.wickets <= 5 (wickets = wickets LOST)
  "lost 3 wickets"    → fixtures__runs.wickets >= 3
  "all out"           → fixtures__runs.wickets = 10

CHASE LOGIC:
  Chasing team = inning = 2 AND scoreboard = 'S2'
  Successful chase = chasing score > first innings score (compare within same fixture_id)
  Target = first innings score + 1

MOST QUERIES NEED ONLY fixtures__balls (NO JOIN):
  Bowler economy, dot balls, wickets per ball → fixtures__balls WHERE bowler__fullname = 'X'
  Batsman strike rate, runs, boundaries       → fixtures__balls WHERE batsman__fullname = 'X'
  Team run rate, phase stats                  → fixtures__balls WHERE team__name = 'X'
  DO NOT join fixtures__bowling or fixtures__batting for these queries.
  Joining causes row multiplication (120 balls × 12 bowling rows = 1440 rows → inflated results).

Use fixtures__runs DIRECTLY (no join to balls) for:
  Team total score, wickets lost, successful chase comparisons.

DATA NOT AVAILABLE: dates, seasons, venues, partnerships, toss, weather.
"""
