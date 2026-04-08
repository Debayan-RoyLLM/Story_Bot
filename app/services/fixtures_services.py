import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import csv
from pathlib import Path

logger = logging.getLogger("app.services.fixtures")

from app.db.database import get_db
from app.functions.fixtures_functions import (
    get_fixture_by_names,
    get_latest_fixture,
    get_info,
    get_player2_name,
    second_team_name,
    get_batting_team_id,
    bowling_team_total,
    bowling_team_id,
    get_current_player_run,
    get_current_player2_run,
    get_bowler_wickets,
    get_last_two_balls,
    get_team_wickets
)

router = APIRouter(prefix="/fixtures", tags=["Fixtures"])

CSV_PATH = Path("data/llm_outputs_01.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def ball_to_over(ball_no: int) -> float:
    over = (ball_no - 1) // 6
    ball = (ball_no - 1) % 6 + 1
    return over + ball / 10


def _write_csv(fixture_id, ball_no, statements_output):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["fixture_id", "ball_no", "question", "sql_query", "answer"])
        for item in statements_output:
            writer.writerow([
                fixture_id,
                ball_no,
                item.get("sentence"),
                item.get("query"),
                item.get("answer")
            ])


def _run_fixture(fixture_id: int, db: Session):
    """Core logic to process a fixture by its ID."""
    narrative_rows = []
    game_state_rows = []
    errors = []

    total_balls = 120  # T20 match

    for ball_no in range(1, total_balls + 1):
        try:
            over = ball_to_over(ball_no)
            data = get_info(db, fixture_id, over)

            if not data:
                continue

            data0 = data[0]

            batsman_id = data0["batsman_id"]
            non_striker_id = data0["non_striker_id"]

            player2_name = get_player2_name(db, non_striker_id)
            second_team = second_team_name(db, fixture_id, over)

            if not second_team:
                logger.warning(f"Ball {ball_no}: no second team data, skipping")
                continue

            bowler_id = second_team[0]["bowler_id"]

            batting_team_id = get_batting_team_id(db, fixture_id, batsman_id)
            bowling_team = bowling_team_id(db, fixture_id, bowler_id)

            target_score = bowling_team_total(db, fixture_id, bowling_team)
            wickets_bowler = get_bowler_wickets(db, fixture_id, bowler_id, over)

            current_ball = second_team[0]["current_ball"]
            current_over = int(current_ball)
            balls_bowled = int((current_ball - current_over) * 10)
            total_balls_bowled = current_over * 6 + balls_bowled
            balls_remaining = 120 - total_balls_bowled

            batsman_runs, batsman_balls, strike_rate = get_current_player_run(
                db, fixture_id, batsman_id, current_ball
            )

            batsman2_runs, batsman2_balls = get_current_player2_run(
                db, fixture_id, non_striker_id, current_ball
            )

            live_score = batsman_runs + batsman2_runs
            runs_required = target_score - live_score

            required_run_rate = round((runs_required / balls_remaining) * 6, 2) if balls_remaining > 0 else 0
            current_run_rate = round((live_score / total_balls_bowled) * 6, 2) if total_balls_bowled > 0 else 0

            team_wicket = get_team_wickets(db, fixture_id, batting_team_id, current_ball)
            wickets_in_hand = 10 - (team_wicket or 0)

            last_two = get_last_two_balls(db, fixture_id, batting_team_id, current_ball)

            narrative = (
                f"{data0['batsman']} and {player2_name} are batting for {data0['team_name']} with "
                f"{runs_required} runs required off {balls_remaining} balls. "
                f"{data0['batsman']} is on {batsman_runs} off {batsman_balls} balls, "
                f"while {player2_name} is on {batsman2_runs} from {batsman2_balls}. "
                f"The bowler {data0['bowler']} has taken {wickets_bowler} wickets. "
                f"Required RR: {required_run_rate}, Current RR: {current_run_rate}. "
                f"Wickets in hand: {wickets_in_hand}. "
                f"Last two balls: {last_two['second_last_ball']}, {last_two['last_ball']}."
            )

            narrative_rows.append({
                "fixture_id": fixture_id,
                "ball_no": ball_no,
                "over": over,
                "narrative": narrative
            })

            game_state_rows.append({
                "req_runs": runs_required,
                "balls_remaining": balls_remaining,
                "batsman_total_runs": batsman_runs,
                "batsman_balls_faced": batsman_balls,
                "nonstriker_total_runs": batsman2_runs,
                "nonstriker_balls_faced": batsman2_balls,
                "team_run_rate": current_run_rate,
                "req_run_rate": required_run_rate,
                "wickets": wickets_in_hand,
                "second_last_ball": last_two["second_last_ball"],
                "last_ball": last_two["last_ball"]
            })

            # Check for match end conditions
            if runs_required <= 0 or wickets_in_hand <= 0:
                break

            if ball_no % 5 == 0:
                from app.services.llm import run_statements, graph, metadata

                statements_output = run_statements(narrative, game_state_rows[-1], graph, metadata, fixture_id=fixture_id)
                _write_csv(fixture_id, ball_no, statements_output)

        except Exception as e:
            logger.error(f"Ball {ball_no}: error processing — {e}")
            errors.append({"ball_no": ball_no, "error": str(e)})
            continue

    # Final LLM run on last ball state
    statements_output = []
    try:
        final_narrative = (
            narrative_rows[-1]["narrative"]
            if narrative_rows and "narrative" in narrative_rows[-1]
            else ""
        )
        final_game_state = game_state_rows[-1] if game_state_rows else {}

        if final_narrative:
            from app.services.llm import run_statements, graph, metadata

            statements_output = run_statements(final_narrative, final_game_state, graph, metadata, fixture_id=fixture_id)
            final_ball_no = narrative_rows[-1]["ball_no"] if narrative_rows else 0
            _write_csv(fixture_id, final_ball_no, statements_output)
    except Exception as e:
        logger.error(f"Final LLM run failed: {e}")
        errors.append({"ball_no": "final", "error": str(e)})

    return {
        "fixture_id": fixture_id,
        "balls_simulated": len(narrative_rows),
        "narratives_written": len(narrative_rows),
        "game_states_written": len(game_state_rows),
        "llm_outputs": statements_output,
        "errors": errors
    }


@router.get("/latest")
def latest_fixture(
    country_id: Optional[int] = None,
    league_id: Optional[int] = None,
    fixture_id: Optional[int] = None,
    country_name: Optional[str] = None,
    league_code: Optional[str] = None,
    season_code: Optional[str] = None,
    localteam_code: Optional[str] = None,
    visitorteam_code: Optional[str] = None,
    round: Optional[str] = None,
    db: Session = Depends(get_db)
):
    # Name-based lookup
    if not fixture_id and country_name and league_code and season_code and localteam_code and visitorteam_code:
        fixture_id = get_fixture_by_names(db, country_name, league_code, season_code, localteam_code, visitorteam_code, round)
        if not fixture_id:
            raise HTTPException(status_code=404, detail="No fixture found for the given names")

    # ID-based lookup
    if not fixture_id and country_id and league_id:
        fixture_id = get_latest_fixture(db, country_id, league_id)
        if not fixture_id:
            raise HTTPException(status_code=404, detail="No fixture found")

    if not fixture_id:
        raise HTTPException(
            status_code=400,
            detail="Provide (country_name, league_code, season_code, localteam_code, visitorteam_code) or (country_id, league_id)"
        )

    return _run_fixture(fixture_id, db)
