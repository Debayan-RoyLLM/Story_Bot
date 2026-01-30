from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import csv
from pathlib import Path

from app.db.database import get_db
from app.functions.fixtures_functions import (
    get_latest_fixture,
    get_info,
    get_player2_name,
    second_team_name,
    get_batting_team_id,
    get_current_run,
    bowling_team_total,
    bowling_team_id,
    get_current_player_run,
    get_current_player2_run,
    get_bowler_wickets,
    get_last_two_balls,
    get_team_wickets
)

router = APIRouter(prefix="/fixtures", tags=["Fixtures"])

GLOBAL_NARRATIVE = None
GLOBAL_GAME_STATE = None

CSV_PATH = Path("data/llm_outputs.csv")
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def ball_to_over(ball_no: int) -> float:
    over = (ball_no - 1) // 6
    ball = (ball_no - 1) % 6 + 1
    return over + ball / 10


@router.get("/latest")
def latest_fixture(country_id: int, league_id: int, db: Session = Depends(get_db)):

    global GLOBAL_NARRATIVE, GLOBAL_GAME_STATE

    fixture_id = get_latest_fixture(db, country_id, league_id)
    if not fixture_id:
        return {"message": "No fixture found"}

    narrative_rows = []
    game_state_rows = []

    total_balls = 120  # Assuming T20 match, adjust if needed

    for ball_no in range(1, total_balls + 1):
        over = ball_to_over(ball_no)
        data = get_info(db, fixture_id, over)

        if not data:
            continue

        data0 = data[0]

        batsman_id = data0["batsman_id"]
        non_striker_id = data0["non_striker_id"]

        player2_name = get_player2_name(db, non_striker_id)
        second_team = second_team_name(db, fixture_id, over)
        bowler_id = second_team[0]["bowler_id"]

        batting_team_id = get_batting_team_id(db, fixture_id, batsman_id)
        bowling_team = bowling_team_id(db, fixture_id, bowler_id)

        Target_Score = bowling_team_total(db, fixture_id, bowling_team)
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
        runs_required = Target_Score - live_score

        required_run_rate = round((runs_required / balls_remaining) * 6, 2) if balls_remaining > 0 else 0
        current_run_rate = round((live_score / total_balls_bowled) * 6, 2) if total_balls_bowled > 0 else 0

        team_wicket = get_team_wickets(db, fixture_id, batting_team_id, current_ball)
        wickets_in_hand = 10 - team_wicket

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
            GLOBAL_NARRATIVE = narrative
            GLOBAL_GAME_STATE = game_state_rows[-1]

            from app.services.LLM import run_statements, graph, metadata

            statements_output = run_statements(GLOBAL_NARRATIVE, GLOBAL_GAME_STATE, graph, metadata)

            file_exists = CSV_PATH.exists()

            with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "fixture_id",
                        "ball_no",
                        "question",
                        "sql_query",
                        "answer"
                    ])

                for item in statements_output:
                    writer.writerow([
                        fixture_id,
                        ball_no,
                        item.get("sentence"),
                        item.get("query"),
                        item.get("answer")
                    ])

    GLOBAL_NARRATIVE = (
        narrative_rows[-1]["narrative"]
        if narrative_rows and "narrative" in narrative_rows[-1]
        else ""
    )
    GLOBAL_GAME_STATE = game_state_rows[-1] if game_state_rows else {}

    from app.services.LLM import run_statements, graph, metadata

    statements_output = run_statements(GLOBAL_NARRATIVE, GLOBAL_GAME_STATE, graph, metadata)

    file_exists = CSV_PATH.exists()

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "fixture_id",
                "ball_no",
                "question",
                "sql_query",
                "answer"
            ])

        for item in statements_output:
            writer.writerow([
                fixture_id,
                narrative_rows[-1]["ball_no"] if narrative_rows else 0,
                item.get("sentence"),
                item.get("query"),
                item.get("answer")
            ])

    return {
        "fixture_id": fixture_id,
        "balls_simulated": len(narrative_rows),
        "narratives_written": len(narrative_rows),
        "game_states_written": len(game_state_rows),
        "llm_outputs": statements_output
    }
