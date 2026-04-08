"""
Streamlit UI for AI Storyboard.

Streams from the FastAPI backend (`/fixtures/latest/stream`) and renders each
story as soon as the pipeline finishes producing it (every 5 balls).

Run:
    # 1. start the API in one terminal
    uvicorn app.main:app --reload
    # 2. start the UI in another
    streamlit run streamlit_app.py
"""

import json
import os
import requests
import streamlit as st

API_URL = os.getenv("AI_STORYBOARD_API_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{API_URL}/fixtures/latest/stream"

st.set_page_config(page_title="AI Storyboard", layout="wide")

st.title("AI Storyboard — Cricket Statistics Engine")
st.caption(
    "Simulates a T20 match ball-by-ball and, every 5 balls, generates "
    "statistical questions, SQL queries, and natural-language answers — "
    "shown live as each batch is produced."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.sidebar.header("Fixture lookup")

mode = st.sidebar.radio(
    "Lookup mode",
    ["By names", "By IDs", "By fixture ID"],
    index=0,
)

params: dict = {}

if mode == "By names":
    st.sidebar.markdown("**Name-based lookup** (all five required)")
    country_name = st.sidebar.text_input("Country name", value="India")
    league_code = st.sidebar.text_input("League code", value="IPL")
    season_code = st.sidebar.text_input("Season code", value="2024")
    localteam_code = st.sidebar.text_input("Local team code", value="MI")
    visitorteam_code = st.sidebar.text_input("Visitor team code", value="CSK")
    round_ = st.sidebar.text_input("Round (optional)", value="")

    params = {
        "country_name": country_name,
        "league_code": league_code,
        "season_code": season_code,
        "localteam_code": localteam_code,
        "visitorteam_code": visitorteam_code,
    }
    if round_.strip():
        params["round"] = round_.strip()

elif mode == "By IDs":
    st.sidebar.markdown("**ID-based lookup** (legacy)")
    country_id = st.sidebar.number_input("Country ID", min_value=1, value=1, step=1)
    league_id = st.sidebar.number_input("League ID", min_value=1, value=1, step=1)
    params = {"country_id": int(country_id), "league_id": int(league_id)}

else:  # By fixture ID
    st.sidebar.markdown("**Direct fixture override**")
    fixture_id = st.sidebar.number_input("Fixture ID", min_value=1, value=1, step=1)
    params = {"fixture_id": int(fixture_id)}

run_btn = st.sidebar.button("Run pipeline", type="primary", use_container_width=True)
st.sidebar.divider()
st.sidebar.caption(f"Streaming endpoint:\n`{STREAM_ENDPOINT}`")


# ---------------------------------------------------------------------------
# Render a single batch (one set of stories produced after a 5-ball window)
# ---------------------------------------------------------------------------
def render_batch(evt: dict, batch_index: int) -> None:
    ball_no = evt.get("ball_no", "?")
    over = evt.get("over", "?")
    label = f"Batch {batch_index} — ball {ball_no} (over {over})"
    if evt.get("final"):
        label += " — final"

    with st.expander(label, expanded=True):
        narrative = evt.get("narrative")
        if narrative:
            st.markdown(f"**Narrative:** {narrative}")

        outputs = evt.get("llm_outputs") or []
        if not outputs:
            st.info("No questions passed the ML filter for this batch.")
            return

        for i, item in enumerate(outputs, start=1):
            question = item.get("sentence") or item.get("question") or "(no question)"
            st.markdown(f"**Q{i}.** {question}")

            answer = item.get("answer")
            if answer:
                st.write(answer)

            st.markdown("---")


# ---------------------------------------------------------------------------
# Trigger the streaming run
# ---------------------------------------------------------------------------
if run_btn:
    status = st.empty()
    progress = st.progress(0.0, text="Connecting...")
    error_holder = st.container()
    stories_header = st.empty()
    stories_area = st.container()

    batch_count = 0
    error_count = 0

    try:
        with requests.get(STREAM_ENDPOINT, params=params, stream=True, timeout=None) as resp:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                progress.empty()
                st.error(f"API returned {resp.status_code}: {detail}")
                st.stop()

            status.info("Streaming live — stories will appear below as they are generated.")

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    evt = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                t = evt.get("type")

                if t == "batch":
                    batch_count += 1
                    ball_no = evt.get("ball_no") or 0
                    if isinstance(ball_no, int) and ball_no > 0:
                        progress.progress(
                            min(ball_no / 120, 1.0),
                            text=f"Ball {ball_no}/120 — {batch_count} stories generated",
                        )
                    stories_header.subheader(f"Generated stories ({batch_count})")
                    with stories_area:
                        render_batch(evt, batch_count)

                elif t == "error":
                    error_count += 1
                    with error_holder:
                        st.warning(
                            f"Ball {evt.get('ball_no')}: {evt.get('error')}"
                        )

                elif t == "done":
                    progress.progress(
                        1.0,
                        text=f"Done — {batch_count} stories, {error_count} errors",
                    )
                    status.success(
                        f"Pipeline finished. Balls simulated: "
                        f"{evt.get('balls_simulated', 0)}, stories: {batch_count}, "
                        f"errors: {error_count}"
                    )

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not reach the API at `{STREAM_ENDPOINT}`. "
            "Is `uvicorn app.main:app --reload` running?"
        )
    except requests.exceptions.ChunkedEncodingError as e:
        st.error(f"Stream interrupted: {e}")
else:
    st.info("Configure the fixture in the sidebar, then click **Run pipeline**.")
