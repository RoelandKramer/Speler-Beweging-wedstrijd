# file: app.py
# run: streamlit run app.py

from __future__ import annotations

from pathlib import Path

import streamlit as st

from load_table import (
    build_player_match_overview,
    list_players_for_team,
    list_teams,
    load_physical_data,
    validate_physical_data,
)


@st.cache_data(show_spinner=False)
def _load_cached(csv_path: str):
    return load_physical_data(csv_path)


def main() -> None:
    st.set_page_config(page_title="Player Match Physical Overview", layout="wide")
    st.title("Player Match Physical Overview")

    csv_path = "physical_data_matches.csv"
    if not Path(csv_path).exists():
        st.error(
            f"Cannot find {csv_path!r} in the current folder.\n\n"
            "Place `physical_data_matches.csv` next to `app.py`."
        )
        st.stop()

    df = _load_cached(csv_path)
    try:
        validate_physical_data(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    teams = list_teams(df)

    with st.sidebar:
        st.header("Filters")
        team = st.selectbox("Team", teams, index=0 if teams else None)

        players = list_players_for_team(df, team) if team else []
        player = st.selectbox("Player", players, index=0 if players else None)

    if not team or not player:
        st.info("Select a team and player in the sidebar.")
        st.stop()

    try:
        _, styler = build_player_match_overview(df, team, player)
        st.write(styler)
    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
