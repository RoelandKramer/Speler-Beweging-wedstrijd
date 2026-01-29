# file: app.py
# run: streamlit run app.py

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from load_table import (
    build_player_match_overview,
    estimate_table_height_px,
    list_players_for_team,
    list_teams,
    load_physical_data,
    styler_to_html,
    validate_physical_data,
)


@st.cache_data(show_spinner=False)
def _load_cached(csv_path: str):
    return load_physical_data(csv_path)


def _default_index(options: list[str], desired: str) -> int:
    desired_cf = desired.casefold()
    for i, opt in enumerate(options):
        if str(opt).casefold() == desired_cf:
            return i
    return 0

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
    default_team_name = "FC Den Bosch"
    team_default_idx = _default_index(teams, default_team_name)

    with st.sidebar:
        st.header("Filters")

        default_team_name = "FC Den Bosch"
        team_default_idx = _default_index(teams, default_team_name)
        
        team = st.selectbox("Team", teams, index=team_default_idx)
        
        players = list_players_for_team(df, team) if team else []
        player = st.selectbox("Player", players, index=0 if players else None)

        # Push logo to bottom
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='position: fixed; bottom: 18px; left: 18px; width: 280px; opacity: 0.98;'></div>",
            unsafe_allow_html=True,
        )

        logo_path = Path("fc_den_bosch_logo.png")
        if logo_path.exists():
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            st.image(str(logo_path), use_container_width=True)
        else:
            st.caption("Logo not found: fc_den_bosch_logo.png")

    if not team or not player:
        st.info("Select a team and player in the sidebar.")
        st.stop()

    try:
        display_df, styler = build_player_match_overview(df, team, player)
        html = styler_to_html(styler)
        height = estimate_table_height_px(n_rows=len(display_df))
        components.html(html, height=height, scrolling=False)
    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
