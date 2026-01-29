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
    table_to_png_bytes,
    validate_physical_data,
)


@st.cache_data(show_spinner=False)
def _load_cached(csv_path: str):
    return load_physical_data(csv_path)


def _default_index(options: list, desired: str) -> int:
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
            "Cannot find {!r} in the current folder.\n\n"
            "Place `physical_data_matches.csv` next to `app.py`.".format(csv_path)
        )
        st.stop()

    df = _load_cached(csv_path)
    try:
        validate_physical_data(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    teams = list_teams(df)
    team_default_idx = _default_index(teams, "FC Den Bosch")

    with st.sidebar:
        st.header("Filters")

        team = st.selectbox("Team", teams, index=team_default_idx)
        players = list_players_for_team(df, team) if team else []
        player_1 = st.selectbox("Player", players, index=0 if players else None)

        compare_options = ["(None)"] + players
        compare_selected = st.selectbox("Compare with second player (optional)", compare_options, index=0)
        player_2 = None if compare_selected == "(None)" else compare_selected
        if player_2 == player_1:
            player_2 = None

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        logo_path = Path("fc_den_bosch_logo.png")
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

    if not team or not player_1:
        st.info("Select a team and player in the sidebar.")
        st.stop()

    try:
        display_df, styler = build_player_match_overview(df, team, player_1, compare_player_name=player_2)
        
        title_text = "Player Match Physical Overview"
        caption_text = (
            "{} vs {} — {} (all matches)".format(player_1, player_2, team)
            if player_2
            else "{} — {} (per match)".format(player_1, team)
        )
        
        # Render the HTML table first
        html = styler_to_html(styler)
        height = estimate_table_height_px(n_rows=len(display_df))
        components.html(html, height=height, scrolling=False)
        
        # Then generate PNG + show download button underneath
        png_bytes = table_to_png_bytes(display_df, title=title_text, caption=caption_text, dpi=200)
        file_name = "physical_table_{}_{}.png".format(
            str(player_1).replace(" ", "_"),
            ("vs_" + str(player_2).replace(" ", "_")) if player_2 else "solo",
        )
        
        st.download_button(
            label="Download Table as PNG",
            data=png_bytes,
            file_name=file_name,
            mime="image/png",
        )

    except Exception as e:
        st.error(str(e))


if __name__ == "__main__":
    main()
