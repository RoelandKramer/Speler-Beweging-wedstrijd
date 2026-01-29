# file: load_table.py

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class OverviewColumns:
    match: str = "Match"
    player: str = "Player"
    pos: str = "Pos"
    minutes: str = "Min"
    total_distance: str = "Total distance (km)"
    m_per_min: str = "m/min"
    runs: str = "Runs"
    sprints: str = "Sprints"


STREAMLIT_FONT_STACK = (
    '"Source Sans 3","Source Sans Pro","Inter",system-ui,-apple-system,'
    '"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif'
)

REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "match_id",
    "match_name",
    "club",
    "Minutes",
    "total_distance",
    "hi_runs",
    "sprint_runs",
    "position",
}

BAR_COLORS = {
    "Total distance (km)": "#FDD8AC",
    "m/min": "#FDCCCD",
    "Runs": "#CFE2FC",
    "Sprints": "#AACAFC",
}

BAR_SCALES = {
    "Total distance (km)": (0, 13.0),  # 13,000m => 13.0km
    "m/min": (0, 140),
    "Runs": (0, 50),
    "Sprints": (0, 20),
}

ROW_BG_PLAYER_1 = "#FFFFFF"
ROW_BG_PLAYER_2 = "#E8F2FF"  # light blue


def load_physical_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def validate_physical_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: {}".format(sorted(missing)))


def list_teams(df: pd.DataFrame) -> list:
    return sorted(df["club"].dropna().astype(str).unique().tolist())


def list_players_for_team(df: pd.DataFrame, team_name: str) -> list:
    players = (
        df.loc[df["club"].astype(str) == str(team_name), "player_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(players)


def _parse_home_away_and_opponent(match_name: str, club: str) -> Tuple[str, str]:
    if not isinstance(match_name, str) or not match_name.strip():
        return "?", "Unknown"

    normalized = " ".join(match_name.strip().split())
    separators = [" vs ", " v ", " VS ", " Vs ", " V "]

    left = right = None
    for sep in separators:
        if sep in normalized:
            parts = normalized.split(sep)
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                break

    if left is None or right is None:
        return "?", normalized

    club_norm = club.strip().casefold()
    left_norm = left.casefold()
    right_norm = right.casefold()

    if club_norm == left_norm:
        return "Home", right
    if club_norm == right_norm:
        return "Away", left

    if club_norm in left_norm:
        return "Home", right
    if club_norm in right_norm:
        return "Away", left

    return "?", "{} vs {}".format(left, right)


def _match_date_from_match_id(match_id) -> Optional[pd.Timestamp]:
    try:
        s = str(int(match_id))
    except Exception:
        return None

    if len(s) != 8:
        return None

    try:
        return pd.to_datetime(s, format="%Y%m%d", errors="raise")
    except Exception:
        return None


def _compute_player_match_metrics(df: pd.DataFrame, team_name: str, player_name: str) -> pd.DataFrame:
    filtered = df.loc[
        (df["club"].astype(str) == str(team_name))
        & (df["player_name"].astype(str) == str(player_name))
    ].copy()

    if filtered.empty:
        return filtered.reset_index(drop=True)

    filtered["_match_date"] = filtered["match_id"].apply(_match_date_from_match_id)

    ha_opp = filtered.apply(
        lambda r: _parse_home_away_and_opponent(str(r["match_name"]), str(r["club"])),
        axis=1,
        result_type="expand",
    )
    filtered["_ha"] = ha_opp[0]
    filtered["_opp"] = ha_opp[1]

    minutes = pd.to_numeric(filtered["Minutes"], errors="coerce")
    total_m = pd.to_numeric(filtered["total_distance"], errors="coerce")
    runs = pd.to_numeric(filtered["hi_runs"], errors="coerce")
    sprints = pd.to_numeric(filtered["sprint_runs"], errors="coerce")

    out = pd.DataFrame(
        {
            "match_id": filtered["match_id"],
            "_match_date": filtered["_match_date"],
            "_match_id_num": pd.to_numeric(filtered["match_id"], errors="coerce"),
            "match_label": filtered.apply(lambda r: "{} • {}".format(r["_ha"], r["_opp"]), axis=1),
            "player_name": filtered["player_name"].astype(str),
            "position": filtered["position"].astype(str),
            "minutes": minutes,
            "total_km": total_m / 1000.0,
            "m_per_min": total_m / minutes,
            "runs": runs,
            "sprints": sprints,
        }
    )
    return out.reset_index(drop=True)


def build_player_match_overview(
    df: pd.DataFrame,
    team_name: str,
    team_player_name: str,
    compare_player_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
    """
    - Always show ALL matches for player 1.
    - If compare player provided: show ALL matches for player 2 too.
    - Sort by most recent match first.
    - If same match_id: player 1 row first.
    - Player 2 rows light blue; player 1 rows white.
    """
    validate_physical_data(df)
    cols = OverviewColumns()

    p1 = _compute_player_match_metrics(df, team_name, team_player_name)
    if p1.empty:
        raise ValueError("No rows found for club={!r} player_name={!r}".format(team_name, team_player_name))
    p1["_player_rank"] = 0
    p1["_row_type"] = "primary"

    compare_active = bool(compare_player_name) and str(compare_player_name) != str(team_player_name)
    if compare_active:
        p2 = _compute_player_match_metrics(df, team_name, str(compare_player_name))
        if not p2.empty:
            p2["_player_rank"] = 1
            p2["_row_type"] = "compare"
            combined_src = pd.concat([p1, p2], ignore_index=True)
        else:
            combined_src = p1.copy()
    else:
        combined_src = p1.copy()

    # sort: date desc, then match_id desc, then player_rank asc (player1 first)
    combined_src["_sort_date"] = combined_src["_match_date"].fillna(pd.Timestamp.min)
    combined_src = combined_src.sort_values(
        by=["_sort_date", "_match_id_num", "match_id", "_player_rank"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # build display table
    display_df = pd.DataFrame(
        {
            "_row_type": combined_src["_row_type"],
            cols.match: combined_src["match_label"],
            cols.player: combined_src["player_name"],
            cols.pos: combined_src["position"],
            cols.minutes: combined_src["minutes"],
            cols.total_distance: combined_src["total_km"],
            cols.m_per_min: combined_src["m_per_min"],
            cols.runs: combined_src["runs"],
            cols.sprints: combined_src["sprints"],
        }
    )

    # styler
    styler = display_df.style.hide(axis="index").hide(axis="columns", subset=["_row_type"])

    table_styles = [
        {
            "selector": "table",
            "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
                ("font-family", STREAMLIT_FONT_STACK),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("font-family", STREAMLIT_FONT_STACK),
                ("font-weight", "700"),
                ("font-size", "14px"),
                ("text-align", "left"),
                ("padding", "10px 12px"),
                ("border-bottom", "2px solid #ddd"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("font-family", STREAMLIT_FONT_STACK),
                ("font-size", "14px"),
                ("padding", "9px 12px"),
                ("border-bottom", "1px solid #eee"),
                ("vertical-align", "middle"),
            ],
        },
        {
            "selector": "caption",
            "props": [
                ("caption-side", "top"),
                ("font-weight", "700"),
                ("font-size", "14px"),
                ("padding", "0 0 10px 0"),
            ],
        },
    ]
    styler = styler.set_table_styles(table_styles)

    right_cols = [cols.minutes, cols.total_distance, cols.m_per_min, cols.runs, cols.sprints]
    styler = styler.set_properties(subset=right_cols, **{"text-align": "right"})
    styler = styler.set_properties(subset=[cols.match, cols.player, cols.pos], **{"text-align": "left"})

    styler = styler.format(
        {
            cols.minutes: "{:.0f}",
            cols.total_distance: "{:.3f}",
            cols.m_per_min: "{:.0f}",
            cols.runs: "{:.0f}",
            cols.sprints: "{:.0f}",
        },
        na_rep="—",
    )

    def _row_bg(row: pd.Series) -> list:
        if row.get("_row_type") == "compare":
            return ["background-color: {};".format(ROW_BG_PLAYER_2)] * len(row)
        return ["background-color: {};".format(ROW_BG_PLAYER_1)] * len(row)

    styler = styler.apply(_row_bg, axis=1)

    for col_name in [cols.total_distance, cols.m_per_min, cols.runs, cols.sprints]:
        vmin, vmax = BAR_SCALES[col_name]
        styler = styler.bar(
            subset=[col_name],
            align="left",
            color=BAR_COLORS[col_name],
            vmin=vmin,
            vmax=vmax,
        )

    caption = "{} — {} (per match)".format(team_player_name, team_name)
    if compare_active:
        caption = "{} vs {} — {} (all matches)".format(team_player_name, compare_player_name, team_name)
    styler = styler.set_caption(caption)

    return display_df.drop(columns=["_row_type"]), styler


def styler_to_html(styler: "pd.io.formats.style.Styler") -> str:
    html = styler.to_html()
    return """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
      body {{
        margin: 0;
        font-family: {font};
      }}
    </style>
    <div style="width:100%; overflow:visible; font-family:{font};">
      {table}
    </div>
    """.format(font=STREAMLIT_FONT_STACK, table=html)


def estimate_table_height_px(n_rows: int) -> int:
    header = 58
    caption = 44
    row_height = 42
    padding = 28
    return header + caption + (n_rows * row_height) + padding
