# file: load_table.py

from __future__ import annotations

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


def load_physical_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def validate_physical_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def list_teams(df: pd.DataFrame) -> list[str]:
    return sorted(df["club"].dropna().astype(str).unique().tolist())


def list_players_for_team(df: pd.DataFrame, team_name: str) -> list[str]:
    players = (
        df.loc[df["club"].astype(str) == str(team_name), "player_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(players)


def _parse_home_away_and_opponent(match_name: str, club: str) -> tuple[str, str]:
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

    return "?", f"{left} vs {right}"


def _match_date_from_match_id(match_id: object) -> Optional[pd.Timestamp]:
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
        return filtered

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
            "match_label": filtered.apply(lambda r: f"{r['_ha']} • {r['_opp']}", axis=1),
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


def _sort_player_matches_desc(p: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return p

    if p["_match_date"].notna().any():
        return p.sort_values("_match_date", ascending=False, kind="mergesort").reset_index(drop=True)

    mid_num = pd.to_numeric(p["match_id"], errors="coerce")
    return p.assign(_mid_num=mid_num).sort_values("_mid_num", ascending=False, kind="mergesort").drop(
        columns=["_mid_num"]
    ).reset_index(drop=True)


def build_player_match_overview(
    df: pd.DataFrame,
    team_name: str,
    team_player_name: str,
    compare_player_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
    """
    Always show ALL matches of player 1.
    If compare_player_name is set:
      - Insert player 2 row under player 1 only for matches that player 2 also played.
      - Player 2 rows are gray.
      - Player 2 matches that player 1 didn't play are skipped (per your requirement).
    """
    validate_physical_data(df)
    cols = OverviewColumns()

    p1 = _compute_player_match_metrics(df, team_name, team_player_name)
    if p1.empty:
        raise ValueError(f"No rows found for club={team_name!r} player_name={team_player_name!r}")
    p1 = _sort_player_matches_desc(p1)

    p2_map: dict[object, dict] = {}
    compare_active = bool(compare_player_name) and str(compare_player_name) != str(team_player_name)
    if compare_active:
        p2 = _compute_player_match_metrics(df, team_name, str(compare_player_name))
        if not p2.empty:
            # one row per match_id; keep first if duplicates exist
            p2 = _sort_player_matches_desc(p2).drop_duplicates(subset=["match_id"], keep="first")
            p2_map = {row["match_id"]: row.to_dict() for _, row in p2.iterrows()}

    rows: list[dict] = []
    for _, r1 in p1.iterrows():
        rows.append(
            {
                "_row_type": "primary",
                cols.match: r1["match_label"],
                cols.player: r1["player_name"],
                cols.pos: r1["position"],
                cols.minutes: r1["minutes"],
                cols.total_distance: r1["total_km"],
                cols.m_per_min: r1["m_per_min"],
                cols.runs: r1["runs"],
                cols.sprints: r1["sprints"],
            }
        )

        if compare_active:
            r2 = p2_map.get(r1["match_id"])
            if r2 is not None:
                rows.append(
                    {
                        "_row_type": "compare",
                        cols.match: r2["match_label"],
                        cols.player: r2["player_name"],
                        cols.pos: r2["position"],
                        cols.minutes: r2["minutes"],
                        cols.total_distance: r2["total_km"],
                        cols.m_per_min: r2["m_per_min"],
                        cols.runs: r2["runs"],
                        cols.sprints: r2["sprints"],
                    }
                )

    display_df = pd.DataFrame(rows)

    # ---- styling ----
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

    def _row_bg(row: pd.Series) -> list[str]:
        if row.get("_row_type") == "compare":
            return ["background-color: #f2f2f2;"] * len(row)
        return [""] * len(row)

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

    caption = f"{team_player_name} — {team_name} (per match)"
    if compare_active and p2_map:
        caption = f"{team_player_name} vs {compare_player_name} — {team_name} (player 1 matches, player 2 where available)"
    styler = styler.set_caption(caption)

    return display_df.drop(columns=["_row_type"]), styler


def styler_to_html(styler: "pd.io.formats.style.Styler") -> str:
    html = styler.to_html()
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');
      body {{
        margin: 0;
        font-family: {STREAMLIT_FONT_STACK};
      }}
    </style>
    <div style="width: 100%; overflow: visible; font-family: {STREAMLIT_FONT_STACK};">
      {html}
    </div>
    """


def estimate_table_height_px(n_rows: int) -> int:
    header = 58
    caption = 44
    row_height = 42
    padding = 28
    return header + caption + (n_rows * row_height) + padding
