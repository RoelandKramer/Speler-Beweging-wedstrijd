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
    """
    Load physical data from CSV.

    Args:
        csv_path: Path to physical_data_matches.csv

    Returns:
        DataFrame with match physical data.
    """
    return pd.read_csv(csv_path)


def validate_physical_data(df: pd.DataFrame) -> None:
    """
    Validate required columns exist.

    Raises:
        ValueError if required columns are missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def list_teams(df: pd.DataFrame) -> list[str]:
    """Return sorted unique teams (club)."""
    return sorted(df["club"].dropna().astype(str).unique().tolist())


def list_players_for_team(df: pd.DataFrame, team_name: str) -> list[str]:
    """Return sorted unique players for a given team."""
    players = (
        df.loc[df["club"].astype(str) == str(team_name), "player_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(players)


def _parse_home_away_and_opponent(match_name: str, club: str) -> tuple[str, str]:
    """
    Parse match_name like: 'PSV vs Go Ahead Eagles' to determine home/away for `club`.

    Returns:
        (home_away, opponent) where home_away in {'Home','Away','?'}.
    """
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
    """
    If match_id looks like YYYYMMDD, parse to datetime; otherwise return None.
    """
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


def build_player_match_overview(
    df: pd.DataFrame,
    team_name: str,
    team_player_name: str,
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
    """
    Build a professional per-match physical overview table for a given club and player.

    - Most recent match on top.
    - Bars use exact colors + fixed scales.
    - Total distance displayed in km.

    Returns:
        (display_df, styler)
    """
    validate_physical_data(df)

    filtered = df.loc[
        (df["club"].astype(str) == str(team_name))
        & (df["player_name"].astype(str) == str(team_player_name))
    ].copy()

    if filtered.empty:
        raise ValueError(f"No rows found for club={team_name!r} player_name={team_player_name!r}")

    filtered["_match_date"] = filtered["match_id"].apply(_match_date_from_match_id)

    ha_opp = filtered.apply(
        lambda r: _parse_home_away_and_opponent(str(r["match_name"]), str(r["club"])),
        axis=1,
        result_type="expand",
    )
    filtered["_ha"] = ha_opp[0]
    filtered["_opp"] = ha_opp[1]

    filtered["_minutes"] = pd.to_numeric(filtered["Minutes"], errors="coerce")
    filtered["_total_m"] = pd.to_numeric(filtered["total_distance"], errors="coerce")
    filtered["_runs"] = pd.to_numeric(filtered["hi_runs"], errors="coerce")
    filtered["_sprints"] = pd.to_numeric(filtered["sprint_runs"], errors="coerce")

    if filtered["_minutes"].isna().all() or filtered["_total_m"].isna().all():
        raise ValueError("Minutes/total_distance columns could not be parsed to numeric values.")

    filtered["_m_per_min"] = filtered["_total_m"] / filtered["_minutes"]
    filtered["_total_km"] = filtered["_total_m"] / 1000.0

    cols = OverviewColumns()
    display_df = pd.DataFrame(
        {
            cols.match: filtered.apply(lambda r: f"{r['_ha']} • {r['_opp']}", axis=1),
            cols.player: filtered["player_name"].astype(str),
            cols.pos: filtered["position"].astype(str),
            cols.minutes: filtered["_minutes"],
            cols.total_distance: filtered["_total_km"],
            cols.m_per_min: filtered["_m_per_min"],
            cols.runs: filtered["_runs"],
            cols.sprints: filtered["_sprints"],
        }
    )

    # Most recent first
    if filtered["_match_date"].notna().any():
        sorter = filtered[["_match_date"]].copy()
        sorter["_idx"] = range(len(sorter))
        sorter = sorter.sort_values(["_match_date", "_idx"], ascending=[False, False])
        display_df = display_df.iloc[sorter["_idx"].to_list()].reset_index(drop=True)
    else:
        match_ids = pd.to_numeric(filtered["match_id"], errors="coerce")
        display_df = display_df.iloc[match_ids.sort_values(ascending=False).index].reset_index(drop=True)

    styler = display_df.style.hide(axis="index")

    table_styles = [
        {
            "selector": "th",
            "props": [
                ("font-weight", "700"),
                ("font-size", "12px"),
                ("text-align", "left"),
                ("padding", "8px 10px"),
                ("border-bottom", "2px solid #ddd"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("font-size", "12px"),
                ("padding", "7px 10px"),
                ("border-bottom", "1px solid #eee"),
                ("vertical-align", "middle"),
            ],
        },
        {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "700")]},
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

    for col_name in [cols.total_distance, cols.m_per_min, cols.runs, cols.sprints]:
        vmin, vmax = BAR_SCALES[col_name]
        styler = styler.bar(
            subset=[col_name],
            align="left",
            color=BAR_COLORS[col_name],
            vmin=vmin,
            vmax=vmax,
        )

    styler = styler.set_caption(f"{team_player_name} — {team_name} (per match)")
    return display_df, styler
