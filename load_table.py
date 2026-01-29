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

ROW_BG_PLAYER_2 = "#E8F2FF"  # light blue


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


def build_player_match_overview(
    df: pd.DataFrame,
    team_name: str,
    team_player_name: str,
    compare_player_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, "pd.io.formats.style.Styler"]:
    """
    - Always show ALL matches for player 1.
    - If compare player provided: show ALL matches for player 2 as well.
    - Sort by match (most recent first).
    - If same match_id: player 1 row first.
    - Player 2 rows are light blue; Player 1 rows are white.
    """
    validate_physical_data(df)
    cols = OverviewColumns()

    p1 = _compute_player_match_metrics(df, team_name, team_player_name)
    if p1.empty:
        raise ValueError(f"No rows found for club={team_name!r} player_name={team_player_name!r}")

    compare_active = bool(compare_player_name) and str(compare_player_name) != str(team_player_name)
    p2 = pd.DataFrame()
    if compare_active:
        p2 = _compute_player_match_metrics(df, team_name, str(compare_player_name))
        # it's allowed to be empty; we still show full p1

    def _to_display(frame: pd.DataFrame, row_type: str, player_rank: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        return pd.DataFrame(
            {
                "_row_type": row_type,
                "_player_rank": player_rank,  # 0 -> player1, 1 -> player2
                "match_id": frame["match_id"],
                "_match_date": frame["_match_date"],
                "_match_id_num": pd.to_numeric(frame["match_id"], errors="coerce"),
                cols.match: frame["match_label"],
                cols.player: frame["player_name"],
                cols.pos: frame["position"],
                cols.minutes: frame["minutes"],
                cols.total_distance: frame["total_km"],
                cols.m_per_min: frame["m_per_min"],
                cols.runs: frame["runs"],
                cols.sprints: frame["sprints"],
            }
        )

    d1 = _to_display(p1, "primary", 0)
    d2 = _to_display(p2, "compare", 1) if compare_active else pd.DataFrame()

    combined = pd.concat([d1, d2], ignore_index=True)

    # Sort key: use parsed date when available, else push to bottom using Timestamp.min,
    # and sort within that by match_id numeric desc.
    combined["_sort_date"] = combined["_match_date"].fillna(pd.Timestamp.min)

    combined = combined.sort_values(
        by=["_sort_date", "_match_id_num", "match_id", "_player_rank"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Keep for styling, then drop helper columns from returned df
    styler_df = combined.copy()

    # ---- styling ----
    styler = styler_df.style.hide(axis="index")
    styler = styler.hide(
        axis="columns",
        subset=["_row_type", "_player_rank", "match_id", "_match_date",_]()
