# =========================
# file: load_table.py
# =========================

from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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
    - Show ALL matches for player 1
    - If compare selected: show ALL matches for player 2 too
    - Sort most recent first; if same match -> player 1 first
    - Player 2 rows light blue; player 1 rows white
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

    combined_src["_sort_date"] = combined_src["_match_date"].fillna(pd.Timestamp.min)
    combined_src = combined_src.sort_values(
        by=["_sort_date", "_match_id_num", "match_id", "_player_rank"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

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

    return display_df, styler


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


def _hex_to_rgb(hex_color: str) -> tuple:
    c = hex_color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _load_font(size: int, bold: bool = False):
    try:
        if bold:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _a4_landscape_px(dpi: int) -> tuple:
    # A4: 11.69 x 8.27 inches (landscape)
    return int(round(11.69 * dpi)), int(round(8.27 * dpi))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def table_to_png_bytes(
    display_df: pd.DataFrame,
    *,
    title: str,
    caption: str,
    dpi: int = 200,
) -> bytes:
    """
    Render the table to A4 landscape PNG (auto-scaled), including title + caption.
    """
    cols = OverviewColumns()
    df = display_df.copy()
    if "_row_type" not in df.columns:
        df["_row_type"] = "primary"

    show_cols = [
        cols.match,
        cols.player,
        cols.pos,
        cols.minutes,
        cols.total_distance,
        cols.m_per_min,
        cols.runs,
        cols.sprints,
    ]

    # Base layout (scaled)
    base_pad_x = 14
    base_row_h = 52
    base_header_h = 56
    base_bar_w = 220
    base_bar_h = 28

    base_col_widths = {
        cols.match: 260,
        cols.player: 180,
        cols.pos: 70,
        cols.minutes: 70,
        cols.total_distance: 320,
        cols.m_per_min: 260,
        cols.runs: 220,
        cols.sprints: 220,
    }

    base_table_w = sum(base_col_widths[c] for c in show_cols)
    base_table_h = base_header_h + (len(df) * base_row_h)

    page_w, page_h = _a4_landscape_px(dpi=dpi)
    margin = int(round(0.35 * dpi))

    base_title_h = 56
    base_caption_h = 34
    base_top_block_h = base_title_h + base_caption_h + 12

    avail_w = page_w - (2 * margin)
    avail_h = page_h - (2 * margin) - base_top_block_h

    scale = _clamp(min(avail_w / float(base_table_w), avail_h / float(base_table_h)), 0.45, 1.35)

    pad_x = int(round(base_pad_x * scale))
    row_h = int(round(base_row_h * scale))
    header_h = int(round(base_header_h * scale))
    bar_w = int(round(base_bar_w * scale))
    bar_h = int(round(base_bar_h * scale))
    title_h = int(round(base_title_h * scale))
    caption_h = int(round(base_caption_h * scale))

    col_widths = {k: int(round(v * scale)) for k, v in base_col_widths.items()}
    table_w = sum(col_widths[c] for c in show_cols)

    img = Image.new("RGB", (page_w, page_h), _hex_to_rgb("#FFFFFF"))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(max(18, int(round(34 * scale))), bold=True)
    caption_font = _load_font(max(14, int(round(18 * scale))), bold=False)
    header_font = _load_font(max(14, int(round(18 * scale))), bold=True)
    cell_font = _load_font(max(12, int(round(16 * scale))), bold=False)

    y = margin
    tw = draw.textlength(title, font=title_font)
    draw.text(((page_w - tw) / 2.0, y), title, font=title_font, fill=_hex_to_rgb("#111827"))

    y += title_h
    cw = draw.textlength(caption, font=caption_font)
    draw.text(((page_w - cw) / 2.0, y), caption, font=caption_font, fill=_hex_to_rgb("#374151"))

    y += caption_h + int(round(14 * scale))

    table_x0 = margin + max(0, (avail_w - table_w) // 2)
    table_y0 = y

    # header
    draw.rectangle([table_x0, table_y0, table_x0 + table_w, table_y0 + header_h], fill=_hex_to_rgb("#FFFFFF"))
    draw.line(
        [table_x0, table_y0 + header_h, table_x0 + table_w, table_y0 + header_h],
        fill=_hex_to_rgb("#DDDDDD"),
        width=max(2, int(round(3 * scale))),
    )

    x = table_x0
    for c in show_cols:
        draw.text((x + pad_x, table_y0 + int(round((header_h - 18 * scale) / 2))), c, font=header_font, fill=_hex_to_rgb("#111827"))
        x += col_widths[c]

    # rows
    for i, row in df.iterrows():
        y0 = table_y0 + header_h + i * row_h
        y1 = y0 + row_h

        bg = ROW_BG_PLAYER_2 if row.get("_row_type") == "compare" else ROW_BG_PLAYER_1
        draw.rectangle([table_x0, y0, table_x0 + table_w, y1], fill=_hex_to_rgb(bg))
        draw.line([table_x0, y1, table_x0 + table_w, y1], fill=_hex_to_rgb("#EEEEEE"), width=max(1, int(round(2 * scale))))

        x = table_x0
        for c in show_cols:
            cell_x0 = x
            cell_x1 = x + col_widths[c]

            if c in [cols.total_distance, cols.m_per_min, cols.runs, cols.sprints]:
                v = row.get(c)
                try:
                    v_float = float(v)
                except Exception:
                    v_float = 0.0

                vmin, vmax = BAR_SCALES[c]
                ratio = 0.0 if vmax <= vmin else (v_float - vmin) / float(vmax - vmin)
                ratio = _clamp(ratio, 0.0, 1.0)

                bx0 = cell_x0 + pad_x
                by0 = y0 + (row_h - bar_h) // 2
                bx1 = bx0 + int(bar_w * ratio)
                by1 = by0 + bar_h
                draw.rectangle([bx0, by0, bx1, by1], fill=_hex_to_rgb(BAR_COLORS[c]))

                txt = "{:.3f}".format(v_float) if (c == cols.total_distance and pd.notna(v)) else ("{:.0f}".format(v_float) if pd.notna(v) else "—")
                text_w = draw.textlength(txt, font=cell_font)
                draw.text((cell_x1 - pad_x - text_w, y0 + int(round((row_h - 16 * scale) / 2))), txt, font=cell_font, fill=_hex_to_rgb("#111827"))
            else:
                val = row.get(c)
                if pd.isna(val):
                    txt = "—"
                elif c == cols.minutes:
                    try:
                        txt = "{:.0f}".format(float(val))
                    except Exception:
                        txt = str(val)
                else:
                    txt = str(val)

                draw.text((cell_x0 + pad_x, y0 + int(round((row_h - 16 * scale) / 2))), txt, font=cell_font, fill=_hex_to_rgb("#111827"))

            x += col_widths[c]

    out = BytesIO()
    img.save(out, format="PNG", optimize=True, dpi=(dpi, dpi))
    return out.getvalue()
