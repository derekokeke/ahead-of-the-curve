"""Turn raw FPL JSON into tidy pandas tables.

Rules followed throughout:
  * Numbers that arrive as strings ("0.57") are converted with errors="coerce".
  * A field missing from the API becomes NaN (unavailable), never 0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FALLBACK_SCORING, PAST_XG_FIRST_SEASON, POSITIONS


@dataclass
class Tables:
    players: pd.DataFrame
    teams: pd.DataFrame
    events: pd.DataFrame
    fixtures: pd.DataFrame
    history: pd.DataFrame
    history_past: pd.DataFrame
    scoring: dict
    season: str
    next_gw: int | None
    current_gw: int | None
    last_finished_gw: int | None
    meta: dict = field(default_factory=dict)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _str(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


# ------------------------------------------------------------- gameweeks ------
def detect_gameweeks(events: pd.DataFrame) -> tuple[int | None, int | None, int | None]:
    """Return (next_gw, current_gw, last_finished_gw) from the events table.

    Prefers the API's own is_next / is_current flags; falls back to the first
    unfinished gameweek if flags are missing (e.g. between seasons).
    """
    if events.empty:
        return None, None, None
    next_rows = events[events["is_next"]]
    if not next_rows.empty:
        next_gw = int(next_rows["event"].iloc[0])
    else:
        unfinished = events[~events["finished"]]
        next_gw = int(unfinished["event"].min()) if not unfinished.empty else None
    cur_rows = events[events["is_current"]]
    current_gw = int(cur_rows["event"].iloc[0]) if not cur_rows.empty else None
    fin = events[events["finished"]]
    last_finished = int(fin["event"].max()) if not fin.empty else None
    return next_gw, current_gw, last_finished


def detect_season(bootstrap: dict, events: pd.DataFrame) -> str:
    url = (bootstrap.get("game_config", {}).get("settings", {}) or {}).get("static_content_url", "")
    m = re.search(r"(20\d\d)_(\d\d)", url or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    if not events.empty and events["deadline_time"].notna().any():
        first = events["deadline_time"].min()
        start = first.year if first.month >= 7 else first.year - 1
        return f"{start}/{(start + 1) % 100:02d}"
    return "unknown season"


def scoring_rules(bootstrap: dict) -> dict:
    """Read points values from the live API, falling back field by field."""
    live = (bootstrap.get("game_config") or {}).get("scoring") or {}
    out = {}
    for key, fallback in FALLBACK_SCORING.items():
        val = live.get(key, fallback)
        out[key] = val
    out["_source"] = "live API game_config.scoring" if live else "fallback constants (API field missing)"
    return out


# --------------------------------------------------------------- builders -----
def build_tables(raw: dict) -> Tables:
    bs = raw.get("bootstrap") or {}

    teams_raw = pd.DataFrame(bs.get("teams", []))
    teams = pd.DataFrame({
        "team_id": teams_raw.get("id", pd.Series(dtype=int)).astype(int),
        "team": _str(teams_raw, "short_name"),
        "team_name": _str(teams_raw, "name"),
    })

    ev = pd.DataFrame(bs.get("events", []))
    events = pd.DataFrame({
        "event": _num(ev, "id").astype("Int64"),
        "name": _str(ev, "name"),
        "deadline_time": pd.to_datetime(ev.get("deadline_time"), utc=True, errors="coerce")
        if "deadline_time" in ev else pd.NaT,
        "finished": _bool(ev, "finished"),
        "is_current": _bool(ev, "is_current"),
        "is_next": _bool(ev, "is_next"),
    })

    el = pd.DataFrame(bs.get("elements", []))
    players = pd.DataFrame({
        "player_id": _num(el, "id").astype(int),
        "name": _str(el, "web_name"),
        "full_name": (_str(el, "first_name") + " " + _str(el, "second_name")).str.strip(),
        "team_id": _num(el, "team").astype("Int64"),
        "position": _num(el, "element_type").map(POSITIONS),
        "price": _num(el, "now_cost") / 10.0,
        "ownership": _num(el, "selected_by_percent"),
        "status": _str(el, "status", "a"),
        "chance_next": _num(el, "chance_of_playing_next_round"),
        "news": _str(el, "news"),
        "season_minutes": _num(el, "minutes"),
        "season_starts": _num(el, "starts"),
        "season_goals": _num(el, "goals_scored"),
        "season_assists": _num(el, "assists"),
        "season_xg": _num(el, "expected_goals"),
        "season_xa": _num(el, "expected_assists"),
        "season_xgi": _num(el, "expected_goal_involvements"),
        "season_xgc": _num(el, "expected_goals_conceded"),
        "season_saves": _num(el, "saves"),
        "season_points": _num(el, "total_points"),
        "season_defcon_actions": _num(el, "defensive_contribution"),
        "penalties_order": _num(el, "penalties_order"),
        # Set-piece squad ranks (1 = first choice). Present-but-empty *_text
        # variants exist in the API; the numeric fields are the usable ones.
        "corners_order": _num(el, "corners_and_indirect_freekicks_order"),
        "frees_order": _num(el, "direct_freekicks_order"),
        "team_join_date": pd.to_datetime(el.get("team_join_date"), errors="coerce"),
        # Market context (0.1m price units converted later). Never used in scores.
        "transfers_in_event": _num(el, "transfers_in_event"),
        "transfers_out_event": _num(el, "transfers_out_event"),
        "cost_change_event": _num(el, "cost_change_event"),
        "cost_change_start": _num(el, "cost_change_start"),
    })
    players = players.merge(teams[["team_id", "team"]], on="team_id", how="left")

    fx = pd.DataFrame(raw.get("fixtures") or [])
    fixtures = pd.DataFrame({
        "fixture_id": _num(fx, "id").astype("Int64"),
        "event": _num(fx, "event"),  # NaN = postponed / not yet scheduled
        "kickoff_time": pd.to_datetime(fx.get("kickoff_time"), utc=True, errors="coerce")
        if "kickoff_time" in fx else pd.NaT,
        "team_h": _num(fx, "team_h").astype("Int64"),
        "team_a": _num(fx, "team_a").astype("Int64"),
        "team_h_score": _num(fx, "team_h_score"),
        "team_a_score": _num(fx, "team_a_score"),
        "finished": _bool(fx, "finished"),
        "finished_provisional": _bool(fx, "finished_provisional"),
        "team_h_difficulty": _num(fx, "team_h_difficulty"),
        "team_a_difficulty": _num(fx, "team_a_difficulty"),
    })
    # A match counts as played once FPL marks it provisionally finished; final
    # bonus/data checks can still adjust totals afterwards.
    fixtures["completed"] = fixtures["finished"] | fixtures["finished_provisional"]

    history = _build_history(raw.get("histories") or {}, fixtures)

    # --- Market context: derived from the fields above, never used in scores ---
    players["net_transfers_event"] = players["transfers_in_event"] - players["transfers_out_event"]
    players["price_change_event_m"] = players["cost_change_event"] / 10.0
    players["price_since_start_m"] = players["cost_change_start"] / 10.0
    players["total_players"] = float(bs.get("total_players")) if bs.get("total_players") else np.nan

    # Per-match history fields: latest value + BPS season total + ownership trend
    # over the last 3 FINISHED rounds (deduped by round: a DGW gives two rows in
    # the same round with the same `selected`). Missing history stays NaN.
    market_cols = ["bps_total", "current_value_m", "net_transfers_balance",
                   "selected_latest", "own_trend_3gw", "own_trend_pct"]
    if not history.empty and "selected" in history.columns:
        hsel = history[["player_id", "round", "selected", "bps", "value", "transfers_balance"]]
        latest = hsel.sort_values("round").groupby("player_id").tail(1).set_index("player_id")
        players["bps_total"] = players["player_id"].map(
            hsel.groupby("player_id")["bps"].sum(min_count=1))
        players["current_value_m"] = players["player_id"].map(latest["value"] / 10.0)
        players["net_transfers_balance"] = players["player_id"].map(latest["transfers_balance"])
        players["selected_latest"] = players["player_id"].map(latest["selected"])
        rounds3 = [float(r) for r in
                   sorted(events.loc[events["finished"], "event"].dropna().astype(int))[-3:]]
        w = (hsel[hsel["round"].isin(rounds3)]
             .groupby(["player_id", "round"])["selected"].max().reset_index())
        if not w.empty:
            sf = w[w["round"] == w["round"].min()].set_index("player_id")["selected"]
            sl = w[w["round"] == w["round"].max()].set_index("player_id")["selected"]
            players["own_trend_3gw"] = players["player_id"].map(sl - sf)
        else:
            players["own_trend_3gw"] = np.nan
        # Approximate % of ALL FPL managers (total_players) — labelled approximate
        # in the UI because FPL's official ownership % is computed differently.
        players["own_trend_pct"] = players["own_trend_3gw"] / players["total_players"] * 100.0
    else:
        for c in market_cols:
            players[c] = np.nan

    history_past = build_history_past(raw.get("histories_past") or {})


    available_ids = {int(k) for k in (raw.get("histories") or {}).keys()}
    # History is "available" if we fetched it, or if the player has no minutes
    # (nothing to fetch: an empty history is then the truth, not a gap).
    players["history_available"] = (
        players["player_id"].isin(available_ids) | (players["season_minutes"].fillna(0) <= 0)
    )

    next_gw, current_gw, last_finished = detect_gameweeks(events)
    return Tables(
        players=players,
        teams=teams,
        events=events,
        fixtures=fixtures,
        history=history,
        history_past=history_past,
        scoring=scoring_rules(bs),
        season=detect_season(bs, events),
        next_gw=next_gw,
        current_gw=current_gw,
        last_finished_gw=last_finished,
        meta=dict(raw.get("meta") or {}),
    )


# ----------------------------------------------------------- history past -----
PAST_SEASON_STATS = ["minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "saves",
                     "bonus", "bps", "total_points"]
PAST_COST_COLS = ["start_cost_m", "end_cost_m"]


def _season_year(name) -> float:
    """'2025/26' -> 2025. NaN when the season name is missing or malformed."""
    m = re.match(r"(\d{4})", str(name or ""))
    return float(m.group(1)) if m else np.nan


def build_history_past(histories_past: dict) -> pd.DataFrame:
    """One row per player per past PL season, newest first.

    Fields the API only introduced in later seasons are NaN for older seasons,
    decided by SEASON (PAST_XG_FIRST_SEASON), never by zero-value sniffing:
    expected_* and starts keys exist in every history_past row but are zero
    placeholders before 2022/23 (verified live 16 Nov 2026: Raya's 2021/22 row
    has 2,160 minutes with starts=0 and xGC="0.00"), while a goalkeeper can
    legitimately have 0 xG in a season where the field is real.
    Seasons with 0 minutes (played away from the PL) carry no usable stats;
    their start/end cost is still known.
    """
    cols = (["player_id", "season", "season_year"] + PAST_SEASON_STATS + PAST_COST_COLS)
    rows = []
    for pid, seasons in (histories_past or {}).items():
        for r in seasons or []:
            rows.append({
                "player_id": int(pid),
                "season": r.get("season_name"),
                "season_year": _season_year(r.get("season_name")),
                "minutes": r.get("minutes"), "starts": r.get("starts"),
                "goals": r.get("goals_scored"), "assists": r.get("assists"),
                "xg": r.get("expected_goals"), "xa": r.get("expected_assists"),
                "xgc": r.get("expected_goals_conceded"), "saves": r.get("saves"),
                "bonus": r.get("bonus"), "bps": r.get("bps"),
                "start_cost_m": r.get("start_cost"), "end_cost_m": r.get("end_cost"),
                "total_points": r.get("total_points"),
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    for c in cols[3:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[PAST_COST_COLS] = out[PAST_COST_COLS] / 10.0  # API stores 0.1m units

    early = out["season_year"] < float(str(PAST_XG_FIRST_SEASON)[:4])
    out.loc[early, ["xg", "xa", "xgc", "starts"]] = np.nan
    played = out["minutes"].fillna(0) > 0
    out.loc[~played, PAST_SEASON_STATS] = np.nan
    return (out.sort_values(["player_id", "season_year"], ascending=[True, False])
            .reset_index(drop=True))


HISTORY_NUMERIC = [
    "fixture", "opponent_team", "round", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "saves", "bonus", "total_points",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution",
]


def _build_history(histories: dict, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, hist in histories.items():
        for r in hist or []:
            rows.append({**r, "player_id": int(pid)})
    if not rows:
        cols = ["player_id", "fixture_id", "team_id", "was_home"] + HISTORY_NUMERIC
        return pd.DataFrame(columns=cols)

    h = pd.DataFrame(rows)
    out = pd.DataFrame({"player_id": h["player_id"].astype(int)})
    for col in HISTORY_NUMERIC:
        # Keep a column only if the API actually sent it; absent stays absent.
        if col in h.columns:
            out[col] = pd.to_numeric(h[col], errors="coerce")
    out["was_home"] = _bool(h, "was_home")
    out["fixture_id"] = out["fixture"].astype("Int64")

    # Which team was the player on in that fixture? Derived from the fixture itself,
    # so it stays correct for players who changed clubs mid-season.
    fx = fixtures[["fixture_id", "team_h", "team_a"]]
    out = out.merge(fx, on="fixture_id", how="left")
    out["team_id"] = np.where(out["was_home"], out["team_h"], out["team_a"])
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    return out.drop(columns=["team_h", "team_a"])
