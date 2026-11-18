"""Plain-English reasons and risks, built from the metrics with simple templates.

Each candidate sentence carries a strength value; we keep the strongest few.
To add a new reason, append another `add(...)` call with a condition.
"""
from __future__ import annotations

import math

import pandas as pd

from .config import MARKET_HEAVY_TRANSFERS, MARKET_OWN_TREND


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def reasons_for(r: pd.Series, horizon_gws: int, max_items: int = 3) -> list[str]:
    out: list[tuple[float, str]] = []

    def add(strength, text):
        out.append((strength, text))

    pos = r["position"]
    if _ok(r.get("attack_pct")) and r["attack_pct"] >= 75 and pos != "GKP":
        add(r["attack_pct"], f"xG {r['xg90_blend']:.2f} + xA {r['xa90_blend']:.2f} per 90 "
                             f"(blended): #{int(r['attack_rank'])} of {int(r['pool_size'])} "
                             f"{pos}s for attacking threat")
    if r.get("n_fixtures_next", 0) >= 2:
        add(95, "Double gameweek next")
    elif r.get("n_fixtures_horizon", 0) > horizon_gws:
        add(80, f"{int(r['n_fixtures_horizon'])} fixtures in the next {horizon_gws} GWs")
    if pos in ("GKP", "DEF") and _ok(r.get("next_cs_prob")) and r["next_cs_prob"] >= 0.35:
        add(60 + 50 * r["next_cs_prob"], f"~{r['next_cs_prob']:.0%} clean-sheet chance next (model)")
    if _ok(r.get("defcon_rate_blend")) and r["defcon_rate_blend"] >= 0.4:
        add(55 + 40 * r["defcon_rate_blend"],
            f"Hits DefCon threshold in ~{r['defcon_rate_blend']:.0%} of 60-min apps")
    if _ok(r.get("next_att_mult")) and r["next_att_mult"] >= 1.15 and pos in ("MID", "FWD"):
        add(60 + 40 * (r["next_att_mult"] - 1), f"Kind attacking fixture ({r['next_fixture']})")
    if _ok(r.get("start_share")) and r["start_share"] >= 0.85 and _ok(r.get("r_starts")):
        add(50, f"Started {int(r['r_starts'])} of last {int(r['r_team_matches'])} team matches")
    if r.get("penalties_order") == 1:
        add(58, "First-choice penalty taker")
    if _ok(r.get("corners_order")) and r["corners_order"] in (1, 2):
        add(45 + 10 * (2 - r["corners_order"]),
            f"{'First' if r['corners_order'] == 1 else 'Second'}-choice corners")
    if _ok(r.get("frees_order")) and r["frees_order"] in (1, 2):
        add(45 + 10 * (2 - r["frees_order"]),
            f"{'First' if r['frees_order'] == 1 else 'Second'}-choice direct free kicks")
    if _ok(r.get("value_score")) and r["value_score"] >= 80:
        add(52, f"Top-20% value among {pos}s at £{r['price']:.1f}m")
    if pos == "GKP" and _ok(r.get("saves90_blend")) and r["saves90_blend"] >= 3.0:
        add(50, f"{r['saves90_blend']:.1f} saves/90 adds save points")
    out.sort(key=lambda x: -x[0])
    return [t for _, t in out[:max_items]]


def risks_for(r: pd.Series, next_gw, max_items: int = 3) -> list[str]:
    out: list[tuple[float, str]] = []

    def add(strength, text):
        out.append((strength, text))

    if r.get("avail_next", 1) < 1:
        chance = r.get("chance_next")
        flag = f"{int(chance)}%" if _ok(chance) else r.get("status", "?")
        news = (r.get("news") or "").strip()
        add(100, f"Flagged {flag}" + (f": {news}" if news else ""))
    if r.get("n_fixtures_next", 1) == 0:
        add(98, f"Blank in GW{next_gw}")
    if _ok(r.get("season_minutes")) and r["season_minutes"] < 450:
        add(85, f"Small sample: {int(r['season_minutes'])} league minutes")
    if not r.get("history_available", True):
        add(80, "Match history unavailable; recent form not assessed")
    if _ok(r.get("r_early_subs")) and r["r_early_subs"] >= 2:
        add(70, f"Off before 60' in {int(r['r_early_subs'])} of last {int(r['r_starts'])} starts")
    if _ok(r.get("start_share")) and r["start_share"] < 0.6 and _ok(r.get("r_starts")):
        add(75, f"Started only {int(r['r_starts'])} of last {int(r['r_team_matches'])} team matches")
    if r.get("minutes_trend") == "falling":
        add(65, "Minutes falling over the last 3 matches")
    if _ok(r.get("goals_minus_xg")) and r["goals_minus_xg"] >= 2:
        add(60, f"{int(r['season_goals'])} goals from {r['season_xg']:.1f} xG; finishing may cool")
    if r["position"] in ("MID", "FWD") and _ok(r.get("next_att_mult")) and r["next_att_mult"] <= 0.85:
        add(55, f"Tough attacking fixture ({r['next_fixture']})")
    if r["position"] in ("GKP", "DEF") and _ok(r.get("next_cs_prob")) and r["next_cs_prob"] < 0.2:
        add(55, f"Low clean-sheet chance next (~{r['next_cs_prob']:.0%})")
    if r.get("past_transfer_flag"):
        add(65, "Joined a new club; last season's rate may not transfer")
    if r.get("confidence_label") == "Low":
        add(50, "Low data confidence")
    out.sort(key=lambda x: -x[0])
    return [t for _, t in out[:max_items]]


def set_pieces_label(r: pd.Series) -> str:
    """Context only, never scored: squad ranks (1 = first choice), shown for 1-2.

    E.g. "Pens 1, Corners 2". These chances are already inside xG/xA, so this is
    a reading aid, not a component.
    """
    parts = []
    for key, short in [("penalties_order", "Pens"), ("corners_order", "Corners"),
                       ("frees_order", "FKs")]:
        v = r.get(key)
        if _ok(v) and v in (1, 2):
            parts.append(f"{short} {int(v)}")
    return ", ".join(parts)


def market_note(r: pd.Series) -> str:
    """Market context. Kept in a SEPARATE column and never in reasons/risks,
    because what other managers are doing is not football quality and never
    changes the opportunity score.
    """
    parts = []
    net = r.get("net_transfers_event")
    if _ok(net) and abs(net) >= MARKET_HEAVY_TRANSFERS:
        parts.append("Heavy transfers in" if net > 0 else "Heavy transfers out")
    elif _ok(net) and abs(net) >= MARKET_HEAVY_TRANSFERS / 10:
        parts.append("Net transfers in" if net > 0 else "Net transfers out")
    chg = r.get("price_change_event_m")
    if _ok(chg) and chg != 0:
        parts.append(f"{'Rose' if chg > 0 else 'Fell'} £{abs(chg):.1f}m this GW")
    start = r.get("price_since_start_m")
    if _ok(start) and abs(start) >= 0.1:
        parts.append(f"{'Up' if start > 0 else 'Down'} £{abs(start):.1f}m since start")
    trend = r.get("own_trend_3gw")
    if _ok(trend) and abs(trend) >= MARKET_OWN_TREND:
        parts.append("Rising ownership" if trend > 0 else "Falling ownership")
    elif _ok(trend) and trend > 0:
        parts.append("Ownership climbing")
    return "; ".join(parts[:3])


def add_market_notes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["set_pieces"] = [set_pieces_label(r) for _, r in df.iterrows()]
    df["market_note"] = [market_note(r) for _, r in df.iterrows()]
    return df


def add_explanations(df: pd.DataFrame, horizon_gws: int, next_gw) -> pd.DataFrame:
    df = df.copy()
    df["reasons"] = ["; ".join(reasons_for(r, horizon_gws)) for _, r in df.iterrows()]
    df["risks"] = ["; ".join(risks_for(r, next_gw)) for _, r in df.iterrows()]
    return df
