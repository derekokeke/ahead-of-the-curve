"""Ahead of the Curve: a personal FPL scouting dashboard.

Run:  streamlit run app.py
Demo (fictional data, no network):  AOTC_DEMO=1 streamlit run app.py
Offline (last good snapshot only):  AOTC_OFFLINE=1 streamlit run app.py

This file only does layout. Data fetching lives in aotc/fpl_data.py,
the maths in aotc/metrics.py, and the sentences in aotc/explain.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aotc import fpl_data
from aotc.config import (AVAILABILITY_HAIRCUT, DEFCON_SOURCE, DEFCON_THRESHOLD,
                         LATER_GW_FACTOR_IF_OUT, POSITION_ORDER, Settings)
from aotc.demo_data import make_demo_raw
from aotc.explain import add_explanations, add_market_notes
from aotc.metrics import compute, player_match_log, team_match_table
from aotc.transform import build_tables

st.set_page_config(page_title="Ahead of the Curve", page_icon="⚽", layout="wide")

DEMO = os.environ.get("AOTC_DEMO") == "1"
OFFLINE = os.environ.get("AOTC_OFFLINE") == "1"
POS_COLOURS = {"GKP": "#A88A2E", "DEF": "#3D6A8A", "MID": "#4F7F5A", "FWD": "#9A4A3C"}
HIGHLIGHT = "#C99A1A"

st.markdown("""
<style>
.aotc-hero {display:flex; align-items:flex-end; gap:1.25rem; margin:0.25rem 0 0.75rem 0;}
.aotc-gw {font-size:4.2rem; line-height:0.9; font-weight:800; letter-spacing:-0.04em;
          color:#1F5C45; font-variant-numeric: tabular-nums;}
.aotc-meta {padding-bottom:0.35rem;}
.aotc-title {font-size:1.05rem; font-weight:700; color:#1E2A30;}
.aotc-sub {font-size:0.92rem; color:#4A5A62;}
</style>
""", unsafe_allow_html=True)


# ============================================================== data ======
@st.cache_data(ttl=1800, show_spinner=False)
def load_tables(refresh_token: int, force: bool):
    raw = make_demo_raw() if DEMO else fpl_data.load_data(force_refresh=force, offline=OFFLINE)
    return build_tables(raw)


@st.cache_data(show_spinner=False, max_entries=20)
def run_model(_tables, refresh_token: int, settings: dict):
    s = Settings(**settings)
    res = compute(_tables, s)
    players = add_explanations(res.players, s.horizon_gws, res.next_gw)
    players = add_market_notes(players)  # set pieces + market note: context only
    return res, players


if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0
    st.session_state.force = False

# ============================================================ sidebar =====
D = Settings()
with st.sidebar:
    st.markdown("### Ahead of the Curve")
    if st.button("Refresh data", help="Re-fetch from the FPL API, ignoring the local cache"):
        st.session_state.refresh_token += 1
        st.session_state.force = True

    st.markdown("**Look-ahead and form**")
    horizon = st.slider("Gameweeks to look ahead", 3, 5, D.horizon_gws,
                        help="The 'next N GWs' ranking. The next-GW ranking is always shown too.")
    recent = st.radio("Recent form window (team matches)", [3, 5, 8],
                      index=[3, 5, 8].index(D.recent_matches), horizontal=True,
                      help="Counted in your team's completed league matches, including ones "
                           "the player missed.")
    min_minutes = st.number_input("Minimum league minutes for a score", 0, 3000,
                                  D.min_season_minutes, step=90)
    own_thr = st.slider("Differential ownership below (%)", 1.0, 30.0, D.ownership_threshold, 0.5)

    with st.expander("Score weights"):
        w_att = st.slider("Attack (xG/xA)", 0.0, 2.0, D.w_attack, 0.1)
        w_cs = st.slider("Clean sheet", 0.0, 2.0, D.w_clean_sheet, 0.1)
        w_dc = st.slider("Defensive contributions", 0.0, 2.0, D.w_defcon, 0.1)
        w_sv = st.slider("Goalkeeper saves", 0.0, 2.0, D.w_saves, 0.1)
        fx_str = st.slider("Fixture influence", 0.0, 1.5, D.fixture_strength, 0.1,
                           help="0 ignores opponents entirely; 1 uses the model as estimated.")
        rec_w = st.slider("Recent vs season weight", 0.0, 1.0, D.recent_weight, 0.05,
                          help="0 = season only; 1 = recent window only (after shrinkage).")
        prior = st.slider("Shrinkage strength (phantom minutes)", 0, 900,
                          int(D.prior_minutes_season), 90,
                          help="How many minutes of position-average output are blended "
                               "into every player's season rate.")
        w_bn = st.slider("Bonus", 0.0, 2.0, D.w_bonus, 0.1,
                         help="Bonus partly overlaps attacking returns, so the default "
                              "weight (0.5) is below the other components.")
        use_past = st.toggle("Use last season as prior", value=D.use_past_seasons,
                             help="Shrink season rates toward the player's own past "
                                  "season (within the last 2), instead of the position "
                                  "baseline. Applies to xG, xA, saves and xGC.")
        decay = st.slider("Past-season decay", 0.0, 1.0, D.past_season_decay, 0.05,
                          help="Weight of last season's minutes as a prior; the season "
                               "before that is squared. 0.5 = half-trust last season.")

settings = Settings(
    recent_matches=recent, horizon_gws=horizon, min_season_minutes=int(min_minutes),
    ownership_threshold=own_thr, w_attack=w_att, w_clean_sheet=w_cs, w_defcon=w_dc,
    w_saves=w_sv, w_bonus=w_bn, fixture_strength=fx_str, recent_weight=rec_w,
    prior_minutes_season=float(prior), use_past_seasons=use_past,
    past_season_decay=float(decay),
)

try:
    with st.spinner("Loading FPL data (the first run fetches ~450 player histories, "
                    "about a minute; later runs use the local cache)..."):
        tables = load_tables(st.session_state.refresh_token, st.session_state.force)
except fpl_data.DataUnavailable as exc:
    st.error(str(exc))
    st.stop()
st.session_state.force = False

if tables.next_gw is None:
    st.info("No upcoming gameweek found: the season looks finished. Rankings need a next "
            "gameweek to look ahead to.")
    st.stop()

res, P = run_model(tables, st.session_state.refresh_token, settings.to_dict())
meta = tables.meta
nxt = res.next_gw
gw_range = f"GW{res.gws[0]}–{res.gws[-1]}" if res.gws else ""

with st.sidebar:
    st.divider()
    refreshed = meta.get("refreshed_at", "unknown")
    try:
        refreshed = datetime.fromisoformat(refreshed).strftime("%a %d %b %Y, %H:%M UTC")
    except (TypeError, ValueError):
        pass
    st.caption(f"Last successful refresh: {refreshed}\n\nSource: {meta.get('source', '?')}")
    loaded = meta.get("histories_expected", 0) - len(meta.get("histories_missing", []))
    st.caption(f"Player histories loaded: {loaded} of {meta.get('histories_expected', 0)}")

# =============================================================== hero =====
ev = tables.events.set_index("event")
deadline = ev.loc[nxt, "deadline_time"] if nxt in ev.index else pd.NaT
if pd.notna(deadline):
    days = (deadline - datetime.now(timezone.utc)).total_seconds() / 86400
    when = f"in {days:.0f} days" if days >= 1.5 else ("tomorrow" if days >= 0.5 else "very soon")
    dl_text = f"Deadline {deadline.strftime('%A %d %B, %H:%M')} UTC, {when}"
else:
    dl_text = "Deadline not published"
st.markdown(f"""
<div class="aotc-hero">
  <div class="aotc-gw">GW{nxt}</div>
  <div class="aotc-meta">
    <div class="aotc-title">{dl_text}</div>
    <div class="aotc-sub">{tables.season} season. Heuristic rankings to support your
    decisions, not predictions of who will score most.</div>
  </div>
</div>""", unsafe_allow_html=True)

if DEMO:
    st.info("Demo mode: every team and player shown is fictional. Unset AOTC_DEMO to use live FPL data.")
for w in meta.get("warnings", []):
    if not DEMO:
        st.warning(w)
if res.strength.source.startswith("FPL FDR"):
    st.warning("Not enough completed matches to build the xG team-strength model, so fixtures "
               "use FPL's difficulty ratings (FDR) instead. Treat fixture effects as rough.")


# ============================================================ helpers =====
def fmt_minutes(df: pd.DataFrame) -> pd.Series:
    return df["minutes_label"] + " (" + df["xmins"].round(0).astype("Int64").astype(str) + "')"


BASE_CONFIG = {
    "name": st.column_config.TextColumn("Player"),
    "team": st.column_config.TextColumn("Team", width="small"),
    "position": st.column_config.TextColumn("Pos", width="small"),
    "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
    "ownership": st.column_config.NumberColumn("Owned", format="%.1f%%"),
    "score": st.column_config.ProgressColumn("Opportunity", min_value=0, max_value=100,
                                             format="%d", help="0–100 heuristic, comparable "
                                             "within a position. Not expected points."),
    "value_score": st.column_config.ProgressColumn("Value", min_value=0, max_value=100,
                                                   format="%d", help="Opportunity per £m, "
                                                   "percentile within position."),
    "minutes": st.column_config.TextColumn("Minutes outlook"),
    "confidence_label": st.column_config.TextColumn("Data confidence", width="small"),
    "reasons": st.column_config.TextColumn("Why", width="large"),
    "risks": st.column_config.TextColumn("Risks", width="large"),
    "fixtures": st.column_config.TextColumn("Fixtures"),
}


def show_table(df: pd.DataFrame, cols: list[str], extra_config: dict | None = None,
               height: int | str = "auto"):
    cfg = {**BASE_CONFIG, **(extra_config or {})}
    st.dataframe(df[cols], hide_index=True, width="stretch", column_config=cfg, height=height)


def download(df: pd.DataFrame, name: str, key: str):
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"aotc_{name}_gw{nxt}.csv", mime="text/csv", key=key)


def ranking_frame(score_col: str, fixture_col: str) -> pd.DataFrame:
    d = P[P["eligible"]].copy()
    d["score"] = d[score_col]
    d["fixtures"] = d[fixture_col]
    d["minutes"] = fmt_minutes(d)
    return d


EXPORT_COLS = ["name", "team", "position", "price", "ownership", "next_fixture",
               "horizon_fixtures", "score_next", "score_horizon", "value_score", "raw_next",
               "raw_horizon", "xmins", "minutes_label", "start_share", "season_minutes",
               "r_minutes", "xg90_blend", "xa90_blend", "xgi90_blend", "defcon_rate_blend",
               "next_cs_prob", "set_pieces", "market_note", "bps90",
                "confidence", "confidence_label", "reasons", "risks"]

tabs = st.tabs(["This week", "Differential finder", "Form vs underlying",
                "Player comparison", "Market watch", "Methodology"])

# ========================================================= This week =====
with tabs[0]:
    mode = st.radio("Rank by", [f"Next gameweek (GW{nxt})", f"Next {horizon} gameweeks ({gw_range})"],
                    horizontal=True)
    use_next = mode.startswith("Next gameweek")
    score_col = "score_next" if use_next else "score_horizon"
    raw_col = "raw_next" if use_next else "raw_horizon"
    fixture_col = "next_fixture" if use_next else "horizon_fixtures"
    R = ranking_frame(score_col, fixture_col)

    st.markdown("#### Captaincy shortlist")
    st.caption(f"Ranked across positions by the raw GW{nxt} opportunity value (rough "
               "points-equivalent units from xG, xA, clean-sheet chance, DefCon and saves). "
               "Useful for ordering, not a points forecast.")
    cap = R[(R["xmins"] >= 60)].nlargest(5, "raw_next").copy()
    cap["fixtures"] = cap["next_fixture"]
    show_table(cap, ["name", "team", "position", "fixtures", "raw_next", "minutes",
                     "confidence_label", "reasons", "risks"],
               {"raw_next": st.column_config.NumberColumn("Opportunity (raw)", format="%.2f")})

    st.markdown("#### Top picks by position")
    top_n = st.slider("Players per position", 5, 25, 10, key="topn")
    ptabs = st.tabs(POSITION_ORDER)
    for pos, ptab in zip(POSITION_ORDER, ptabs):
        with ptab:
            d = R[R["position"] == pos].nlargest(top_n, raw_col)
            show_table(d, ["name", "team", "price", "ownership", "fixtures", "score",
                           "value_score", "minutes", "set_pieces", "market_note",
                           "confidence_label", "reasons", "risks"])
            sfx = "_next" if use_next else "_horizon"
            comp = d[["name"] + [f"c_{c}{sfx}" for c in
                                 ["attack", "clean_sheet", "defcon", "saves", "bonus"]]].copy()
            comp.columns = ["name", "Attack (xG/xA)", "Clean sheet", "DefCon", "Saves", "Bonus"]
            long = comp.melt("name", var_name="Component", value_name="Value")
            long = long[long["Value"] > 0]
            fig = px.bar(long, y="name", x="Value", color="Component", orientation="h",
                         color_discrete_sequence=["#1F5C45", "#3D6A8A", "#7E9C8A",
                                                  "#A88A2E", "#9A4A3C"],
                         category_orders={"name": d["name"].tolist()})
            fig.update_layout(height=60 + 28 * len(d), margin=dict(l=0, r=0, t=30, b=0),
                              title=dict(text="What drives each ranking (before weights)",
                                         font=dict(size=14)),
                              xaxis_title="Points-equivalent units", yaxis_title=None,
                              legend=dict(orientation="h", y=-0.15), plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch", key=f"comp_{pos}")
    download(R.sort_values(score_col, ascending=False)[EXPORT_COLS], "rankings", "dl_rank")

# ================================================= Differential finder =====
with tabs[1]:
    st.caption("Low ownership is a filter, not a bonus: it never raises a player's football score.")
    c1, c2, c3 = st.columns(3)
    pos_sel = c1.multiselect("Positions", POSITION_ORDER, POSITION_ORDER)
    pmin, pmax = float(P["price"].min()), float(P["price"].max())
    max_price = c2.slider("Max price (£m)", pmin, pmax, pmax, 0.1)
    d_own = c3.slider("Ownership below (%)", 1.0, 30.0, own_thr, 0.5, key="d_own")
    c4, c5, c6 = st.columns(3)
    d_min = c4.number_input("Min league minutes", 0, 3000, int(min_minutes), step=90, key="d_min")
    d_score = c5.slider(f"Min opportunity ({horizon} GWs)", 0, 100,
                        int(settings.differential_min_score), 5)
    d_xmin = c6.slider("Min expected minutes", 0, 90, int(settings.differential_min_xmins), 5)

    pool = P[P["eligible"] & P["position"].isin(pos_sel) & (P["price"] <= max_price)
             & (P["season_minutes"] >= d_min)].copy()
    pool["is_diff"] = ((pool["ownership"] < d_own) & (pool["score_horizon"] >= d_score)
                       & (pool["xmins"] >= d_xmin))
    if st.toggle("Only players with set-piece duties", value=False,
                 help="Shows first- or second-choice penalty takers, corner takers "
                      "or direct free kick takers. Context only, never scored."):
        pool["has_set_pieces"] = pool["set_pieces"].fillna("").str.len() > 0
        pool["is_diff"] = pool["is_diff"] & pool["has_set_pieces"]
    diffs = pool[pool["is_diff"]].sort_values("score_horizon", ascending=False).copy()
    diffs["score"], diffs["fixtures"] = diffs["score_horizon"], diffs["horizon_fixtures"]
    diffs["minutes"] = fmt_minutes(diffs)
    diffs["rising"] = np.where(diffs["own_trend_3gw"].notna()
                               & (diffs["own_trend_3gw"] > 0), "Yes", "")
    st.markdown(f"**{len(diffs)} differentials** match these filters.")
    if diffs.empty:
        st.info("No players meet every filter. Try raising the ownership limit or lowering the "
                "minimum opportunity score.")
    else:
        show_table(diffs, ["name", "team", "position", "price", "ownership", "fixtures", "score",
                           "value_score", "minutes", "set_pieces", "rising", "market_note",
                           "confidence_label", "reasons", "risks"])
        download(diffs[EXPORT_COLS], "differentials", "dl_diff")

    plot = pool.dropna(subset=["ownership", "score_horizon"]).copy()
    plot["own_plot"] = plot["ownership"].clip(lower=0.1)
    fig = px.scatter(plot, x="own_plot", y="score_horizon", color="position", log_x=True,
                     color_discrete_map=POS_COLOURS, hover_name="name",
                     hover_data={"team": True, "price": ":.1f", "ownership": ":.1f",
                                 "own_plot": False, "score_horizon": ":.0f"},
                     labels={"own_plot": "Ownership % (log scale)",
                             "score_horizon": f"Opportunity score, {horizon} GWs"})
    hl = plot[plot["is_diff"]]
    fig.add_trace(go.Scatter(x=hl["own_plot"], y=hl["score_horizon"], mode="markers",
                             marker=dict(size=14, color="rgba(0,0,0,0)",
                                         line=dict(color=HIGHLIGHT, width=2.5)),
                             name="Differential", hoverinfo="skip"))
    fig.add_vline(x=d_own, line_dash="dot", line_color="#888")
    fig.add_hline(y=d_score, line_dash="dot", line_color="#888")
    fig.update_layout(height=460, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch", key="diff_scatter")
    st.caption("Gold rings mark differentials: bottom-right of the dotted lines is low ownership "
               "with a strong score.")

# ================================================== Form vs underlying =====
with tabs[2]:
    st.caption(f"Compares the last {recent} team matches. Percentiles are within position among "
               f"players with {settings.min_recent_minutes}+ recent minutes. 'Underlying' uses "
               "xG, xA, xGC-based clean-sheet chance, DefCon and saves; it ignores fixtures.")
    F = P[P["form_pool"]].copy()
    F["xgi90_recent_raw"] = F["xgi90_raw_recent"]
    form_cfg = {
        "r_minutes": st.column_config.NumberColumn("Recent mins", format="%d"),
        "xgi90_recent_raw": st.column_config.NumberColumn("Recent xGI/90", format="%.2f"),
        "underlying_pct": st.column_config.NumberColumn("Underlying pct", format="%d"),
        "points_recent90": st.column_config.NumberColumn("Pts/90 recent", format="%.1f"),
        "points_pct": st.column_config.NumberColumn("Points pct", format="%d"),
        "form_gap": st.column_config.NumberColumn("Gap", format="%+d"),
        "goals_minus_xg": st.column_config.NumberColumn("Goals − xG", format="%+.1f"),
        "assists_minus_xa": st.column_config.NumberColumn("Assists − xA", format="%+.1f"),
    }
    st.markdown("#### Underlying numbers ahead of returns")
    st.caption("Strong recent process, disappointing FPL points. Candidates to buy or hold, "
               "provided the minutes hold up.")
    unlucky = F[(F["underlying_pct"] >= 60) & (F["form_gap"] >= 25)].sort_values(
        "form_gap", ascending=False)
    show_table(unlucky, ["name", "team", "position", "price", "ownership", "r_minutes",
                         "xgi90_recent_raw", "underlying_pct", "points_recent90", "points_pct",
                         "form_gap", "risks"], form_cfg)

    st.markdown("#### Returns ahead of underlying numbers")
    st.caption("Points or goals running ahead of the chances. Elite finishers can sustain some "
               "over-performance, so treat this as a reason to look closer rather than a sell signal.")
    lucky = F[((F["form_gap"] <= -25) & (F["points_pct"] >= 60)) | (F["goals_minus_xg"] >= 2)]
    lucky = lucky.sort_values("form_gap")
    show_table(lucky, ["name", "team", "position", "price", "ownership", "r_minutes",
                       "underlying_pct", "points_pct", "form_gap", "goals_minus_xg",
                       "assists_minus_xa"], form_cfg)
    download(pd.concat([unlucky.assign(group="underlying ahead"),
                        lucky.assign(group="returns ahead")])[["group"] + EXPORT_COLS +
                       ["underlying_pct", "points_pct", "form_gap", "goals_minus_xg"]],
             "form_vs_underlying", "dl_form")

    att = P[P["eligible"] & P["position"].isin(["MID", "FWD"])].dropna(subset=["season_xg"])
    if not att.empty:
        mx = float(max(att["season_xg"].max(), att["season_goals"].max())) + 0.5
        fig = px.scatter(att, x="season_xg", y="season_goals", color="position",
                         color_discrete_map=POS_COLOURS, hover_name="name",
                         labels={"season_xg": "Season xG", "season_goals": "Season goals"})
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Goals = xG",
                                 line=dict(color="#888", dash="dot")))
        fig.update_layout(height=420, plot_bgcolor="white", margin=dict(l=0, r=0, t=30, b=0),
                          title=dict(text="Goals vs xG this season (MID/FWD)", font=dict(size=14)))
        st.plotly_chart(fig, width="stretch", key="g_xg")

# =================================================== Player comparison =====
with tabs[3]:
    opts = P[P["season_minutes"] > 0].sort_values("score_horizon", ascending=False)
    label_map = {f"{r.name} ({r.team}, {r.position}, £{r.price:.1f}m)": r.player_id
                 for r in opts.itertuples()}
    default = list(label_map)[:3]
    chosen = st.multiselect("Compare 2–4 players", list(label_map), default=default,
                            max_selections=4)
    ids = [label_map[c] for c in chosen]
    if len(ids) < 2:
        st.info("Pick at least two players to compare.")
    else:
        C = P.set_index("player_id").loc[ids]
        rows = {
            "Position": C["position"], "Price (£m)": C["price"], "Owned (%)": C["ownership"],
            "League minutes": C["season_minutes"],
            f"Minutes, last {recent} team matches": C["r_minutes"],
            f"Starts, last {recent}": C["r_starts"], "Expected minutes next": C["xmins"].round(0),
            "Minutes outlook": C["minutes_label"], "xG/90 season (raw)": C["xg90_raw_season"],
            "xG/90 blended": C["xg90_blend"], "xA/90 season (raw)": C["xa90_raw_season"],
            "xA/90 blended": C["xa90_blend"], "xGI/90 blended": C["xgi90_blend"],
            "xGC/90 season": C["xgc90_raw_season"], "DefCon hit rate": C["defcon_rate_blend"],
            f"FPL pts/90, last {recent}": C["points_recent90"],
                        "Goals − xG (season)": C["goals_minus_xg"],
            f"Opportunity GW{nxt}": C["score_next"], f"Opportunity {gw_range}": C["score_horizon"],
            "Value score": C["value_score"], "Data confidence": C["confidence_label"],
            "Next fixtures": C["horizon_fixtures"],
            "Set pieces": C["set_pieces"], "Market note": C["market_note"],
            "Last season minutes": C["past_minutes"],
            "Last season xG/90": C["past_xg90"], "Last season xA/90": C["past_xa90"],
            "Bonus/90 blended": C["bonus90_blend"], "BPS/90 season": C["bps90"],
            "Net transfers this GW": C["net_transfers_event"],
            "Ownership trend (3 GWs)": C["own_trend_3gw"],
        }
        table = pd.DataFrame(rows).T
        table.columns = C["name"].tolist()
        st.dataframe(table.map(lambda v: f"{v:.2f}" if isinstance(v, float) and not np.isnan(v)
                               else ("—" if isinstance(v, float) else v)), width="stretch")
        st.caption("— means unavailable (e.g. no minutes or missing history), never zero.")

        rates = C[["name", "xg90_blend", "xa90_blend"]].melt("name", var_name="Metric",
                                                             value_name="Per 90")
        rates["Metric"] = rates["Metric"].map({"xg90_blend": "xG/90", "xa90_blend": "xA/90"})
        fig = px.bar(rates, x="name", y="Per 90", color="Metric", barmode="group",
                     color_discrete_sequence=["#1F5C45", "#7E9C8A"])
        fig.update_layout(height=340, plot_bgcolor="white", xaxis_title=None,
                          margin=dict(l=0, r=0, t=30, b=0),
                          title=dict(text="Blended attacking output per 90", font=dict(size=14)))
        st.plotly_chart(fig, width="stretch", key="cmp_rates")

        tm = team_match_table(tables.fixtures)
        log = player_match_log(tables.players[tables.players["player_id"].isin(ids)],
                               tables.history, tm, last_n=8)
        log = log.merge(P[["player_id", "name"]], on="player_id")
        log["match"] = "−" + (log["match_no_desc"] - 1).astype(str)
        log.loc[log["match_no_desc"] == 1, "match"] = "Latest"
        order = ["−7", "−6", "−5", "−4", "−3", "−2", "−1", "Latest"]
        c1, c2 = st.columns(2)
        f1 = px.line(log, x="match", y="minutes", color="name", markers=True,
                     category_orders={"match": order})
        f1.update_layout(height=320, plot_bgcolor="white", yaxis_range=[-5, 95],
                         margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Team match",
                         title=dict(text="Minutes in each of the last 8 team matches",
                                    font=dict(size=14)))
        c1.plotly_chart(f1, width="stretch", key="cmp_min")
        f2 = px.bar(log, x="match", y="expected_goal_involvements", color="name",
                    barmode="group", category_orders={"match": order})
        f2.update_layout(height=320, plot_bgcolor="white", margin=dict(l=0, r=0, t=30, b=0),
                         xaxis_title="Team match", yaxis_title="xGI",
                         title=dict(text="xGI in each match", font=dict(size=14)))
        c2.plotly_chart(f2, width="stretch", key="cmp_xgi")

        det = res.fixture_detail[res.fixture_detail["player_id"].isin(ids)]
        if not det.empty:
            piv = det.groupby(["player_id", "event"])["raw"].sum().unstack("event")
            piv = piv.reindex(index=ids, columns=res.gws).fillna(0.0)
            piv.index = C["name"].tolist()
            piv.columns = [f"GW{g}" for g in piv.columns]
            st.markdown("**Opportunity by upcoming gameweek** (raw units; 0 = blank or unavailable)")
            st.dataframe(piv.round(2), width="stretch")
        download(C.reset_index()[EXPORT_COLS], "comparison", "dl_cmp")

# ====================================================== Market watch =====
with tabs[4]:
    st.caption("What other managers are doing. Read this as context only: it is never "
               "part of the opportunity score, and a popular player is not a better "
               "player. Ownership trend compares the last 3 finished gameweeks "
               "(deduplicated per round); the % column is approximate, using FPL's "
               "total-manager count rather than FPL's own calculation.")
    M = P[P["eligible"]].copy()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Heaviest transfers in this gameweek")
        tin = M.dropna(subset=["net_transfers_event"]).nlargest(10, "net_transfers_event")
        show_table(tin, ["name", "team", "position", "price", "ownership",
                         "net_transfers_event", "price_change_event_m", "score_horizon",
                         "market_note"])
    with c2:
        st.markdown("#### Heaviest transfers out")
        tout = M.dropna(subset=["net_transfers_event"]).nsmallest(10, "net_transfers_event")
        show_table(tout, ["name", "team", "position", "price", "ownership",
                          "net_transfers_event", "price_change_event_m", "score_horizon",
                          "market_note"])
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Biggest risers in ownership (3 GWs)")
        rise = M.dropna(subset=["own_trend_3gw"]).nlargest(10, "own_trend_3gw")
        show_table(rise, ["name", "team", "position", "price", "ownership", "own_trend_3gw",
                          "own_trend_pct", "score_horizon", "market_note"])
    with c4:
        st.markdown("#### Biggest fallers in ownership (3 GWs)")
        fall = M.dropna(subset=["own_trend_3gw"]).nsmallest(10, "own_trend_3gw")
        show_table(fall, ["name", "team", "position", "price", "ownership", "own_trend_3gw",
                          "own_trend_pct", "score_horizon", "market_note"])
    st.markdown("#### Rising differentials")
    st.caption("Low-owned players whose ownership is climbing: getting in before the "
               "crowd, not because popularity helps their score.")
    rd = M[(M["is_differential"]) & (M["own_trend_3gw"].fillna(0) > 0)] \
        .sort_values("own_trend_3gw", ascending=False)
    if rd.empty:
        st.info("No rising differentials right now.")
    else:
        show_table(rd, ["name", "team", "position", "price", "ownership", "own_trend_3gw",
                        "score_horizon", "fixtures", "market_note"])
    download(M.sort_values("net_transfers_event", ascending=False)[
        ["name", "team", "position", "price", "ownership", "net_transfers_event",
         "price_change_event_m", "price_since_start_m", "own_trend_3gw", "own_trend_pct",
         "bps_total", "score_horizon", "market_note"]], "market_watch", "dl_mkt")

# ======================================================== Methodology =====
with tabs[5]:
    sc = tables.scoring

    def by_pos(v):
        return ", ".join(f"{k} {v[k]}" for k in POSITION_ORDER if k in v) if isinstance(v, dict) else v

    st.markdown(f"""
#### What this is
A transparent heuristic that ranks players on underlying output, minutes and fixtures.
It is **not** an expected-points model, and it has **not** been tested against simpler
methods yet (see *How to evaluate it* below).

#### Data
* **FPL API** (no key): `bootstrap-static` (players, prices, ownership, availability, season
  totals, scoring rules), `fixtures` (schedule, results, FDR), `element-summary/{{id}}`
  (per-match history for every player with minutes).
* Detected season **{tables.season}**, next gameweek **GW{nxt}**, last finished **GW{tables.last_finished_gw}**.
* Scoring values read from **{sc.get('_source')}**: goals ({by_pos(sc['goals_scored'])}), assists {sc['assists']},
  clean sheets ({by_pos(sc['clean_sheets'])}), DefCon points ({by_pos(sc['defensive_contribution'])}).
* DefCon thresholds are not in the API. Source: {DEFCON_SOURCE}. Thresholds used: {by_pos(DEFCON_THRESHOLD)} actions.
* Caching: bootstrap and fixtures 30 min, player histories 6 h or until a new GW finishes;
  3 retries with back-off, {fpl_data.MAX_WORKERS} concurrent requests, last-good snapshot on disk.

#### Formulas (and where to change them in `aotc/metrics.py`)
1. **Per 90** (`per90`): total ÷ minutes × 90. Zero minutes gives *unavailable*, not 0.
2. **Shrinkage** (`shrink_rate`, `blended_rates`): every rate gets "phantom" minutes at a baseline.
   * season_shrunk = (season total + baseline × {settings.prior_minutes_season:.0f} min) ÷ (minutes + {settings.prior_minutes_season:.0f})
   * recent_shrunk = same idea over the last {recent} team matches, shrunk toward season_shrunk with {settings.prior_minutes_recent:.0f} phantom min
   * blended = {settings.recent_weight:.2f} × recent_shrunk + {1 - settings.recent_weight:.2f} × season_shrunk
   * *Example:* a sub with 1.0 xG in 20 minutes has a raw 4.5 xG/90. With a 0.30 baseline and
     450 phantom minutes he becomes (1.0 + 1.5) ÷ (470/90) ≈ 0.48 xG/90.
   * **Past-season prior** (when "Use last season as prior" is on): for xG/90, xA/90, saves/90
     and xGC/90 the baseline is replaced by the player's OWN last usable past season (newest
     season with minutes > 0 within the last 2), itself shrunk toward the position baseline.
     The past season contributes {settings.past_season_decay:.2f}× its real minutes as phantom
     minutes (the season before that, squared); the phantom minutes cap of {settings.prior_minutes_season:.0f}
     still applies to the season blend. Bonus and DefCon are NOT shrunk to the past: the Bonus
     Points System was rebalanced for 2026/27, and past seasons have no per-match DefCon detail.
     Players who joined a new club get a risk note, because their past rate describes another
     team. xG/xA/xGC/starts only exist in the API from 2022/23 (before that the fields are
     zero placeholders); a keeper's genuine 0 xG stays 0, never NaN.
3. **Attack index** (`attack_index`): blended xG/90 × goal points for his position + blended xA/90 × {sc['assists']}.
   xGI is shown but not scored, because xGI = xG + xA would double-count.
4. **Minutes outlook** (`minutes_outlook`): start share = (recent starts + 2 × season start share) ÷
   (recent team matches + 2). Expected minutes = start share × average minutes when starting +
   sub share × average minutes off the bench. Matches the player missed count as zeros.
5. **Team strength** (`team_strength`): team xG for/against per match (summed from player rows),
   shrunk with {settings.team_prior_matches:.0f} matches of league average. Current source: **{res.strength.source}**,
   league average {res.strength.league_avg:.2f} xG, home ×{res.strength.home_factor:.2f}, away ×{res.strength.away_factor:.2f}.
6. **Fixture outlook** (`upcoming_fixture_outlook`): λ_for = league avg × attack × opponent concede ×
   home/away; clean-sheet chance = e^(−λ_against). Fixture influence ({settings.fixture_strength:.1f}) is an exponent on the opponent effect.
7. **Components per fixture** (`fixture_components`): attack index × fixture multiplier × expected minutes;
   clean-sheet points × CS chance × share of 60-min matches; DefCon points × hit rate × 60-min share;
   GK saves/90 ÷ 3; bonus = blended bonus/90 × expected minutes, weighted {settings.w_bonus:.1f}
   because bonus overlaps attacking returns. Summed over fixtures (DGWs count twice, blanks zero)
   and multiplied by the availability haircut.
8. **Opportunity score**: percentile of the weighted sum within position, among players with
   {settings.min_season_minutes}+ league minutes. 100 = best in position for that horizon.
9. **Value**: opportunity per £m, percentile within position.
10. **Data confidence** (`data_confidence`): 50% sample (minutes ÷ 900, plus HALF of up to 900
    past-season minutes, capped), 30% minutes stability, 20% input coverage.
    It describes how much data stands behind a row, not a statistical interval.
11. **Context columns (never scored)**: set-piece ranks ("Pens 1", "Corners 2", "FKs 1") are a
    reading aid — penalty and set-piece chances are already inside xG/xA. Market signals
    (net transfers, price changes, 3-GW ownership trend) live in a separate `market_note`
    column and the Market watch tab: what other managers do is not football quality.
    FPL's undocumented `price_change_percent` / `price_change_projections` fields are
    deliberately left out until verified.

#### Current weights
""")
    wt = pd.DataFrame({
        "Setting": ["Attack", "Clean sheet", "DefCon", "Saves", "Bonus", "Fixture influence",
                    "Recent weight", "Use past season", "Past-season decay",
                    "Season phantom minutes", "Recent phantom minutes", "DefCon phantom apps",
                    "Min league minutes", "Ownership threshold %"],
        "Current": [settings.w_attack, settings.w_clean_sheet, settings.w_defcon, settings.w_saves,
                    settings.w_bonus, settings.fixture_strength, settings.recent_weight,
                    "on" if settings.use_past_seasons else "off", settings.past_season_decay,
                    settings.prior_minutes_season,
                    settings.prior_minutes_recent, settings.prior_apps_defcon,
                    settings.min_season_minutes, settings.ownership_threshold],
        "Default": [D.w_attack, D.w_clean_sheet, D.w_defcon, D.w_saves, D.w_bonus,
                    D.fixture_strength, D.recent_weight,
                    "on" if D.use_past_seasons else "off", D.past_season_decay,
                    D.prior_minutes_season, D.prior_minutes_recent,
                    D.prior_apps_defcon, D.min_season_minutes, D.ownership_threshold],
    })
    wt["Current"], wt["Default"] = wt["Current"].astype(str), wt["Default"].astype(str)
    st.dataframe(wt, hide_index=True, width="content")

    haircut_text = ", ".join(f"{k}% → ×{v}" for k, v in AVAILABILITY_HAIRCUT.items())
    st.markdown(f"""
#### Availability
FPL's chance-of-playing flag is club news, not a calibrated probability. It maps to a coarse
haircut on the next GW: {haircut_text}. Injured or suspended players get
×{LATER_GW_FACTOR_IF_OUT} in later GWs (an assumption, because return dates aren't structured). Players who left the club get 0.

#### Missing data
* A missing API field or failed player history shows as blank/"—" and is never treated as 0.
* Team matches a player missed count as 0 minutes, but only when his history loaded.
* If player rows cover under 80% of a match's minutes, that match's team strength uses actual goals.
* Postponed or unscheduled fixtures are excluded from form windows and look-aheads.
""")
    if not res.unscheduled.empty:
        st.markdown("**Currently unscheduled fixtures**")
        st.dataframe(res.unscheduled[["fixture_id", "home", "away"]], hide_index=True)

    st.markdown("""
#### Known limitations
* No expected-points model: appearance points, bonus, cards, goals-conceded deductions, penalties
  saved and own goals are not scored.
* Team strength is not adjusted for opponent quality (schedule strength), and the home/away factor is league-wide.
* FPL xG is one provider's model. Non-penalty xG isn't available, so penalty takers' xG includes penalties.
* A player who moved between Premier League clubs only has matches for his current club in the form window.
* Clean-sheet chance uses a simple Poisson approximation, and the 60-minute share isn't a calibrated probability.
* Very early in a season, the FDR fallback replaces the xG team model.

#### How to evaluate it (not done yet)
Save a frozen copy of the inputs **before each deadline** (prices, ownership, flags, histories up to that point).
Score the rankings after the gameweek against **only** those saved inputs, comparing with simple baselines:
(a) season FPL points per match, (b) FPL `form`, (c) price, (d) FPL's own `ep_next`.
Use rank correlation with actual points and the hit rate of the top-10 per position, over many gameweeks.
Don't backtest past gameweeks with today's ownership, injury flags or match data, because that leaks the future.
""")
    with st.expander("Team strength table"):
        if not res.strength.table.empty:
            ts = res.strength.table.merge(tables.teams, on="team_id")
            st.dataframe(ts[["team", "matches", "xgf_pm", "xga_pm", "attack", "concede"]]
                         .sort_values("attack", ascending=False).round(2), hide_index=True)
        else:
            st.write("Not available: FDR fallback in use.")
    with st.expander("Position baselines used for shrinkage"):
        st.dataframe(res.baselines.round(3))
