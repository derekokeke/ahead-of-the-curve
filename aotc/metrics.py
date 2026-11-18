"""Football maths. Every number shown in the dashboard is produced here.

Reading order if you are learning the code:
  per90 / shrink_rate          -> the two basic building blocks
  team_match_table             -> one row per team per completed league match
  player_match_log             -> each player against each of his team's matches
  summarise_log                -> totals over a window (season or last N)
  blended_rates                -> recent + season + baseline, with shrinkage
  minutes_outlook              -> expected minutes, start share, rotation flags
  team_strength                -> attack/concede indices from match xG
  upcoming_fixture_outlook     -> expected goals for/against per future fixture
  fixture_components           -> per-player, per-fixture component values
  compute                      -> glues it all together into one player table
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    AVAILABILITY_HAIRCUT,
    BASELINE_MIN_MINUTES,
    CONFIDENCE_PAST_CAP,
    DEFAULT_LEAGUE_XG,
    DEFCON_THRESHOLD,
    FDR_ATTACK_MULTIPLIER,
    LATER_GW_FACTOR_IF_OUT,
    PAST_SEASONS_USED,
    TEAM_XG_MIN_COVERAGE,
    Settings,
)
from .transform import Tables

STAT_COLS = [
    "minutes", "starts", "goals_scored", "assists", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "total_points", "saves",
    "defensive_contribution", "bonus", "clean_sheets",
]


# ================================================================ basics =====
def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def _like(template, values):
    if isinstance(template, pd.Series):
        return pd.Series(values, index=template.index)
    values = np.asarray(values)
    return float(values) if values.ndim == 0 else values


def per90(total, minutes):
    """Rate per 90 minutes. Zero (or missing) minutes gives NaN, never 0.

    Example: 1 goal in 45 minutes -> 2.0 goals per 90.
    """
    t, m = _arr(total), _arr(minutes)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(m > 0, t / m * 90.0, np.nan)
    return _like(total, out)


def shrink_rate(successes, trials, prior_rate, prior_trials):
    """Credibility-weighted rate: (observed + prior pseudo-observations) / (trials + prior trials).

        With few trials the answer stays near `prior_rate`; with many it approaches the
        observed rate. If the prior is missing, the raw rate is returned.
    If the observed data is missing (NaN), the result is NaN so that "unavailable"
    is never silently replaced by a guess.
    `prior_rate` and `prior_trials` may be scalars or arrays of the same length
    (per-player priors, e.g. the player's own past season).
    """
    s, n, p = _arr(successes), _arr(trials), _arr(prior_rate)
    k = np.where(np.isnan(p), 0.0, _arr(prior_trials))
    p0 = np.where(np.isnan(p), 0.0, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n + k > 0, (s + p0 * k) / (n + k), np.nan)
    return _like(successes, out)


def shrink_per90(total, minutes, prior_per90, prior_minutes):
    """shrink_rate expressed per 90 minutes (trials are 90-minute blocks)."""
    return shrink_rate(total, _arr(minutes) / 90.0, prior_per90, prior_minutes / 90.0)


# ========================================================= match tables =====
def team_match_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed league match, newest first per team.

    Postponed fixtures (event is NaN and not completed) never appear here, and a
    double gameweek simply contributes two rows, so windows are counted in team
    matches rather than gameweeks.
    """
    done = fixtures[fixtures["completed"]].copy()
    home = pd.DataFrame({
        "team_id": done["team_h"], "opp_id": done["team_a"], "is_home": True,
        "gf": done["team_h_score"], "ga": done["team_a_score"],
    })
    away = pd.DataFrame({
        "team_id": done["team_a"], "opp_id": done["team_h"], "is_home": False,
        "gf": done["team_a_score"], "ga": done["team_h_score"],
    })
    for part in (home, away):
        part["fixture_id"] = done["fixture_id"].values
        part["event"] = done["event"].values
        part["kickoff_time"] = done["kickoff_time"].values
    tm = pd.concat([home, away], ignore_index=True)
    tm = tm.sort_values(["team_id", "kickoff_time", "fixture_id"], ascending=[True, False, False])
    tm["match_no_desc"] = tm.groupby("team_id").cumcount() + 1  # 1 = most recent
    return tm.reset_index(drop=True)


def player_match_log(players: pd.DataFrame, history: pd.DataFrame,
                     team_matches: pd.DataFrame, last_n: int | None = None) -> pd.DataFrame:
    """Each player against each of his CURRENT team's completed matches.

    A team match with no history row means the player did not feature (0 minutes),
    which is how missed matches count against recent form. That is only valid when
    we actually have the player's history; otherwise stats stay NaN (unavailable).
    """
    tm = team_matches if last_n is None else team_matches[team_matches["match_no_desc"] <= last_n]
    base = players[["player_id", "team_id", "position", "history_available"]].merge(
        tm[["team_id", "fixture_id", "match_no_desc", "opp_id", "is_home", "event"]],
        on="team_id", how="inner",
    )
    present = [c for c in STAT_COLS if c in history.columns]
    h = history[["player_id", "fixture_id"] + present]
    log = base.merge(h, on=["player_id", "fixture_id"], how="left")
    avail = log["history_available"].to_numpy(dtype=bool)

    for c in STAT_COLS:
        if c in present:
            log[c] = np.where(avail, log[c].fillna(0.0), np.nan)
        else:
            log[c] = np.nan  # field not provided by the API: unavailable, not zero

    m = log["minutes"]
    if "starts" not in present:
        # Older data had no 'starts'. Approximation, flagged in Methodology.
        log["starts"] = np.where(avail, (m >= 60).astype(float), np.nan)
    st = log["starts"]

    def flag(cond):
        return np.where(avail, cond.astype(float), np.nan)

    log["app"] = flag(m > 0)
    log["app60"] = flag(m >= 60)
    log["early_sub"] = flag((st >= 1) & (m < 60))
    log["sub_app"] = flag((st < 1) & (m > 0))
    log["start_minutes"] = np.where(avail, np.where(st >= 1, m, 0.0), np.nan)
    log["sub_minutes"] = np.where(avail, np.where(st < 1, m, 0.0), np.nan)

    thr = log["position"].map(DEFCON_THRESHOLD)
    dc = log["defensive_contribution"]
    hit = (dc >= thr) & (m >= 60)
    log["defcon_hit"] = np.where(avail & thr.notna().to_numpy() & dc.notna().to_numpy(),
                                 hit.astype(float), np.nan)
    return log


def summarise_log(log: pd.DataFrame) -> pd.DataFrame:
    """Totals over whatever window `log` covers. min_count=1 keeps all-NaN as NaN."""
    g = log.groupby("player_id")

    def s(c):
        return g[c].sum(min_count=1)

    out = pd.DataFrame({
        "team_matches": g.size(),
        "minutes": s("minutes"), "starts": s("starts"), "apps": s("app"),
        "apps60": s("app60"), "early_subs": s("early_sub"), "sub_apps": s("sub_app"),
        "start_minutes": s("start_minutes"), "sub_minutes": s("sub_minutes"),
        "xg": s("expected_goals"), "xa": s("expected_assists"),
        "xgi": s("expected_goal_involvements"), "xgc": s("expected_goals_conceded"),
        "goals": s("goals_scored"), "assists": s("assists"), "points": s("total_points"),
        "saves": s("saves"), "bonus": s("bonus"), "defcon_hits": s("defcon_hit"),
        "minutes_sd": g["minutes"].std(ddof=0),
    })
    out["minutes_last3_avg"] = log[log["match_no_desc"] <= 3].groupby("player_id")["minutes"].mean()
    out["minutes_earlier_avg"] = log[log["match_no_desc"] > 3].groupby("player_id")["minutes"].mean()
    return out


# ============================================================ baselines =====
def select_past_season(past: pd.DataFrame, current_year: int | None,
                       decay: float) -> pd.DataFrame:
    """Newest usable past season per player, within the last PAST_SEASONS_USED.

    Usable = a PL season (history_past row) with minutes > 0. The effective prior
    minutes are past minutes x decay^k, where k = 1 for last season and 2 for the
    season before (the user-approved judgement call). Returns one row per player
    (indexed by player_id) with the season totals; NaN when there is no usable
    past season, so those players keep the exact current (baseline) behaviour.
    """
    cols = ["past_minutes_eff", "past_xg", "past_xa", "past_saves", "past_xgc", "past_bonus"]
    if past is None or past.empty or not current_year:
        return pd.DataFrame(columns=cols)
    p = past[past["season_year"].notna()].copy()
    p["k"] = current_year - p["season_year"]  # 1 = last season, 2 = two seasons ago
    p = p[(p["k"] >= 1) & (p["k"] <= PAST_SEASONS_USED) & (p["minutes"].fillna(0) > 0)]
    if p.empty:
        return pd.DataFrame(columns=cols)
    p = p.sort_values(["player_id", "k"]).groupby("player_id").tail(1)  # newest = smallest k
    out = pd.DataFrame(index=pd.Index(p["player_id"].astype(int), name="player_id"))
    out["past_minutes_eff"] = p["minutes"].to_numpy() * (float(decay) ** p["k"].to_numpy())
    out["past_minutes"] = p["minutes"].to_numpy()
    out["past_season"] = p["season"].to_numpy()
    for c, src in [("past_xg", "xg"), ("past_xa", "xa"), ("past_saves", "saves"),
                   ("past_xgc", "xgc"), ("past_bonus", "bonus")]:
        out[c] = p[src].to_numpy()
    return out


def position_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Median per-90 rates by position among players with a decent sample.

    These are the "sensible baseline" that small samples are shrunk toward.
    """
    q = df[df["season_minutes"] >= BASELINE_MIN_MINUTES]
    too_few = q.empty or q.groupby("position").size().min() < 5
    if too_few:
        q = df[df["season_minutes"] > 0]  # early season: use everyone who has played
    b = q.assign(
        xg90=per90(q["season_xg"], q["season_minutes"]),
        xa90=per90(q["season_xa"], q["season_minutes"]),
                saves90=per90(q["season_saves"], q["season_minutes"]),
        xgc90=per90(q["season_xgc"], q["season_minutes"]),
        bonus90=per90(q["s_bonus"], q["season_minutes"]),
        defcon_rate=q["s_defcon_hits"] / q["s_apps60"].where(q["s_apps60"] > 0),
    ).groupby("position")[["xg90", "xa90", "saves90", "xgc90", "bonus90",
                           "defcon_rate"]].median()
    return b


def blended_rates(df: pd.DataFrame, base: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Three-step blend for each rate (xG, xA, saves, xGC, bonus per 90; DefCon hit rate).

    1. prior = position baseline (phantom minutes = s.prior_minutes_season).
       When s.use_past_seasons and the player has a usable past season, the prior
       is instead the player's OWN past rate (itself shrunk toward the position
       baseline), with phantom minutes = past minutes x past_season_decay^k.
       Players with no usable past season behave exactly as before. DefCon is
       excluded: past seasons have only season totals, not per-match hits.
       Bonus uses the past only when s.bonus_use_past is set (BPS was rebalanced
       for 2026/27, so older bonus rates may not transfer).
    2. recent_shrunk = last-N-matches rate shrunk toward season_shrunk
    3. blended       = recent_weight * recent_shrunk + (1 - recent_weight) * season_shrunk
       (falls back to season_shrunk when recent data is unavailable)
    """
    out = pd.DataFrame(index=df.index)
    pos = df["position"]
    use_past = bool(getattr(s, "use_past_seasons", True))
    past_mins = (df["past_minutes_eff"].where(df["past_minutes_eff"].fillna(0) > 0)
                 if "past_minutes_eff" in df.columns
                 else pd.Series(np.nan, index=df.index))
    for stat, season_col, recent_col, past_col in [
        ("xg90", "season_xg", "r_xg", "past_xg"),
        ("xa90", "season_xa", "r_xa", "past_xa"),
        ("saves90", "season_saves", "r_saves", "past_saves"),
        ("xgc90", "season_xgc", "r_xgc", "past_xgc"),
    ]:
        baseline = pos.map(base[stat]) if stat in base else pd.Series(np.nan, index=df.index)
        past_tot = (df[past_col] if past_col in df.columns
                    else pd.Series(np.nan, index=df.index))
        use_own = use_past & past_tot.notna() & past_mins.notna()
        # Step 0: the player's own past rate shrunk toward the position baseline...
        own_prior = shrink_per90(past_tot, past_mins, baseline, s.prior_minutes_season)
        prior = np.where(use_own & own_prior.notna(), own_prior, baseline)
        prior_min = np.where(use_own & own_prior.notna(), past_mins, s.prior_minutes_season)
        season_sh = shrink_per90(df[season_col], df["season_minutes"], prior, prior_min)
        recent_sh = shrink_per90(df[recent_col], df["r_minutes"], season_sh, s.prior_minutes_recent)
        out[f"{stat}_season"] = season_sh
        out[f"{stat}_recent"] = recent_sh
        out[f"{stat}_blend"] = np.where(
            np.isnan(_arr(recent_sh)), season_sh,
            s.recent_weight * recent_sh + (1 - s.recent_weight) * season_sh,
        )

    # Bonus: fifth component. Past-season prior only when s.bonus_use_past.
    baseline_b = pos.map(base["bonus90"]) if "bonus90" in base else pd.Series(np.nan, index=df.index)
    past_bonus = (df["past_bonus"] if "past_bonus" in df.columns
                  else pd.Series(np.nan, index=df.index))
    use_own_b = use_past & bool(getattr(s, "bonus_use_past", False)) \
        & past_bonus.notna() & past_mins.notna()
    own_prior_b = shrink_per90(past_bonus, past_mins, baseline_b, s.prior_minutes_season)
    prior_b = np.where(use_own_b & own_prior_b.notna(), own_prior_b, baseline_b)
    prior_min_b = np.where(use_own_b & own_prior_b.notna(), past_mins, s.prior_minutes_season)
    b_season = shrink_per90(df["s_bonus"], df["season_minutes"], prior_b, prior_min_b)
    b_recent = shrink_per90(df["r_bonus"], df["r_minutes"], b_season, s.prior_minutes_recent)
    out["bonus90_season"] = b_season
    out["bonus90_recent"] = b_recent
    out["bonus90_blend"] = np.where(
        np.isnan(_arr(b_recent)), b_season,
        s.recent_weight * b_recent + (1 - s.recent_weight) * b_season,
    )

    prior_dc = pos.map(base["defcon_rate"]) if "defcon_rate" in base else np.nan
    dc_season = shrink_rate(df["s_defcon_hits"], df["s_apps60"], prior_dc, s.prior_apps_defcon)
    dc_recent = shrink_rate(df["r_defcon_hits"], df["r_apps60"], dc_season, s.prior_apps_defcon)
    out["defcon_rate_season"] = dc_season
    out["defcon_rate_recent"] = dc_recent
    out["defcon_rate_blend"] = np.where(
        np.isnan(_arr(dc_recent)), dc_season,
        s.recent_weight * dc_recent + (1 - s.recent_weight) * dc_season,
    )
    out.loc[~pos.isin(DEFCON_THRESHOLD.keys()), ["defcon_rate_season", "defcon_rate_recent",
                                                  "defcon_rate_blend"]] = np.nan
    # No league minutes at all: there is nothing to shrink, so report "unavailable"
    # rather than showing the position baseline as if it were the player's own rate.
    out.loc[df["season_minutes"].fillna(0) <= 0, :] = np.nan
    return out


# ======================================================= minutes outlook =====
def availability_factor(status: str, chance) -> tuple[float, float]:
    """(factor for next GW, factor for later GWs). A coarse haircut, not a probability."""
    status = (status or "a").lower()
    if status in ("u", "n"):  # left the club / not available (e.g. loaned out)
        return 0.0, 0.0
    if chance is not None and not pd.isna(chance):
        nearest = min(AVAILABILITY_HAIRCUT, key=lambda k: abs(k - float(chance)))
        nxt = AVAILABILITY_HAIRCUT[nearest]
    else:
        nxt = 1.0 if status in ("a", "d") else 0.0
    later = 1.0 if status in ("a", "d") else max(nxt, LATER_GW_FACTOR_IF_OUT)
    return nxt, later


def minutes_outlook(df: pd.DataFrame) -> pd.DataFrame:
    """Expected minutes per match from recent starts/sub patterns, shrunk to the season.

    start_share = (recent starts + 2 * season start share) / (recent team matches + 2)
    xmins       = start_share * avg minutes when starting
                + sub_share   * avg minutes off the bench
    p60         = similar shrunk share of team matches with 60+ minutes
    """
    k = 2.0
    n_r = df["r_team_matches"].fillna(0)
    n_s = df["s_team_matches"].replace(0, np.nan)

    season_start = (df["s_starts"] / n_s).fillna(df["season_starts"] / n_s)
    season_sub = (df["s_sub_apps"] / n_s).fillna(0.0)
    season_p60 = (df["s_apps60"] / n_s).fillna(season_start)

    def shrunk(recent_count, season_share):
        return ((recent_count.fillna(season_share * n_r) + k * season_share) / (n_r + k)).clip(0, 1)

    start_share = shrunk(df["r_starts"], season_start.fillna(0))
    sub_share = shrunk(df["r_sub_apps"], season_sub)
    p60 = shrunk(df["r_apps60"], season_p60.fillna(0))

    avg_start = (df["r_start_minutes"] / df["r_starts"].replace(0, np.nan))
    avg_start = avg_start.fillna(df["s_start_minutes"] / df["s_starts"].replace(0, np.nan))
    avg_start = avg_start.fillna(df["season_minutes"] / df["season_starts"].replace(0, np.nan))
    avg_start = avg_start.fillna(85.0).clip(upper=90)
    avg_sub = (df["r_sub_minutes"] / df["r_sub_apps"].replace(0, np.nan))
    avg_sub = avg_sub.fillna(df["s_sub_minutes"] / df["s_sub_apps"].replace(0, np.nan)).fillna(20.0)

    xmins = start_share * avg_start + np.minimum(sub_share, 1 - start_share) * avg_sub

    trend = df["r_minutes_last3_avg"] - df["r_minutes_earlier_avg"]
    trend_label = np.select([trend >= 15, trend <= -15], ["rising", "falling"], "steady")
    trend_label = np.where(trend.isna(), "n/a", trend_label)

    early_rate = df["r_early_subs"] / df["r_starts"].replace(0, np.nan)
    label = np.select(
        [(start_share >= 0.85) & (early_rate.fillna(0) < 0.25), start_share >= 0.65,
         start_share >= 0.35],
        ["Nailed", "Likely starter", "Rotation risk"], "Bench/fringe",
    )
    return pd.DataFrame({
        "start_share": start_share, "sub_share": sub_share, "p60": p60,
        "xmins": xmins, "minutes_trend": trend_label, "minutes_label": label,
        "early_sub_rate": early_rate,
    }, index=df.index)


# ========================================================= team strength =====
def team_xg_by_match(history: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """Team xG for/against per match, summed from player rows.

    If player rows cover <80% of a match's minutes (histories missing), that match
    falls back to actual goals, and the source is recorded.
    """
    if history.empty or "expected_goals" not in history.columns:
        tm = team_matches.copy()
        tm["xgf"], tm["xga"], tm["xg_source"] = tm["gf"], tm["ga"], "goals"
        return tm
    agg = history.groupby(["team_id", "fixture_id"]).agg(
        xg_sum=("expected_goals", "sum"), mins=("minutes", "sum")).reset_index()
    tm = team_matches.merge(agg, on=["team_id", "fixture_id"], how="left")
    ok = (tm["mins"] / 990.0) >= TEAM_XG_MIN_COVERAGE
    tm["xgf"] = np.where(ok, tm["xg_sum"], tm["gf"])
    tm["xg_source"] = np.where(ok, "xG", "goals")
    opp = tm[["team_id", "fixture_id", "xgf"]].rename(columns={"team_id": "opp_id", "xgf": "xga"})
    tm = tm.merge(opp, on=["opp_id", "fixture_id"], how="left")
    tm["xga"] = tm["xga"].fillna(tm["ga"])
    return tm


@dataclass
class StrengthModel:
    table: pd.DataFrame        # per team: matches, attack, concede (1.0 = league average)
    league_avg: float
    home_factor: float
    away_factor: float
    source: str                # "xG model" or "FPL FDR fallback"


def team_strength(tmx: pd.DataFrame, s: Settings) -> StrengthModel:
    """attack  = (team xG for + k * league avg) / (matches + k) / league avg
    concede = (team xG against + k * league avg) / (matches + k) / league avg
    (k = team_prior_matches). 1.20 attack means 20% more xG than an average side.
    """
    if tmx.empty or tmx.groupby("team_id").size().median() < 3:
        return StrengthModel(pd.DataFrame(), DEFAULT_LEAGUE_XG, 1.0, 1.0, "FPL FDR fallback")
    k = s.team_prior_matches
    league_avg = float(tmx["xgf"].mean())
    g = tmx.groupby("team_id").agg(matches=("fixture_id", "size"), xgf=("xgf", "sum"),
                                  xga=("xga", "sum"),
                                  xg_share=("xg_source", lambda x: (x == "xG").mean()))
    g["attack"] = (g["xgf"] + k * league_avg) / (g["matches"] + k) / league_avg
    g["concede"] = (g["xga"] + k * league_avg) / (g["matches"] + k) / league_avg
    g["xgf_pm"] = g["xgf"] / g["matches"]
    g["xga_pm"] = g["xga"] / g["matches"]
    home = float(tmx.loc[tmx["is_home"], "xgf"].mean() / league_avg)
    away = float(tmx.loc[~tmx["is_home"], "xgf"].mean() / league_avg)
    home = float(np.clip(home if np.isfinite(home) else 1.08, 1.0, 1.25))
    away = float(np.clip(away if np.isfinite(away) else 0.92, 0.8, 1.0))
    src = "xG model" if g["xg_share"].mean() >= 0.8 else "xG model (partly goals-based)"
    return StrengthModel(g.reset_index(), league_avg, home, away, src)


def upcoming_fixture_outlook(fixtures: pd.DataFrame, model: StrengthModel,
                             gws: list[int], s: Settings) -> pd.DataFrame:
    """One row per team per scheduled fixture in `gws` (DGWs give two rows; blanks none).

    lambda_for     = league avg * team attack * opponent concede * home/away factor
    lambda_against = league avg * opponent attack * team concede * opposite factor
    cs_prob        = exp(-lambda_against)   (Poisson chance of conceding zero)
    fixture_strength (setting) is an exponent that dials the opponent effect up/down.
    """
    fut = fixtures[fixtures["event"].isin(gws) & ~fixtures["completed"]]
    rows = []
    for _, f in fut.iterrows():
        for team, opp, home, diff in [(f.team_h, f.team_a, True, f.team_h_difficulty),
                                      (f.team_a, f.team_h, False, f.team_a_difficulty)]:
            rows.append({"team_id": team, "opp_id": opp, "is_home": home, "event": int(f.event),
                         "fixture_id": f.fixture_id, "fdr": diff, "kickoff_time": f.kickoff_time})
    out = pd.DataFrame(rows, columns=["team_id", "opp_id", "is_home", "event", "fixture_id",
                                      "fdr", "kickoff_time"])
    if out.empty:
        return out.assign(lambda_for=[], lambda_against=[], att_mult=[], cs_prob=[])

    L = model.league_avg
    if model.source.startswith("xG"):
        t = model.table.set_index("team_id")
        att, con = out["team_id"].map(t["attack"]), out["team_id"].map(t["concede"])
        o_att, o_con = out["opp_id"].map(t["attack"]), out["opp_id"].map(t["concede"])
        hf = np.where(out["is_home"], model.home_factor, model.away_factor)
        of = np.where(out["is_home"], model.away_factor, model.home_factor)
        out["lambda_for"] = (L * att.fillna(1) * o_con.fillna(1) * hf).astype(float)
        out["lambda_against"] = (L * o_att.fillna(1) * con.fillna(1) * of).astype(float)
    else:
        mult = out["fdr"].map(FDR_ATTACK_MULTIPLIER).fillna(1.0)
        out["lambda_for"] = L * mult
        out["lambda_against"] = L / mult

    a = s.fixture_strength
    out["att_mult"] = (out["lambda_for"] / L) ** a
    lam_against_adj = L * (out["lambda_against"] / L) ** a
    out["cs_prob"] = np.exp(-lam_against_adj)
    return out


# ============================================================ components =====
def attack_index(xg90, xa90, position: pd.Series, scoring: dict):
    """Attacking threat per 90 in FPL points-equivalents: xG x goal points + xA x assist points.

    Uses xG and xA (which don't overlap) and NOT xGI, so nothing is counted twice.
    A MID with 0.30 xG/90 and 0.20 xA/90 -> 0.30*5 + 0.20*3 = 2.1.
    """
    gpts = position.map(scoring["goals_scored"]).astype(float)
    return _arr(xg90) * gpts + _arr(xa90) * float(scoring["assists"])


def fixture_components(df: pd.DataFrame, outlook: pd.DataFrame, next_gw: int,
                       scoring: dict, s: Settings) -> pd.DataFrame:
    """Per player, per fixture: attack, clean sheet, DefCon, saves and bonus components.

    All in rough FPL points-equivalents, then scaled by expected minutes and the
    availability haircut. Not an expected-points forecast: appearance points,
    cards, goals-conceded deductions and own goals are deliberately left out.
    Bonus IS included (w_bonus, default 0.5): it partly overlaps attacking
    returns, because big xG/xA days usually earn bonus as well.
    """
    cols = ["player_id", "team_id", "position", "attack90", "defcon_rate_blend",
            "saves90_blend", "bonus90_blend", "xmins", "p60", "avail_next", "avail_later"]
    pf = df[cols].merge(outlook, on="team_id", how="inner")
    if pf.empty:
        return pf
    cs_pts = pf["position"].map(scoring["clean_sheets"]).astype(float)
    dc_pts = pf["position"].map(scoring["defensive_contribution"]).astype(float)
    mins_share = pf["xmins"] / 90.0

    pf["c_attack"] = pf["attack90"] * pf["att_mult"] * mins_share
    pf["c_clean_sheet"] = cs_pts * pf["cs_prob"] * pf["p60"]
    pf["c_defcon"] = (dc_pts * pf["defcon_rate_blend"] * pf["p60"]).fillna(0.0)
    is_gk = pf["position"] == "GKP"
    pf["c_saves"] = np.where(is_gk, pf["saves90_blend"].fillna(0) / 3.0 * float(scoring["saves"])
                             * mins_share, 0.0)
    pf["c_bonus"] = pf["bonus90_blend"].fillna(0.0) * mins_share
    avail = np.where(pf["event"] == next_gw, pf["avail_next"], pf["avail_later"])
    for c in ["c_attack", "c_clean_sheet", "c_defcon", "c_saves", "c_bonus"]:
        pf[c] = pf[c].fillna(0.0) * avail
    pf["raw"] = (s.w_attack * pf["c_attack"] + s.w_clean_sheet * pf["c_clean_sheet"]
                 + s.w_defcon * pf["c_defcon"] + s.w_saves * pf["c_saves"]
                 + s.w_bonus * pf["c_bonus"])
    return pf


def pct_within(values: pd.Series, groups: pd.Series, mask: pd.Series) -> pd.Series:
    """Percentile (0-100) within each group, computed only over rows where mask is True."""
    out = pd.Series(np.nan, index=values.index)
    v = values.where(mask)
    ranked = v.groupby(groups).rank(pct=True) * 100
    out[mask] = ranked[mask]
    return out


# ============================================================ confidence =====
def data_confidence(df: pd.DataFrame, strength_source: str) -> pd.DataFrame:
    """0-100 heuristic: sample size (50%), minutes stability (30%), input coverage (20%).

    Sample size counts current-season minutes plus HALF of up to
    CONFIDENCE_PAST_CAP (900) effective past-season minutes, so a proven past
    season raises confidence a little without letting it substitute for current
    form. Not a statistical confidence interval: it says how much data stands
    behind a row.
    """
    past_bonus_min = (df["past_minutes_eff"].fillna(0).clip(upper=CONFIDENCE_PAST_CAP) * 0.5
                      if "past_minutes_eff" in df.columns else 0.0)
    sample = ((df["season_minutes"].fillna(0) + past_bonus_min) / 900.0).clip(0, 1)
    stability = (1 - df["r_minutes_sd"] / 45.0).clip(0, 1).fillna(0.3)
    inputs = pd.concat([
        df["history_available"].astype(float),
        df["season_xg"].notna().astype(float),
        df["ownership"].notna().astype(float),
        df["price"].notna().astype(float),
        pd.Series(1.0 if strength_source.startswith("xG") else 0.5, index=df.index),
    ], axis=1).mean(axis=1)
    score = 100 * (0.5 * sample + 0.3 * stability + 0.2 * inputs)
    label = np.select([score >= 70, score >= 45], ["High", "Medium"], "Low")
    return pd.DataFrame({"confidence": score.round(0), "confidence_label": label}, index=df.index)


# =============================================================== compute =====
@dataclass
class Results:
    players: pd.DataFrame        # one row per player, every metric
    fixture_detail: pd.DataFrame # one row per player per upcoming fixture
    strength: StrengthModel
    outlook: pd.DataFrame        # team fixture outlook over the horizon
    baselines: pd.DataFrame
    gws: list[int]
    next_gw: int | None
    unscheduled: pd.DataFrame    # postponed / unscheduled fixtures


def _fixture_label(rows: pd.DataFrame, team_short: dict) -> str:
    if rows.empty:
        return "Blank"
    return ", ".join(f"{team_short.get(o, '?')} ({'H' if h else 'A'})"
                     for o, h in zip(rows["opp_id"], rows["is_home"]))


def compute(t: Tables, s: Settings) -> Results:
    df = t.players.copy()
    tm = team_match_table(t.fixtures)

    season = summarise_log(player_match_log(df, t.history, tm)).add_prefix("s_")
    recent = summarise_log(player_match_log(df, t.history, tm, s.recent_matches)).add_prefix("r_")
    df = df.merge(season, left_on="player_id", right_index=True, how="left")
    df = df.merge(recent, left_on="player_id", right_index=True, how="left")
    # Players whose team has no completed matches yet get 0 team matches, not NaN.
    for c in ["s_team_matches", "r_team_matches"]:
        df[c] = df[c].fillna(0)

    base = position_baselines(df)

    # Past-season prior: one row per player with the newest usable past season
    # (NaN columns when none), merged before blended_rates uses it.
    try:
        cur_year = int(str(t.season).split("/")[0])
    except (ValueError, TypeError, AttributeError):
        cur_year = None
    past_sel = select_past_season(t.history_past, cur_year, s.past_season_decay)
    if s.use_past_seasons and not past_sel.empty:
        df = df.merge(past_sel, left_on="player_id", right_index=True, how="left")
    # Display columns exist even with no past data: NaN, not 0
    for c in ["past_minutes", "past_xg90", "past_xa90", "past_bonus90"]:
        if c not in df.columns:
            df[c] = np.nan
    if "past_xg" in df.columns:
        df["past_xg90"] = per90(df["past_xg"], df["past_minutes"])
        df["past_xa90"] = per90(df["past_xa"], df["past_minutes"])
        df["past_bonus90"] = per90(df["past_bonus"], df["past_minutes"])

    df = pd.concat([df, blended_rates(df, base, s)], axis=1)

    # Raw (unshrunk) rates for display alongside the blended ones
    for stat, tot in [("xg", "season_xg"), ("xa", "season_xa"), ("xgi", "season_xgi"),
                      ("xgc", "season_xgc")]:
        df[f"{stat}90_raw_season"] = per90(df[tot], df["season_minutes"])
        df[f"{stat}90_raw_recent"] = per90(df[f"r_{stat}"], df["r_minutes"])
    df["bonus90_raw_season"] = per90(df["s_bonus"], df["season_minutes"])
    df["bps90"] = per90(df["bps_total"], df["season_minutes"])  # display only, not scored
    df["xgi90_blend"] = df["xg90_blend"] + df["xa90_blend"]  # display only, not scored

    # Past-season transferability flag: joined the current club at/after the start
    # of the current season build-up, so past-season rates describe another club
    # (and history_past records no position, so the current one is assumed).
    join_year = pd.to_datetime(df["team_join_date"], errors="coerce").dt.year \
        if "team_join_date" in df.columns else pd.Series(np.nan, index=df.index)
    has_past = df["past_minutes_eff"].fillna(0) > 0 if "past_minutes_eff" in df.columns \
        else pd.Series(False, index=df.index)
    df["new_to_club"] = (join_year.notna() & (join_year >= cur_year)).fillna(False)
    df["past_transfer_flag"] = df["new_to_club"] & has_past

    df["attack90"] = attack_index(df["xg90_blend"], df["xa90_blend"], df["position"], t.scoring)
    df = pd.concat([df, minutes_outlook(df)], axis=1)
    av = [availability_factor(st, ch) for st, ch in zip(df["status"], df["chance_next"])]
    df["avail_next"], df["avail_later"] = zip(*av) if av else ([], [])

    # Fixtures
    next_gw = t.next_gw
    max_gw = int(t.events["event"].max()) if not t.events.empty else 38
    gws = [g for g in range(next_gw, next_gw + s.horizon_gws) if g <= max_gw] if next_gw else []
    model = team_strength(team_xg_by_match(t.history, tm), s)
    outlook = upcoming_fixture_outlook(t.fixtures, model, gws, s)
    detail = fixture_components(df, outlook, next_gw or -1, t.scoring, s)

    comp = ["c_attack", "c_clean_sheet", "c_defcon", "c_saves", "c_bonus", "raw"]
    if not detail.empty:
        nxt = detail[detail["event"] == next_gw].groupby("player_id")[comp].sum().add_suffix("_next")
        hor = detail.groupby("player_id")[comp].sum().add_suffix("_horizon")
        nfx = detail[detail["event"] == next_gw].groupby("player_id").size().rename("n_fixtures_next")
        nfh = detail.groupby("player_id").size().rename("n_fixtures_horizon")
        df = df.merge(pd.concat([nxt, hor, nfx, nfh], axis=1), left_on="player_id",
                      right_index=True, how="left")
    for c in [f"{x}_next" for x in comp] + [f"{x}_horizon" for x in comp] + \
             ["n_fixtures_next", "n_fixtures_horizon"]:
        if c not in df:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)  # no fixture rows = blank gameweek(s): genuinely zero

    # Fixture labels
    team_short = dict(zip(t.teams["team_id"], t.teams["team"]))
    by_team = {k: v for k, v in outlook.groupby("team_id")} if not outlook.empty else {}
    def labels(team_id):
        rows = by_team.get(team_id, outlook.iloc[0:0])
        nxt = _fixture_label(rows[rows["event"] == next_gw], team_short)
        parts = [f"{g}: {_fixture_label(rows[rows['event'] == g], team_short)}" for g in gws]
        att = rows.loc[rows["event"] == next_gw, "att_mult"]
        cs = rows.loc[rows["event"] == next_gw, "cs_prob"]
        return nxt, " | ".join(parts), (att.mean() if len(att) else np.nan), \
            (cs.mean() if len(cs) else np.nan)
    lab = {tid: labels(tid) for tid in df["team_id"].dropna().unique()}
    default = ("Blank", "", np.nan, np.nan)
    for i, col in enumerate(["next_fixture", "horizon_fixtures", "next_att_mult", "next_cs_prob"]):
        df[col] = [lab.get(x, default)[i] for x in df["team_id"]]

    # Scores (percentiles within position over the eligible pool)
    eligible = (df["season_minutes"].fillna(0) >= s.min_season_minutes) & (df["status"] != "u")
    df["eligible"] = eligible
    df["score_next"] = pct_within(df["raw_next"], df["position"], eligible)
    df["score_horizon"] = pct_within(df["raw_horizon"], df["position"], eligible)
    df["value_metric"] = df["raw_horizon"] / df["price"]
    df["value_score"] = pct_within(df["value_metric"], df["position"], eligible)
    df["attack_pct"] = pct_within(df["attack90"], df["position"], eligible)
    df["attack_rank"] = df["attack90"].where(eligible).groupby(df["position"]).rank(
        ascending=False, method="min")
    df["pool_size"] = df["position"].map(eligible.groupby(df["position"]).sum())

    # Form vs underlying (recent window)
    cs_pts = df["position"].map(t.scoring["clean_sheets"]).astype(float)
    dc_pts = df["position"].map(t.scoring["defensive_contribution"]).astype(float)
    r_att = attack_index(df["xg90_recent"], df["xa90_recent"], df["position"], t.scoring)
    df["underlying_recent90"] = (
        r_att + cs_pts * np.exp(-df["xgc90_recent"].fillna(df["xgc90_season"]))
        + (dc_pts * df["defcon_rate_recent"]).fillna(0)
        + np.where(df["position"] == "GKP", df["saves90_recent"].fillna(0) / 3.0, 0.0)
    )
    df.loc[df["r_minutes"].isna(), "underlying_recent90"] = np.nan
    df["points_recent90"] = per90(df["r_points"], df["r_minutes"])
    form_pool = eligible & (df["r_minutes"].fillna(0) >= s.min_recent_minutes)
    df["form_pool"] = form_pool
    df["underlying_pct"] = pct_within(df["underlying_recent90"], df["position"], form_pool)
    df["points_pct"] = pct_within(df["points_recent90"], df["position"], form_pool)
    df["form_gap"] = df["underlying_pct"] - df["points_pct"]
    df["goals_minus_xg"] = df["season_goals"] - df["season_xg"]
    df["assists_minus_xa"] = df["season_assists"] - df["season_xa"]

    df = pd.concat([df, data_confidence(df, model.source)], axis=1)
    df["is_differential"] = (
        (df["ownership"] < s.ownership_threshold)
        & (df["score_horizon"] >= s.differential_min_score)
        & (df["xmins"] >= s.differential_min_xmins)
        & eligible
    )

    unscheduled = t.fixtures[t.fixtures["event"].isna() & ~t.fixtures["completed"]].copy()
    unscheduled["home"] = unscheduled["team_h"].map(team_short)
    unscheduled["away"] = unscheduled["team_a"].map(team_short)

    return Results(df, detail, model, outlook, base, gws, next_gw, unscheduled)
