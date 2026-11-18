"""Checks for the calculations that are easiest to get subtly wrong.

Run with:  python -m pytest -q
"""
import copy
import math

import numpy as np
import pandas as pd
import pytest

from aotc.config import Settings
from aotc.demo_data import make_demo_raw
from aotc.metrics import (availability_factor, compute, per90, player_match_log,
                          shrink_per90, summarise_log, team_match_table)
from aotc.transform import build_tables, detect_gameweeks


@pytest.fixture(scope="module")
def raw():
    return make_demo_raw()


@pytest.fixture(scope="module")
def tables(raw):
    return build_tables(raw)


@pytest.fixture(scope="module")
def results(tables):
    return compute(tables, Settings())


# ----------------------------------------------------------------- per 90 ----
def test_per90_basic():
    assert per90(1, 45) == pytest.approx(2.0)
    assert per90(0.5, 90) == pytest.approx(0.5)


def test_per90_zero_minutes_is_unavailable_not_zero():
    assert math.isnan(per90(0, 0))
    s = per90(pd.Series([1.0, 0.0]), pd.Series([90.0, 0.0]))
    assert s.iloc[0] == pytest.approx(1.0)
    assert math.isnan(s.iloc[1])


# --------------------------------------------------------------- shrinkage ---
def test_small_sample_is_pulled_toward_baseline():
    # A sub scores from 1.0 xG in 20 minutes: raw 4.5 xG/90.
    raw_rate = per90(1.0, 20)
    shrunk = shrink_per90(1.0, 20, prior_per90=0.30, prior_minutes=450)
    assert raw_rate == pytest.approx(4.5)
    assert 0.30 < shrunk < 0.6  # (1.0 + 0.3*5) / ((20+450)/90) = 0.479


def test_large_sample_stays_close_to_observed():
    shrunk = shrink_per90(18.0, 2700, prior_per90=0.30, prior_minutes=450)  # 0.60 xG/90 raw
    assert shrunk == pytest.approx((18 + 1.5) / 35, rel=1e-6)
    assert abs(shrunk - 0.60) < 0.05


def test_missing_observation_stays_missing():
    assert math.isnan(shrink_per90(np.nan, 0, 0.3, 450))


def test_low_minute_subs_do_not_top_rankings(results):
    df = results.players
    subs = df[df["name"].str.contains("SUB")]
    assert (subs["season_minutes"] < 270).all()
    assert subs["score_next"].isna().all(), "below min minutes => no opportunity score"
    # Even with the minutes filter switched off, expected minutes keep them down the list.
    loose = compute(build_tables(make_demo_raw()), Settings(min_season_minutes=1))
    mids = loose.players[loose.players["position"] == "MID"]
    top5 = mids.nlargest(5, "raw_next")["name"]
    assert not top5.str.contains("SUB").any()


# ----------------------------------------------------------- missing values --
def test_missing_history_gives_nan_not_zero(raw):
    broken = copy.deepcopy(raw)
    victim = next(k for k, v in broken["histories"].items() if len(v) >= 5)
    del broken["histories"][victim]
    t = build_tables(broken)
    r = compute(t, Settings())
    row = r.players.set_index("player_id").loc[int(victim)]
    assert not row["history_available"]
    assert math.isnan(row["r_minutes"]) and math.isnan(row["r_xg"])
    assert row["season_minutes"] > 0  # bootstrap totals are still known


def test_missing_api_fields_become_nan(raw):
    broken = copy.deepcopy(raw)
    for e in broken["bootstrap"]["elements"]:
        e.pop("expected_goals_conceded", None)
        e.pop("selected_by_percent", None)
    t = build_tables(broken)
    assert t.players["season_xgc"].isna().all()
    assert t.players["ownership"].isna().all()
    compute(t, Settings())  # must not crash


def test_missed_team_matches_count_in_recent_window(tables):
    tm = team_match_table(tables.fixtures)
    injured = tables.players.loc[tables.players["status"] == "i", "player_id"].iloc[0]
    # Demo striker is injured from GW5; his team's last 4 matches are GW5-8.
    summ = summarise_log(player_match_log(tables.players, tables.history, tm, last_n=4))
    row = summ.loc[injured]
    assert row["team_matches"] == 4
    assert row["minutes"] == 0 and row["starts"] == 0  # missed matches count as 0, not dropped
    # Season-long, he still has minutes from GW1-4.
    season = summarise_log(player_match_log(tables.players, tables.history, tm))
    assert season.loc[injured, "minutes"] > 0


# ------------------------------------------------- blank / double / postponed -
def test_double_and_blank_gameweeks(results, tables):
    df = results.players.set_index("player_id")
    nxt = results.next_gw
    dgw_team_players = tables.players[tables.players["team_id"].isin([1, 2])]["player_id"]
    starters = df.loc[dgw_team_players]
    starters = starters[starters["xmins"] > 60]
    assert (starters["n_fixtures_next"] == 2).all()
    assert starters["next_fixture"].str.contains(",").all()
    # Same teams blank in the following gameweek
    gw_after = nxt + 1
    detail = results.fixture_detail
    assert detail[(detail["event"] == gw_after) & detail["team_id"].isin([1, 2])].empty
    assert (starters["n_fixtures_horizon"] == Settings().horizon_gws).all()  # 2 + 0 + 1 + 1


def test_blank_player_scores_zero_next_gw():
    raw = make_demo_raw()
    nxt = next(e["id"] for e in raw["bootstrap"]["events"] if e["is_next"])
    for f in raw["fixtures"]:  # make team 3 blank next GW
        if f["event"] == nxt and 3 in (f["team_h"], f["team_a"]):
            f["event"] = None
    r = compute(build_tables(raw), Settings())
    team3 = r.players[r.players["team_id"] == 3]
    assert (team3["raw_next"] == 0).all()
    assert (team3["next_fixture"] == "Blank").all()


def test_postponed_fixture_excluded_everywhere(tables, results):
    post = tables.fixtures[tables.fixtures["event"].isna() & ~tables.fixtures["completed"]]
    assert len(post) == 1
    fid = post["fixture_id"].iloc[0]
    assert fid not in set(team_match_table(tables.fixtures)["fixture_id"])
    assert fid not in set(results.outlook["fixture_id"])
    assert fid in set(results.unscheduled["fixture_id"])


# ------------------------------------------------------- rules & semantics ---
def test_defcon_threshold_is_position_specific(tables):
    players = pd.DataFrame({"player_id": [1, 2, 3], "team_id": [1, 1, 1],
                            "position": ["DEF", "MID", "GKP"], "history_available": True})
    history = pd.DataFrame({"player_id": [1, 2, 3], "fixture_id": [10, 10, 10],
                            "minutes": [90, 90, 90], "starts": [1, 1, 1],
                            "defensive_contribution": [11, 11, 0]})
    tm = pd.DataFrame({"team_id": [1], "fixture_id": [10], "match_no_desc": [1],
                       "opp_id": [2], "is_home": [True], "event": [1]})
    log = player_match_log(players, history, tm).set_index("player_id")
    assert log.loc[1, "defcon_hit"] == 1.0   # DEF: 11 >= 10
    assert log.loc[2, "defcon_hit"] == 0.0   # MID: 11 < 12
    assert math.isnan(log.loc[3, "defcon_hit"])  # GKP: not eligible


def test_ownership_does_not_change_football_score(raw):
    a = compute(build_tables(raw), Settings()).players.set_index("player_id")
    changed = copy.deepcopy(raw)
    for e in changed["bootstrap"]["elements"]:
        e["selected_by_percent"] = "0.5"
    b = compute(build_tables(changed), Settings()).players.set_index("player_id")
    pd.testing.assert_series_equal(a["score_horizon"], b["score_horizon"])


def test_availability_haircuts():
    assert availability_factor("u", 0) == (0.0, 0.0)
    assert availability_factor("i", 0) == (0.0, 0.5)
    assert availability_factor("d", 75) == (0.8, 1.0)
    assert availability_factor("a", None) == (1.0, 1.0)


def test_gameweek_detection_with_and_without_flags(tables):
    assert tables.next_gw == 9 and tables.current_gw == 8 and tables.season == "2026/27"
    ev = tables.events.copy()
    ev["is_next"] = False
    assert detect_gameweeks(ev)[0] == 9  # falls back to first unfinished GW


def test_scores_are_0_to_100_within_position(results):
    s = results.players.dropna(subset=["score_horizon"])
    assert s["score_horizon"].between(0, 100).all()
    assert (s.groupby("position")["score_horizon"].max() == 100).all()
