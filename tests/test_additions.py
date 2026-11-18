"""Checks for the data additions: past-season prior, bonus component, set-piece
context labels, market signals, and the v2 summary cache.

Run with:  python -m pytest -q
"""
import copy

import numpy as np
import pandas as pd
import pytest

from aotc import fpl_data
from aotc.config import Settings
from aotc.demo_data import make_demo_raw
from aotc.explain import market_note, set_pieces_label
from aotc.metrics import compute
from aotc.transform import build_tables

# Every addition switched off: this must reproduce the old behaviour exactly.
OFF = Settings(use_past_seasons=False, w_bonus=0.0)

# Demo roster ids: 13 players per team, ROLES order
# GK1 GK2 CB1 CB2 FB1 FB2 DM CM AM WM SUB ST1 ST2 -> role i (1-based) = (t-1)*13 + i
ID_GK1_T1, ID_AM_T1, ID_SUB_T1, ID_ST1_T1 = 1, 9, 11, 12
ID_CM_T1, ID_CM_T2, ID_WM_T2, ID_ST1_T5 = 7, 21, 23, 64


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(fpl_data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(fpl_data, "SNAPSHOT_FILE", tmp_path / "snapshot.json.gz")
    return tmp_path


@pytest.fixture(scope="module")
def raw():
    return make_demo_raw()


@pytest.fixture(scope="module")
def tables(raw):
    return build_tables(raw)


def _sorted_scores(res, col):
    return res.players.sort_values("player_id")[col].to_numpy(dtype=float)


# ------------------------------------------------------- 1. regression -------
def test_scores_identical_when_additions_off(raw):
    """With past prior and bonus off, history_past data must not move any score."""
    base = compute(build_tables({**raw, "histories_past": {}}), OFF)
    full = compute(build_tables(raw), OFF)
    for col in ["score_next", "score_horizon", "raw_next", "raw_horizon"]:
        assert np.array_equal(_sorted_scores(base, col), _sorted_scores(full, col),
                              equal_nan=True), col


# ---------------------------------------------- 2. past-season prior pull ----
def test_small_sample_pulled_toward_own_past_rate(raw):
    """A tiny current sample leans on the player's own past season; a big one
    barely moves."""
    raw2 = copy.deepcopy(raw)
    raw2["histories_past"][str(ID_SUB_T1)] = [
        {"season_name": "2025/26", "minutes": 3000, "goals_scored": 12,
         "assists": 6, "expected_goals": "14.00", "expected_assists": "6.00",
         "expected_goals_conceded": "30.00", "saves": 0, "bonus": 15, "bps": 700,
         "start_cost": 55, "end_cost": 60, "total_points": 160}]
    t = build_tables(raw2)
    on = compute(t, Settings(w_bonus=0.0)).players.set_index("player_id")
    off = compute(t, OFF).players.set_index("player_id")
    # Small sample: shrinking toward the past rate (~0.42 xG/90) beats the baseline
    sub_shift = on.loc[ID_SUB_T1, "xg90_season"] - off.loc[ID_SUB_T1, "xg90_season"]
    assert sub_shift > 0
        # Larger current sample: the decayed past prior has less influence
    am_shift = abs(on.loc[ID_AM_T1, "xg90_season"] - off.loc[ID_AM_T1, "xg90_season"])
    assert am_shift < sub_shift


def test_player_without_past_season_uses_baseline(tables):
    """No usable past season -> exactly the old (baseline) shrinkage."""
    assert ID_CM_T1 not in tables.history_past["player_id"].values  # no past data
    on = compute(tables, Settings(w_bonus=0.0)).players.set_index("player_id")
    off = compute(tables, OFF).players.set_index("player_id")
    assert on.loc[ID_CM_T1, "xg90_season"] == off.loc[ID_CM_T1, "xg90_season"]


# ------------------------------------- 3. past-field availability rules ------
def test_past_field_rules(raw, tables):
    """Placeholder zeros before 2022/23 become NaN (season rule); a keeper's real
    0 xG stays 0; a missing key stays NaN. compute() must never crash on them."""
    past = tables.history_past
    row = past[past["player_id"] == ID_CM_T2].iloc[0]
    assert row["season"] == "2021/22"
    assert all(pd.isna(row[c]) for c in ["xg", "xa", "xgc", "starts"])
    assert row["minutes"] == 2033 and row["bonus"] == 8
    # Keeper with real minutes and 0 xG in 2025/26: xG stays 0, not NaN
    gk = past[past["player_id"] == ID_GK1_T1].iloc[0]
    assert gk["minutes"] == 3330
    assert gk["xg"] == 0.0
    # A past row with the expected_goals key missing entirely: NaN, no crash
    st = past[past["player_id"] == ID_ST1_T5].iloc[0]
    assert pd.isna(st["xg"]) and st["xa"] == 2.0
    r = compute(tables, Settings())
    assert len(r.players) == len(tables.players)


# -------------------------------------------- 4. context never scores --------
def test_context_fields_never_change_scores(raw, tables):
    """Set-piece ranks, join dates and market fields never move a score."""
    a = compute(tables, OFF)
    raw2 = copy.deepcopy(raw)
    for e in raw2["bootstrap"]["elements"]:
        for k in ["penalties_order", "corners_order", "frees_order"]:
            e[k] = None
        for k in ["transfers_in_event", "transfers_out_event", "cost_change_event",
                  "cost_change_start"]:
            e[k] = 0
        e["team_join_date"] = "2025-07-01"
    b = compute(build_tables(raw2), OFF)
    for col in ["score_next", "score_horizon", "raw_horizon"]:
        assert np.array_equal(_sorted_scores(a, col), _sorted_scores(b, col),
                              equal_nan=True), col
    # ...and the label/note builders stay inert on empty input
    assert set_pieces_label(pd.Series()) == ""
    assert market_note(pd.Series()) == ""


# --------------------------------------- 5. missing fields stay NaN ----------
def test_missing_market_fields_are_nan_not_zero(raw):
    raw2 = copy.deepcopy(raw)
    for e in raw2["bootstrap"]["elements"]:
        for k in ["cost_change_event", "cost_change_start", "transfers_in_event",
                  "transfers_out_event", "team_join_date"]:
            e.pop(k, None)
    for rows in raw2["histories"].values():
        for h in rows:
            h.pop("selected", None)  # no selected column -> trend impossible
    t = build_tables(raw2)
    for c in ["net_transfers_event", "price_change_event_m", "price_since_start_m",
              "own_trend_3gw", "own_trend_pct"]:
        assert t.players[c].isna().all(), c


# ------------------------------------------------ 6. cache format bump -------
def test_old_cache_format_is_refetched(isolated, monkeypatch):
    raw = make_demo_raw()
    calls = []

    def _get(session, path):
        calls.append(path)
        if path == "bootstrap-static/":
            return raw["bootstrap"]
        if path == "fixtures/":
            return raw["fixtures"]
        pid = path.split("/")[1]
        return {"history": raw["histories"].get(pid, []),
                "history_past": raw.get("histories_past", {}).get(pid, [])}

    monkeypatch.setattr(fpl_data, "_get_json", _get)
    fpl_data.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Old v1 item: no "v" field, fresh timestamp, future gw -> stale only by version
    fpl_data._write_gz(fpl_data.CACHE_DIR / "summaries.json.gz",
                       {"items": {str(ID_AM_T1): {"fetched_at": fpl_data._now(), "gw": 99,
                                                  "history": []}}})
    out = fpl_data.load_data()
    assert out["meta"]["source"] == "live"
    assert f"element-summary/{ID_AM_T1}/" in calls  # the old item was re-fetched
    assert out["histories_past"].get(str(ID_AM_T1)) == \
        raw["histories_past"][str(ID_AM_T1)]
    stored = fpl_data._read_gz(fpl_data.CACHE_DIR / "summaries.json.gz")
    assert stored["items"][str(ID_AM_T1)]["v"] == fpl_data.SCHEMA_VERSION


# ------------------------------------------------------ 7. bonus -------------
def test_bonus_component(tables):
    """w_bonus adds a bonus component scaled by minutes; switching it off leaves
    zero-bonus players untouched."""
    on = compute(tables, Settings()).players.set_index("player_id")
    off = compute(tables, Settings(w_bonus=0.0)).players.set_index("player_id")
    # ST1 scores goals -> earns bonus in the demo data
    assert (on.loc[ID_ST1_T1, "bonus90_raw_season"] or 0) > 0
    assert on.loc[ID_ST1_T1, "raw_horizon"] > off.loc[ID_ST1_T1, "raw_horizon"]
    # GK1 makes no goal involvements -> no bonus -> identical raw score
    assert pd.isna(on.loc[ID_GK1_T1, "bonus90_raw_season"]) \
        or on.loc[ID_GK1_T1, "bonus90_raw_season"] == 0
    assert on.loc[ID_GK1_T1, "raw_horizon"] == off.loc[ID_GK1_T1, "raw_horizon"]
    assert "c_bonus" in compute(tables, Settings()).fixture_detail.columns