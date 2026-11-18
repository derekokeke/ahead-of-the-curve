"""Fictional demo data in the same shape as the live FPL API.

Used by the tests and by `AOTC_DEMO=1 streamlit run app.py`. Every team and player
here is invented; the app shows a clear banner when demo data is in use.

Deliberate edge cases:
  * GW9 (next GW): teams 1 and 2 play twice (double gameweek)
  * GW10: teams 1 and 2 blank (their fixture was moved into GW9)
  * one fixture postponed (event = None, not finished)
  * an injured player, a doubtful (75%) player, a player who left the club
  * a super-sub with tiny minutes, a rotation player, a backup keeper with 0 minutes
  * xG fields as strings, chance_of_playing as null, like the real API
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

TEAM_NAMES = [("Northbridge", "NOR"), ("Kestrel Vale", "KES"), ("Saltmarsh", "SAL"),
              ("Ironford", "IRO"), ("Bramblecourt", "BRA"), ("Linden Park", "LIN")]

SCORING = {
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assists": 3, "saves": 1,
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
}
POS_ID = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
# role: (position, p_start, sub_p, xg_w, xa_w, defcon_mean)
ROLES = [
    ("GK1", "GKP", 1.0, 0.0, 0.00, 0.00, 0),
    ("GK2", "GKP", 0.0, 0.0, 0.00, 0.00, 0),
    ("CB1", "DEF", 1.0, 0.0, 0.05, 0.02, 11),
    ("CB2", "DEF", 0.95, 0.0, 0.04, 0.02, 9),
    ("FB1", "DEF", 0.95, 0.0, 0.03, 0.10, 6),
    ("FB2", "DEF", 0.7, 0.25, 0.03, 0.08, 6),
    ("DM", "MID", 1.0, 0.0, 0.04, 0.06, 12),
    ("CM", "MID", 1.0, 0.0, 0.08, 0.10, 9),
    ("AM", "MID", 0.95, 0.0, 0.26, 0.24, 6),
    ("WM", "MID", 0.9, 0.05, 0.16, 0.18, 6),
    ("SUB", "MID", 0.0, 0.7, 0.20, 0.12, 4),
    ("ST1", "FWD", 0.95, 0.0, 0.38, 0.10, 4),
    ("ST2", "FWD", 0.55, 0.35, 0.22, 0.08, 4),
]


def _round_robin(n):
    teams = list(range(1, n + 1))
    rounds = []
    for r in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs if r % 2 == 0 else [(b, a) for a, b in pairs])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def make_demo_raw(seed: int = 7, completed_rounds: int = 8, total_rounds: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    n = len(TEAM_NAMES)
    att = {t: rng.uniform(0.75, 1.3) for t in range(1, n + 1)}
    dfn = {t: rng.uniform(0.75, 1.3) for t in range(1, n + 1)}
    start = datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)

    rr = _round_robin(n)
    schedule = []  # (round, home, away)
    for r in range(1, total_rounds + 1):
        pairs = rr[(r - 1) % len(rr)]
        if ((r - 1) // len(rr)) % 2 == 1:
            pairs = [(b, a) for a, b in pairs]
        schedule += [(r, h, a) for h, a in pairs]
    # DGW / blank: move the team-1 v team-2 game nearest to GW10 into GW9
    nxt = completed_rounds + 1
    for i, (r, h, a) in enumerate(schedule):
        if r == nxt + 1 and {h, a} == {1, 2}:
            schedule[i] = (nxt, h, a)
            break
    else:  # ensure the pairing exists in GW10 so the edge case is always present
        schedule = [(r, h, a) for r, h, a in schedule if not (r == nxt + 1 and 1 in (h, a))]
        schedule.append((nxt, 1, 2))
    # Postpone one completed-round fixture (team 5 home)
    postponed_idx = next(i for i, (r, h, a) in enumerate(schedule)
                         if r == completed_rounds - 1 and (h == 5 or a == 5))

    players = []
    pid = 1
    for t in range(1, n + 1):
        for role, pos, p_start, sub_p, xg_w, xa_w, dcm in ROLES:
            players.append(dict(id=pid, team=t, role=role, pos=pos, p_start=p_start, sub_p=sub_p,
                                xg_w=xg_w, xa_w=xa_w, dcm=dcm))
            pid += 1
    injured = next(p for p in players if p["team"] == 3 and p["role"] == "ST1")
    doubtful = next(p for p in players if p["team"] == 4 and p["role"] == "AM")
    departed = next(p for p in players if p["team"] == 6 and p["role"] == "WM")

    histories = {p["id"]: [] for p in players}
    fixtures = []
    for fid, (r, h, a) in enumerate(schedule, start=1):
        kickoff = start + timedelta(days=7 * (r - 1), hours=2 + (fid % 3))
        postponed = fid - 1 == postponed_idx
        played = r <= completed_rounds and not postponed
        fx = dict(id=fid, code=1000 + fid, event=None if postponed else r, finished=played,
                  finished_provisional=played, started=played, kickoff_time=kickoff.isoformat()
                  .replace("+00:00", "Z"), team_h=h, team_a=a, team_h_score=None,
                  team_a_score=None, team_h_difficulty=int(np.clip(round(3 * dfn[a] * 1.1), 2, 5)),
                  team_a_difficulty=int(np.clip(round(3 * dfn[h]), 2, 5)), stats=[])
        if played:
            team_xg = {h: 1.35 * att[h] * dfn[a] * 1.08, a: 1.35 * att[a] * dfn[h] * 0.92}
            goals = {h: 0, a: 0}
            rows = {}
            for p in [p for p in players if p["team"] in (h, a)]:
                if p is injured and r > 4:
                    mins = 0
                elif p is departed and r > 3:
                    mins = 0
                elif rng.random() < p["p_start"]:
                    mins = 90 if rng.random() < 0.75 else int(rng.integers(55, 85))
                elif rng.random() < p["sub_p"]:
                    mins = int(rng.integers(8, 30))
                else:
                    mins = 0
                starts = int(mins >= 45 and p["p_start"] > 0 and mins >= 55)
                share = mins / 90
                xg = team_xg[p["team"]] * p["xg_w"] * share
                xa = team_xg[p["team"]] * p["xa_w"] * share
                g = int(rng.poisson(xg)) if mins else 0
                ast = int(rng.poisson(xa)) if mins else 0
                goals[p["team"]] += g
                rows[p["id"]] = dict(p=p, mins=mins, starts=starts, xg=xg, xa=xa, g=g, a=ast)
            fx["team_h_score"], fx["team_a_score"] = goals[h], goals[a]
            for prow in rows.values():
                p, mins = prow["p"], prow["mins"]
                if mins == 0:
                    continue  # the real API also returns rows for 0 mins; absence is handled too
                opp = a if p["team"] == h else h
                conceded = goals[opp]
                cs = int(conceded == 0 and mins >= 60)
                dc = int(rng.poisson(p["dcm"] * mins / 90))
                saves = int(rng.poisson(3)) if p["pos"] == "GKP" else 0
                thr = {"DEF": 10, "MID": 12, "FWD": 12}.get(p["pos"])
                pts = (2 if mins >= 60 else 1) + prow["g"] * SCORING["goals_scored"][p["pos"]] \
                    + prow["a"] * 3 + cs * SCORING["clean_sheets"][p["pos"]] \
                    + (2 if thr and dc >= thr else 0) + saves // 3
                histories[p["id"]].append(dict(
                    element=p["id"], fixture=fid, opponent_team=opp, total_points=pts,
                    was_home=p["team"] == h, kickoff_time=fx["kickoff_time"], round=r,
                    minutes=mins, goals_scored=prow["g"], assists=prow["a"], clean_sheets=cs,
                    goals_conceded=conceded if mins else 0, saves=saves, bonus=0,
                    starts=prow["starts"], expected_goals=f"{prow['xg']:.2f}",
                    expected_assists=f"{prow['xa']:.2f}",
                    expected_goal_involvements=f"{prow['xg'] + prow['xa']:.2f}",
                    expected_goals_conceded=f"{team_xg[opp] * mins / 90:.2f}",
                                        defensive_contribution=dc))
                # Market/bonus context rows (live API also sends these per match)
                b = int((prow["g"] + prow["a"]) > 0)
                hist = histories[p["id"]]
                hist[-1].update(
                    bonus=b, bps=int(pts * 2 + 10),
                    selected=int(300000 + 500000 * (p["xg_w"] + p["xa_w"])),
                    transfers_balance=int(rng.integers(-4000, 6000)),
                    value=int(round(10 * {"GKP": 4.5, "DEF": 4.8, "MID": 5.5, "FWD": 6.0}[p["pos"]])))
        fixtures.append(fx)

    elements = []
    for p in players:
        hist = histories[p["id"]]
        tot = lambda k: sum(float(x[k]) for x in hist)
        status, chance, news = "a", None, ""
        if p is injured:
            status, chance, news = "i", 0, "Hamstring injury - Expected back in 3 weeks"
        elif p is doubtful:
            status, chance, news = "d", 75, "Knock - 75% chance of playing"
        elif p is departed:
            status, chance, news = "u", 0, "Has joined another club permanently"
        base_price = {"GKP": 4.5, "DEF": 4.8, "MID": 5.5, "FWD": 6.0}[p["pos"]]
        price = base_price + 4 * (p["xg_w"] + p["xa_w"]) * att[p["team"]] + rng.uniform(-0.3, 0.6)
        own = float(np.clip(rng.lognormal(1.2, 1.1) * (1 + 3 * (p["xg_w"] + p["xa_w"])), 0.1, 65))
        elements.append(dict(
            id=p["id"], web_name=f"{TEAM_NAMES[p['team'] - 1][1].title()} {p['role']}",
            first_name="Demo", second_name=f"{p['role']} {p['team']}", team=p["team"],
            element_type=POS_ID[p["pos"]], now_cost=int(round(price * 10)),
            selected_by_percent=f"{own:.1f}", status=status,
            chance_of_playing_next_round=chance, news=news,
            minutes=int(tot("minutes")), starts=int(tot("starts")),
            goals_scored=int(tot("goals_scored")), assists=int(tot("assists")),
            expected_goals=f"{tot('expected_goals'):.2f}",
            expected_assists=f"{tot('expected_assists'):.2f}",
            expected_goal_involvements=f"{tot('expected_goal_involvements'):.2f}",
            expected_goals_conceded=f"{tot('expected_goals_conceded'):.2f}",
            saves=int(tot("saves")), total_points=int(tot("total_points")),
            defensive_contribution=int(tot("defensive_contribution")),
            penalties_order=1 if p["role"] == "ST1" else None,
            corners_order=1 if (p["role"] == "AM" and p["team"] == 1)
            else (2 if (p["role"] == "WM" and p["team"] == 1) else None),
            frees_order=1 if (p["role"] == "CM" and p["team"] == 1) else None,
            team_join_date="2026-07-20" if (p["role"] == "WM" and p["team"] == 2)
            else "2025-07-01",
            transfers_in_event=int(20000 * (1 + 3 * (p["xg_w"] + p["xa_w"]))),
            transfers_out_event=int(rng.integers(1000, 20000)),
            cost_change_event=1 if (p["role"] == "ST1" and p["team"] == 1)
            else (-1 if p["role"] == "GK2" else 0),
            cost_change_start=int(rng.integers(-4, 5)),
        ))

    events = []
    for r in range(1, total_rounds + 1):
        events.append(dict(id=r, name=f"Gameweek {r}",
                           deadline_time=(start + timedelta(days=7 * (r - 1))).isoformat()
                           .replace("+00:00", "Z"),
                           finished=r <= completed_rounds, is_current=r == completed_rounds,
                           is_next=r == completed_rounds + 1))
        bootstrap = dict(
        events=events,
        teams=[dict(id=i + 1, name=nm, short_name=sh) for i, (nm, sh) in enumerate(TEAM_NAMES)],
        elements=elements,
        element_types=[dict(id=v, singular_name_short=k) for k, v in POS_ID.items()],
        total_players=11_000_000,
        game_config=dict(settings=dict(static_content_url=".../2026_27/"), scoring=SCORING),
    )

    # --- history_past: a few players with usable past seasons (test fixtures) ---
    def past_row(season, minutes, starts=30, goals=5, assists=4, xg="4.00", xa="3.00",
                 xgc="30.00", saves=0, bonus=8, bps=600, start_cost=55, end_cost=58,
                 points=120):
        row = dict(season_name=season, minutes=minutes, starts=starts,
                   goals_scored=goals, assists=assists, saves=saves, bonus=bonus,
                   bps=bps, start_cost=start_cost, end_cost=end_cost, total_points=points)
        if xg is not None:
            row["expected_goals"] = xg
        if xa is not None:
            row["expected_assists"] = xa
        if xgc is not None:
            row["expected_goals_conceded"] = xgc
        return row

    histories_past: dict[int, list[dict]] = {}
    for p in players:
        if p["role"] == "AM":
            # Strong last season: should lift players with small current samples
            histories_past[p["id"]] = [
                past_row("2025/26", minutes=2900, goals=9, assists=8, xg="12.00",
                         xa="11.00", xgc="35.00", bonus=18, bps=760, points=178),
                past_row("2024/25", minutes=2400, goals=6, assists=6, xg="8.00",
                         xa="7.00", xgc="33.00", bonus=11, bps=620, points=140),
            ]
        elif p["role"] == "GK1" and p["team"] == 1:
            # Keeper: 0 xG is legitimate in a season where the field exists
            # (2024/25+), so it must stay 0, never become NaN
            histories_past[p["id"]] = [
                past_row("2025/26", minutes=3330, goals=0, assists=1, xg="0.00",
                         xa="0.05", xgc="27.56", saves=110, bonus=11, bps=633, points=162)]
        elif p["role"] == "CM" and p["team"] == 2:
            # Only a pre-2022/23 season: xG/starts are API placeholders there, and
            # the season is outside the 2-season window anyway
            histories_past[p["id"]] = [past_row("2021/22", minutes=2033)]
        elif p["role"] == "WM" and p["team"] == 2:
            # Joined the current club for this season: prior must be flagged
            histories_past[p["id"]] = [
                past_row("2025/26", minutes=2600, goals=7, assists=5, xg="9.00",
                         xa="6.00", xgc="30.00", bonus=10, bps=650, points=150)]
        elif p["role"] == "ST1" and p["team"] == 5:
            # One row with the expected_goals key missing entirely
            histories_past[p["id"]] = [
                past_row("2025/26", minutes=2500, goals=10, assists=3, xg=None,
                         xa="2.00", xgc="28.00", bonus=9, bps=560, points=130)]

    hist_out = {str(k): v for k, v in histories.items()
                if next(e for e in elements if e["id"] == k)["minutes"] > 0}
    return dict(bootstrap=bootstrap, fixtures=fixtures, histories=hist_out,
                histories_past={str(k): v for k, v in histories_past.items()},
                meta=dict(source="demo", refreshed_at=datetime.now(timezone.utc)
                          .isoformat(timespec="seconds"),
                          warnings=["Demo data: every team and player is fictional."],
                          histories_expected=len(hist_out), histories_missing=[]))
