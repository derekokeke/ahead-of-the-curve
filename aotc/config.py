"""All tunable numbers live here, so you can change behaviour without hunting through code.

Anything that the FPL API itself exposes (points per goal, clean-sheet points, etc.)
is read from the live `game_config.scoring` block at runtime instead of being hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

BASE_URL = "https://fantasy.premierleague.com/api/"

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]

# --- Defensive contributions (DefCon) ---------------------------------------------
# The API gives the points value (2) in game_config.scoring.defensive_contribution,
# but NOT the action thresholds. These come from the official Premier League article
# "How do players score defensive contribution points in Fantasy?" (20 Jul 2026),
# which confirms the 2025/26 mechanic continues unchanged in 2026/27:
#   DEF: 10+ clearances, blocks, interceptions and tackles (CBIT)  -> 2 pts
#   MID/FWD: 12+ CBIT plus ball recoveries (CBIRT)                 -> 2 pts
#   Capped at 2 pts per match. Goalkeepers are not eligible.
# The per-match `defensive_contribution` field in the API is already the
# position-appropriate action count (CBIT for DEF, CBIRT for MID/FWD).
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DEFCON_SOURCE = (
    "premierleague.com/en/news/4361991 (20 Jul 2026): DEF 10 CBIT, MID/FWD 12 CBIRT, "
    "2 pts, capped per match"
)

# Fallback scoring if game_config.scoring is missing from the API response.
# These match the 2026/27 values observed in the live API on 16 Nov 2026.
FALLBACK_SCORING = {
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assists": 3,
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "saves": 1,  # per 3 saves
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
}

# --- Availability ---------------------------------------------------------------
# FPL's chance_of_playing_next_round (0/25/50/75/100) is a club-news flag, NOT a
# calibrated probability of starting. We map it to a deliberately coarse haircut
# on minutes. Edit these if you disagree.
AVAILABILITY_HAIRCUT = {100: 1.0, 75: 0.8, 50: 0.5, 25: 0.2, 0: 0.0}
# For injured/suspended players, gameweeks after the next one get this factor
# (structured return dates are not in the API, so this is an explicit assumption).
LATER_GW_FACTOR_IF_OUT = 0.5

# --- Fixture-difficulty fallback ------------------------------------------------
# Used only if we cannot build a team-strength model from match xG
# (e.g. very early season or player histories failed to load).
FDR_ATTACK_MULTIPLIER = {1: 1.30, 2: 1.15, 3: 1.00, 4: 0.85, 5: 0.70}
DEFAULT_LEAGUE_XG = 1.35  # typical PL team xG per match, used only by the fallback

# Minimum league minutes for a player to help define a position's baseline rate.
BASELINE_MIN_MINUTES = 450
# Minutes in a team match covered by player rows needed to trust summed team xG.
TEAM_XG_MIN_COVERAGE = 0.8

# --- Past seasons (element-summary history_past) ----------------------------------
# Phase 0 evidence (live API, 16 Nov 2026, checked on Raya/Gabriel/Gross):
# history_past rows ALWAYS contain the expected_* and starts keys, but before
# 2022/23 they are zero placeholders (Raya 2021/22: 2,160 minutes yet
# starts = 0 and expected_goals_conceded = "0.00"; real values appear from
# 2022/23). So availability is a SEASON-based rule, not a zero-value rule:
# a goalkeeper can legitimately have 0 xG in a season where the field exists.
PAST_XG_FIRST_SEASON = "2022/23"
# How many recent seasons may serve as a prior, and the decay of the most
# recent one (squared for the season before that). Set by the user brief.
PAST_SEASONS_USED = 2

# --- Bonus and market ------------------------------------------------------------
# Bonus partly overlaps attacking returns (big xG/xA days usually earn bonus),
# so its default weight is deliberately below the other components.
# Bonus is NOT shrunk toward the past season by default: the Bonus Points System
# was rebalanced for 2026/27, so older bonus rates may not transfer.
# Set-piece orders and market signals are context only: they never change scores.
CONFIDENCE_PAST_CAP = 900.0     # max past minutes counted in the confidence sample
MARKET_HEAVY_TRANSFERS = 100_000  # net transfers this GW for "Heavy transfers in/out"
MARKET_OWN_TREND = 50_000       # manager-count change over 3 GWs for Rising/Falling


@dataclass(frozen=True)
class Settings:
    """User-adjustable settings. The sidebar in app.py edits a copy of these."""

    # Windows and filters
    recent_matches: int = 5          # last N completed TEAM league matches
    horizon_gws: int = 4             # look-ahead for the multi-week ranking
    min_season_minutes: int = 270    # below this a player gets no opportunity score
    min_recent_minutes: int = 180    # needed for the form-vs-underlying comparison

    # Differentials
    ownership_threshold: float = 10.0
    differential_min_score: float = 60.0
    differential_min_xmins: float = 60.0

    # Shrinkage (explained in Methodology): "phantom" minutes at a baseline rate
    prior_minutes_season: float = 450.0   # season rate shrinks toward position baseline
    prior_minutes_recent: float = 270.0   # recent rate shrinks toward player's season rate
    prior_apps_defcon: float = 3.0        # DefCon hit-rate shrinkage, in 60+ min apps
    recent_weight: float = 0.4            # final blend: recent vs season

    # Component weights (1.0 = take the component at face value)
    w_attack: float = 1.0
    w_clean_sheet: float = 1.0
    w_defcon: float = 1.0
    w_saves: float = 1.0
    fixture_strength: float = 1.0   # 0 ignores opponents; 1 = full model effect

    # Team strength shrinkage, in matches of league-average performance
    team_prior_matches: float = 4.0

    # Past-season prior (history_past): shrink season rates toward the player's
    # own most recent usable past rate instead of straight to the baseline.
    use_past_seasons: bool = True
    past_season_decay: float = 0.5    # weight of last season's minutes; squared for two ago
    bonus_use_past: bool = False      # BPS rebalanced for 2026/27: past bonus off by default

    # Bonus as a fifth component (points-equivalents per 90 x expected minutes).
    w_bonus: float = 0.5              # < 1 because bonus overlaps attacking returns

    def to_dict(self) -> dict:
        return asdict(self)
