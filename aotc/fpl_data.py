"""Fetching raw FPL data: HTTP, disk cache, retries and a last-good snapshot.

Nothing in here does football maths. It returns raw JSON plus a `meta` dict that
records where the data came from and when, so the UI can be honest about it.

Order of preference for each endpoint:
  1. fresh disk cache (younger than its TTL)  -> no network call
  2. live API                                  -> written to cache
  3. stale disk cache                          -> used with a warning
  4. last-good snapshot                        -> used with a warning
"""
from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import truststore  # use the OS trust store (e.g. Windows cert store). Some
    # corporate networks re-sign TLS via a proxy CA that certifi does not ship,
    # which fails every request with CERTIFICATE_VERIFY_FAILED.
except ImportError:  # pragma: no cover - optional dependency
    truststore = None

from .config import BASE_URL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_FILE = DATA_DIR / "snapshot.json.gz"

TIMEOUT = (5, 30)          # (connect, read) seconds
MAX_WORKERS = 6            # concurrent element-summary requests: polite, not hammering
TTL_SECONDS = {
    "bootstrap": 30 * 60,
    "fixtures": 30 * 60,
    "summary": 6 * 60 * 60,
}
# Bump when the cached per-player payload changes shape (e.g. v2 added
# history_past). Cached items with a different/missing version are treated as
# stale and re-fetched once, then stored in the new format.
SCHEMA_VERSION = 2
# Only these history_past fields are kept, to keep the cache small.
HISTORY_PAST_KEEP = [
    "season_name", "minutes", "starts", "goals_scored", "assists",
    "expected_goals", "expected_assists", "expected_goals_conceded", "saves",
    "bonus", "bps", "start_cost", "end_cost", "total_points",
]


class DataUnavailable(RuntimeError):
    """Raised when neither the API, the cache nor a snapshot can provide data."""


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def make_session() -> requests.Session:
    if truststore is not None:
        truststore.inject_into_ssl()
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,  # waits 0s, 2s, 4s between attempts
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "ahead-of-the-curve/0.1 (personal FPL scouting dashboard)"
    return session


def _get_json(session: requests.Session, path: str):
    resp = session.get(BASE_URL + path, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()  # raises ValueError on HTML error pages


# ---------------------------------------------------------------- disk cache ---
def _read_gz(path: Path):
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None  # corrupt cache is treated as no cache


def _write_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(path)  # atomic: never leaves a half-written file behind


def _cached_endpoint(session, name: str, path: str, force: bool, warnings: list):
    """Return (data, fetched_at, source) for a single JSON endpoint."""
    cache_path = CACHE_DIR / f"{name}.json.gz"
    cached = _read_gz(cache_path)
    if cached and not force and _now() - cached["fetched_at"] < TTL_SECONDS[name]:
        return cached["data"], cached["fetched_at"], "cache"
    try:
        data = _get_json(session, path)
        ts = _now()
        _write_gz(cache_path, {"fetched_at": ts, "data": data})
        return data, ts, "live"
    except (requests.RequestException, ValueError) as exc:
        if cached:
            warnings.append(
                f"Could not refresh {name} ({exc.__class__.__name__}); "
                f"using cached copy from {_iso(cached['fetched_at'])}."
            )
            return cached["data"], cached["fetched_at"], "stale cache"
        raise


def _last_finished_gw(bootstrap: dict) -> int:
    finished = [e["id"] for e in bootstrap.get("events", []) if e.get("finished")]
    return max(finished) if finished else 0


def _load_histories(session, bootstrap: dict, force: bool, warnings: list,
                    progress: Callable[[float], None] | None):
    """Fetch element-summary history for every player who has league minutes.

    Players with zero minutes have nothing to fetch: their history is legitimately empty.
    """
    needed = [e["id"] for e in bootstrap.get("elements", []) if (e.get("minutes") or 0) > 0]
    cache_path = CACHE_DIR / "summaries.json.gz"
    cache = _read_gz(cache_path) or {"items": {}}
    items: dict = cache.get("items", {})
    last_gw = _last_finished_gw(bootstrap)

    def is_stale(item) -> bool:
        return (
            force
            or item.get("v") != SCHEMA_VERSION  # old cache format: refresh once
            or _now() - item["fetched_at"] > TTL_SECONDS["summary"]
            or item.get("gw", -1) < last_gw
        )

    to_fetch = [pid for pid in needed if str(pid) not in items or is_stale(items[str(pid)])]
    failed: list[int] = []
    if to_fetch:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_get_json, session, f"element-summary/{pid}/"): pid
                       for pid in to_fetch}
            for done, fut in enumerate(as_completed(futures), start=1):
                pid = futures[fut]
                try:
                    data = fut.result()
                    items[str(pid)] = {
                        "v": SCHEMA_VERSION,
                        "fetched_at": _now(),
                        "gw": last_gw,
                        # history_past/fixtures are trimmed; keep the cache small
                        "history": data.get("history", []),
                        "history_past": [
                            {k: row.get(k) for k in HISTORY_PAST_KEEP if k in row}
                            for row in data.get("history_past", [])
                        ],
                    }
                except (requests.RequestException, ValueError):
                    failed.append(pid)
                if progress:
                    progress(done / len(futures))
        _write_gz(cache_path, {"items": items})

    histories = {str(pid): items[str(pid)]["history"] for pid in needed if str(pid) in items}
    histories_past = {str(pid): items[str(pid)].get("history_past", [])
                      for pid in needed if str(pid) in items}
    missing = [pid for pid in needed if str(pid) not in items]
    stale_used = [pid for pid in failed if str(pid) in items]
    if stale_used:
        warnings.append(f"{len(stale_used)} player histories could not refresh; older copies used.")
    if missing:
        warnings.append(
            f"{len(missing)} player histories are unavailable. Their recent-form stats "
            "show as unavailable (not zero)."
        )
    return histories, len(needed), missing, histories_past


# ------------------------------------------------------------------ snapshot ---
def save_snapshot(raw: dict) -> None:
    _write_gz(SNAPSHOT_FILE, raw)


def load_snapshot() -> dict | None:
    return _read_gz(SNAPSHOT_FILE)


def _from_snapshot(reason: str) -> dict:
    snap = load_snapshot()
    if snap is None:
        raise DataUnavailable(
            f"{reason} No local snapshot exists yet, so there is nothing to fall back to. "
            "Check your connection and press Refresh."
        )
    meta = snap.setdefault("meta", {})
    meta["source"] = "snapshot"
    meta.setdefault("warnings", [])
    meta["warnings"] = [f"{reason} Showing the last good snapshot from {meta.get('refreshed_at', 'unknown time')}."]
    return snap


# --------------------------------------------------------------- entry point ---
def load_data(force_refresh: bool = False, offline: bool = False,
              progress: Callable[[float], None] | None = None) -> dict:
    """Return {"bootstrap", "fixtures", "histories", "meta"}.

    meta = {source, refreshed_at, warnings, histories_expected, histories_missing}
    """
    if offline:
        return _from_snapshot("Offline mode.")

    warnings: list[str] = []
    session = make_session()
    try:
        bootstrap, bs_ts, bs_src = _cached_endpoint(session, "bootstrap", "bootstrap-static/",
                                                    force_refresh, warnings)
        fixtures, _, fx_src = _cached_endpoint(session, "fixtures", "fixtures/",
                                               force_refresh, warnings)
    except (requests.RequestException, ValueError) as exc:
        return _from_snapshot(f"The FPL API is unreachable ({exc.__class__.__name__}).")

    histories, expected, missing, histories_past = _load_histories(
        session, bootstrap, force_refresh, warnings, progress)
    sources = {bs_src, fx_src}
    source = "live" if sources == {"live"} else ("stale cache" if "stale cache" in sources else "cache")
    raw = {
        "bootstrap": bootstrap,
        "fixtures": fixtures,
        "histories": histories,
        "histories_past": histories_past,
        "meta": {
            "source": source,
            "refreshed_at": _iso(bs_ts),
            "warnings": warnings,
            "histories_expected": expected,
            "histories_missing": missing,
        },
    }
    # Only keep a snapshot that is essentially complete.
    if expected == 0 or len(missing) / expected < 0.05:
        save_snapshot(raw)
    return raw
