"""Checks for the fetch layer using a fake API (no network needed)."""
import pytest
import requests

from aotc import fpl_data
from aotc.demo_data import make_demo_raw


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(fpl_data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(fpl_data, "SNAPSHOT_FILE", tmp_path / "snapshot.json.gz")
    return tmp_path


def fake_api(raw, fail_ids=(), calls=None, down=False):
    def _get(session, path):
        if calls is not None:
            calls.append(path)
        if down:
            raise requests.ConnectionError("network down")
        if path == "bootstrap-static/":
            return raw["bootstrap"]
        if path == "fixtures/":
            return raw["fixtures"]
        pid = int(path.split("/")[1])
        if pid in fail_ids:
            raise requests.HTTPError("503")
        return {"history": raw["histories"].get(str(pid), [])}
    return _get


def test_fetch_then_cache_hit(isolated, monkeypatch):
    raw, calls = make_demo_raw(), []
    monkeypatch.setattr(fpl_data, "_get_json", fake_api(raw, calls=calls))
    first = fpl_data.load_data()
    assert first["meta"]["source"] == "live"
    n_first = len(calls)
    second = fpl_data.load_data()
    assert len(calls) == n_first, "second load should be served entirely from cache"
    assert second["meta"]["source"] == "cache"
    assert fpl_data.SNAPSHOT_FILE.exists()


def test_partial_history_failure_is_reported_not_zeroed(isolated, monkeypatch):
    raw = make_demo_raw()
    bad = int(next(iter(raw["histories"])))
    monkeypatch.setattr(fpl_data, "_get_json", fake_api(raw, fail_ids={bad}))
    out = fpl_data.load_data()
    assert str(bad) not in out["histories"]
    assert bad in out["meta"]["histories_missing"]
    assert any("unavailable" in w for w in out["meta"]["warnings"])


def test_api_down_uses_snapshot(isolated, monkeypatch):
    raw = make_demo_raw()
    monkeypatch.setattr(fpl_data, "_get_json", fake_api(raw))
    fpl_data.load_data()
    for f in (fpl_data.CACHE_DIR).glob("*.json.gz"):
        f.unlink()  # remove cache so only the snapshot can help
    monkeypatch.setattr(fpl_data, "_get_json", fake_api(raw, down=True))
    out = fpl_data.load_data()
    assert out["meta"]["source"] == "snapshot"
    assert out["meta"]["warnings"]


def test_api_down_without_snapshot_raises_clear_error(isolated, monkeypatch):
    monkeypatch.setattr(fpl_data, "_get_json", fake_api({}, down=True))
    with pytest.raises(fpl_data.DataUnavailable, match="No local snapshot"):
        fpl_data.load_data()
