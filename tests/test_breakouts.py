"""P5b breakout detector — pure hermetic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.baselines import RandomWalk
from pythia.breakouts import (
    EXPECTED_RATE,
    breakout_rate,
    build_breakouts_response,
    detect_breakouts,
    run_breakout_scan,
)


def _series(vals, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="B")
    return pd.Series(vals, index=idx)


def test_detect_up_down_none():
    realized = _series([0.0, 0.05, -0.05, 0.01])
    p10 = _series([-0.02, -0.02, -0.02, -0.02])
    p90 = _series([0.02, 0.02, 0.02, 0.02])
    bo = detect_breakouts(realized, p10, p90)
    assert list(bo["direction"]) == ["none", "up", "down", "none"]
    assert list(bo["exceeded"]) == [False, True, True, False]
    # magnitude = distance past the breached edge
    assert bo.loc[1, "magnitude"] == 0.05 - 0.02  # up past p90
    assert bo.loc[2, "magnitude"] == -0.02 - (-0.05)  # p10 - realized (down)
    assert bo.loc[0, "magnitude"] == 0.0


def test_calibrated_rate_near_expected():
    # 100 rows, exactly 20 outside the band → rate 0.20.
    n = 100
    realized = np.zeros(n)
    realized[:10] = 0.05  # up breakouts
    realized[10:20] = -0.05  # down breakouts
    r = _series(realized)
    p10 = _series(np.full(n, -0.02))
    p90 = _series(np.full(n, 0.02))
    bo = detect_breakouts(r, p10, p90)
    rate = breakout_rate(bo, window=n)
    assert abs(rate.breakout_rate - 0.20) < 1e-9
    assert rate.expected == EXPECTED_RATE
    assert rate.n == n


def test_rolling_window_uses_tail():
    n = 50
    realized = np.zeros(n)
    realized[-5:] = 0.05  # last 5 are breakouts
    r = _series(realized)
    p10 = _series(np.full(n, -0.02))
    p90 = _series(np.full(n, 0.02))
    bo = detect_breakouts(r, p10, p90)
    # window=10 → 5 of last 10 exceeded → 0.5
    assert abs(breakout_rate(bo, window=10).breakout_rate - 0.5) < 1e-9


def test_empty_is_safe():
    bo = detect_breakouts(_series([]), _series([]), _series([]))
    assert bo.empty
    rate = breakout_rate(bo)
    assert rate.n == 0 and np.isnan(rate.breakout_rate)


def _price_frame(n=400):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # deterministic pseudo-random walk (no Math.random / np.random needed)
    steps = np.array([((i * 37) % 11 - 5) / 500.0 for i in range(n)])
    close = 400.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"QQQ_close": close}, index=idx)


def test_scan_is_oos_and_has_schema():
    scan = run_breakout_scan(
        _price_frame(),
        lambda: RandomWalk("QQQ_close"),
        target_col="QQQ_close",
        model_version="vtest",
        initial_train=120,
        eval_size=40,
    )
    assert not scan.empty
    assert list(scan.columns) == [
        "model_version",
        "symbol",
        "ts",
        "horizon",
        "direction",
        "realized",
        "p10",
        "p90",
        "exceeded",
        "magnitude",
        "oos",
    ]
    assert scan["oos"].all()
    assert (scan["model_version"] == "vtest").all()
    # direction only ever up/down/none; exceeded iff not none
    assert set(scan["direction"].unique()) <= {"up", "down", "none"}
    assert (scan["exceeded"] == (scan["direction"] != "none")).all()


def test_response_badge_and_diagnostic_note():
    scan = run_breakout_scan(
        _price_frame(),
        lambda: RandomWalk("QQQ_close"),
        target_col="QQQ_close",
        model_version="vtest",
        initial_train=120,
        eval_size=40,
    )
    resp = build_breakouts_response(scan, window=20)
    assert resp["expected_rate"] == EXPECTED_RATE
    assert resp["badge"] in ("green", "amber")
    assert 0.0 <= resp["rate"] <= 1.0
    assert "not a trade signal" in resp["note"].lower()
    # events carry the audit fields
    for ev in resp["events"]:
        assert ev["direction"] in ("up", "down")
        assert "ts" in ev and "magnitude" in ev


def test_response_empty_scan_is_safe():
    resp = build_breakouts_response(pd.DataFrame(), window=20)
    assert resp["rate"] is None
    assert resp["lifetime_rate"] is None
    assert resp["badge"] == "amber"
    assert resp["events"] == []


def test_response_reports_lifetime_and_recent_separately():
    # A structurally-calibrated band (~20% lifetime) that breaches heavily in
    # the RECENT window must read as recent-drift, NOT structurally broken.
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    exceeded = np.zeros(n, dtype=bool)
    exceeded[:36] = True  # ~20% lifetime breach spread in the early part
    exceeded[-18:] = True  # last 18 of 20 breach → recent rate ~0.9
    scan = pd.DataFrame(
        {
            "model_version": "v",
            "symbol": "QQQ",
            "ts": ts,
            "horizon": 1,
            "direction": np.where(exceeded, "up", "none"),
            "realized": 0.0,
            "p10": -0.01,
            "p90": 0.01,
            "exceeded": exceeded,
            "magnitude": 0.0,
            "oos": True,
        }
    )
    resp = build_breakouts_response(scan, window=20)
    assert resp["rate"] > 0.30  # recent window is hot
    assert 0.10 <= resp["lifetime_rate"] <= 0.30  # lifetime is calibrated
    assert resp["lifetime_calibrated"] is True
    # verdict must name BOTH so recent-drift isn't mistaken for a broken band
    v = resp["verdict"].lower()
    assert "lifetime" in v and "recent" in v
    assert "structurally calibrated" in v or "recent vol-regime drift" in v


# --- /breakouts serve wiring (P5b): scorecard from report_json['breakouts'] ---
from fastapi.testclient import TestClient  # noqa: E402

from test_registry_serve import _sqlite_register, _sqlite_registry  # noqa: E402
from pythia.serve.app import create_app  # noqa: E402


def _client_with(report_json: dict) -> TestClient:
    reg = _sqlite_registry()
    _sqlite_register(
        reg,
        model_name="tft_lite_daily_qqq",
        model_version="v1",
        dataset_hash="a" * 64,
        report_json=report_json,
        artifact_uri="local://x",
        git_sha="feedface",
    )
    return TestClient(create_app(registry=reg))


def test_breakouts_route_serves_block_when_present():
    block = {
        "expected_rate": 0.20,
        "window": 20,
        "rate": 0.18,
        "n": 20,
        "badge": "green",
        "verdict": "band calibrated",
        "events": [],
    }
    client = _client_with({"tft_lite_daily_qqq": {"coverage_80": 0.79}, "breakouts": block})
    data = client.get("/breakouts").json()
    assert data["breakouts"] == block
    assert data["model_version"] == "v1"


def test_breakouts_route_null_when_absent():
    client = _client_with({"tft_lite_daily_qqq": {"coverage_80": 0.79}})
    data = client.get("/breakouts").json()
    assert data["breakouts"] is None  # graceful-empty; panel hides scorecard
