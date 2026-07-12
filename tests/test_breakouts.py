"""P5b breakout detector — pure hermetic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.breakouts import EXPECTED_RATE, breakout_rate, detect_breakouts


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
    assert bo.loc[1, "magnitude"] == 0.05 - 0.02       # up past p90
    assert bo.loc[2, "magnitude"] == -0.02 - (-0.05)   # p10 - realized (down)
    assert bo.loc[0, "magnitude"] == 0.0


def test_calibrated_rate_near_expected():
    # 100 rows, exactly 20 outside the band → rate 0.20.
    n = 100
    realized = np.zeros(n)
    realized[:10] = 0.05      # up breakouts
    realized[10:20] = -0.05   # down breakouts
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
