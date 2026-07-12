"""P3 intraday walk-forward — hermetic end-to-end (fixture bars + p_move/tilt).

Wires assemble-output-shaped bars through the forward-3-bar horizon + purge +
forward-session mask + the four baselines, all offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.backtest.intraday import run_intraday_backtest
from pythia.data.intraday import forward_session_mask


def _multi_session_bars(days: int = 5, bars_per_day: int = 39):
    """~1 session of 10-min bars 09:30–15:50, across ``days`` sessions."""
    stamps = []
    for d in range(days):
        day = pd.Timestamp("2026-06-08") + pd.Timedelta(days=d)
        start = day + pd.Timedelta(hours=9, minutes=30)
        stamps.extend(start + pd.Timedelta(minutes=10 * b) for b in range(bars_per_day))
    idx = pd.DatetimeIndex(stamps)
    steps = np.array([((i * 23) % 19 - 9) / 1500.0 for i in range(len(idx))])
    px = 400.0 * np.exp(np.cumsum(steps))
    df = pd.DataFrame({"QQQ_close": px}, index=idx)
    return df, idx


def _pmove(idx):
    return pd.Series([0.05 + 0.4 * (((i * 13) % 9) / 9.0) for i in range(len(idx))],
                     index=idx)


def _tilt(idx):
    return pd.Series([(((i * 7) % 5) - 2) / 5.0 for i in range(len(idx))], index=idx)


def test_forward_session_mask_excludes_end_of_session_bars():
    df, idx = _multi_session_bars(days=2, bars_per_day=39)
    mask = forward_session_mask(df, horizon=3)
    # Last bar of session 1 (index 38) → forward-3 crosses into session 2 → False.
    assert not mask.iloc[38]
    # A mid-session bar (index 10) → forward-3 stays same day → True.
    assert mask.iloc[10]
    # The final `horizon` bars have no forward target → False.
    assert not mask.iloc[-1]


def test_intraday_backtest_scores_all_four_baselines():
    df, idx = _multi_session_bars(days=6, bars_per_day=39)
    reports = run_intraday_backtest(
        df, "QQQ_close", p_move=_pmove(idx), tilt=_tilt(idx),
        horizon=3, initial_train=80, eval_size=39,
    )
    for name in ("random_walk", "last_return", "raptor_p_move", "raptor_direction"):
        assert name in reports, name
        r = reports[name]
        assert r.n_eval_obs > 0, f"{name} scored 0 obs"
        assert np.isfinite(r.crps)
        assert np.isfinite(r.coverage_80)
    # raptor baselines are skill-scored vs RW like everything else.
    assert reports["raptor_p_move"].mae_skill_vs_rw is not None


def test_intraday_backtest_without_raptor_histories_scores_only_price_baselines():
    df, idx = _multi_session_bars(days=6, bars_per_day=39)
    reports = run_intraday_backtest(
        df, "QQQ_close", p_move=None, tilt=None,
        horizon=3, initial_train=80, eval_size=39,
    )
    assert set(reports) == {"random_walk", "last_return"}
