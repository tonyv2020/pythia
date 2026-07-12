"""P5a range backtest — the harness scores the realized-range target via
``target_fn`` + range baselines, leak-free, no torch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.baselines import LastRange, RollingRange
from pythia.features.targets import realized_range_target


def _frame(n: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    close = 400.0 + np.cumsum([((i * 17) % 7 - 3) / 50.0 for i in range(n)])
    spread = np.array([1.0 + ((i * 13) % 5) / 5.0 for i in range(n)])
    return pd.DataFrame(
        {"QQQ_close": close, "QQQ_high": close + spread, "QQQ_low": close - spread},
        index=idx,
    )


def _range_fn(frame: pd.DataFrame) -> pd.Series:
    # Reindex to the FULL frame index (NaN where undefined) so the harness's
    # per-split dropna handles edges — matches the default target's contract.
    r = realized_range_target(frame["QQQ_high"], frame["QQQ_low"])
    return r.reindex(frame.index)


def test_range_backtest_scores_target_and_baselines():
    df = _frame()
    splits = list(expanding_walk_forward(df.index, initial_train_size=60, eval_size=20))
    reports = run_backtest(
        df, "QQQ_close", splits,
        {"last_range": lambda: LastRange("QQQ_high", "QQQ_low"),
         "rolling_range": lambda: RollingRange("QQQ_high", "QQQ_low")},
        rw_name="last_range",
        target_fn=_range_fn,
    )
    for name in ("last_range", "rolling_range"):
        r = reports[name]
        assert r.n_eval_obs > 0
        assert np.isfinite(r.crps)
        assert np.isfinite(r.coverage_80)
    # rolling_range is skill-scored vs the last_range floor
    assert reports["rolling_range"].mae_skill_vs_rw is not None


def test_range_target_is_positive_and_matches_log_hl():
    df = _frame(30)
    r = _range_fn(df).dropna()
    assert (r > 0).all()
    # spot-check one row equals log(high/low)
    i = df.index[5]
    assert r.loc[i] == np.log(df.loc[i, "QQQ_high"] / df.loc[i, "QQQ_low"])


def test_target_fn_none_is_price_return_default():
    # Sanity: target_fn=None still runs the default price path (no range cols
    # needed). Uses a plain close-only frame.
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    df = pd.DataFrame({"QQQ_close": 400.0 + np.arange(120) * 0.1}, index=idx)
    splits = list(expanding_walk_forward(df.index, initial_train_size=60, eval_size=20))
    from pythia.baselines import RandomWalk
    reports = run_backtest(df, "QQQ_close", splits,
                           {"random_walk": lambda: RandomWalk("QQQ_close")})
    assert reports["random_walk"].n_eval_obs > 0
