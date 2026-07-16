"""P3 eval_mask on run_backtest — scoring-time mask, walk-forward geometry
intact. Daily P1 must stay a no-op (eval_mask=None)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.baselines import LastReturn, RandomWalk


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # Deterministic-ish price path (no RNG in tests).
    steps = np.array([((i * 37) % 11 - 5) / 500.0 for i in range(n)])
    px = 100.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"QQQ_close": px}, index=idx)


def _factories():
    return {
        "random_walk": lambda: RandomWalk("QQQ_close"),
        "last_return": lambda: LastReturn("QQQ_close"),
    }


def _splits(df):
    return list(expanding_walk_forward(df.index, initial_train_size=40, eval_size=20))


def test_eval_mask_none_is_noop():
    df = _frame()
    splits = _splits(df)
    a = run_backtest(df, "QQQ_close", splits, _factories())
    b = run_backtest(df, "QQQ_close", splits, _factories(), eval_mask=None)
    assert a["random_walk"].n_eval_obs == b["random_walk"].n_eval_obs
    assert a["random_walk"].mae == b["random_walk"].mae


def test_eval_mask_reduces_scored_obs():
    df = _frame()
    splits = _splits(df)
    full = run_backtest(df, "QQQ_close", splits, _factories())
    # Mask out every other row.
    mask = pd.Series(np.arange(len(df)) % 2 == 0, index=df.index)
    masked = run_backtest(df, "QQQ_close", splits, _factories(), eval_mask=mask)
    assert masked["random_walk"].n_eval_obs < full["random_walk"].n_eval_obs
    assert masked["random_walk"].n_eval_obs > 0
    # skill-vs-rw still computed on the masked scoring set
    assert masked["last_return"].mae_skill_vs_rw is not None


def test_eval_mask_callable_form():
    df = _frame()
    splits = _splits(df)

    # Callable: keep only rows whose position is a multiple of 3.
    def keep(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(np.arange(len(frame)) % 3 == 0, index=frame.index)

    res = run_backtest(df, "QQQ_close", splits, _factories(), eval_mask=keep)
    assert res["random_walk"].n_eval_obs > 0
    assert np.isfinite(res["random_walk"].coverage_80)


def test_eval_mask_all_false_yields_no_obs():
    df = _frame()
    splits = _splits(df)
    mask = pd.Series(False, index=df.index)
    res = run_backtest(df, "QQQ_close", splits, _factories(), eval_mask=mask)
    # No rows scored → empty report with the documented warning.
    assert res["random_walk"].n_eval_obs == 0
