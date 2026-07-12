"""P3 horizon + purge (twin-approved contract): forward-h target, train purge,
eval cap, and the loud guard — no forward leak at the split boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.backtest.harness import _target_returns
from pythia.backtest.protocols import Model, ProbForecast
from pythia.backtest.splits import slice_train_eval
from pythia.baselines import RandomWalk


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2026-06-08 09:30", periods=n, freq="10min")
    steps = np.array([((i * 31) % 17 - 8) / 1000.0 for i in range(n)])
    px = 400.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"QQQ_close": px}, index=idx)


def test_target_returns_h1_is_trailing_one_step():
    df = _frame(10)
    r = _target_returns(df, "QQQ_close", 1)
    px = df["QQQ_close"]
    assert np.isnan(r.iloc[0])
    assert r.iloc[3] == pytest.approx(np.log(px.iloc[3] / px.iloc[2]))


def test_target_returns_h3_is_forward_three_bar():
    df = _frame(10)
    r = _target_returns(df, "QQQ_close", 3)
    px = df["QQQ_close"]
    # y_true[t] == log(px[t+3]/px[t]); last 3 rows NaN.
    assert r.iloc[0] == pytest.approx(np.log(px.iloc[3] / px.iloc[0]))
    assert r.iloc[6] == pytest.approx(np.log(px.iloc[9] / px.iloc[6]))
    assert r.iloc[-3:].isna().all()


class _SpyRW(Model):
    """RandomWalk that records the exact train index it is fit on."""

    def __init__(self, target_col: str, recorder: list):
        self.target_col = target_col
        self.recorder = recorder
        self._sigma = None

    def fit(self, train: pd.DataFrame) -> None:
        self.recorder.append(train.index)
        px = train[self.target_col].astype(float)
        r = np.log(px / px.shift(1)).dropna()
        self._sigma = max(float(r.std(ddof=1)), 1e-9)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.zeros(n), index=eval_index),
            sigma=pd.Series(np.full(n, self._sigma), index=eval_index),
        )


def test_horizon_purge_removes_last_h_minus_1_train_rows():
    df = _frame(120)
    splits = list(expanding_walk_forward(df.index, initial_train_size=40, eval_size=20))
    h = 3
    recorder: list = []
    run_backtest(
        df, "QQQ_close", splits,
        {"random_walk": lambda: _SpyRW("QQQ_close", recorder)},
        horizon=h,
    )
    # For each split, the fit train index must equal the raw train window with
    # its last h-1 rows removed — and must never intersect the eval window.
    for split, seen in zip(splits, recorder):
        raw_train, eval_frame = slice_train_eval(df, split)
        expected = raw_train.index[: -(h - 1)]
        assert list(seen) == list(expected), "fit train != raw train purged by h-1"
        assert set(seen).isdisjoint(set(eval_frame.index))
        # The purged rows are exactly the last h-1 of the raw train window.
        assert list(raw_train.index[-(h - 1):]) == list(raw_train.index[len(expected):])


def test_horizon_gt_eval_size_raises():
    df = _frame(120)
    splits = list(expanding_walk_forward(df.index, initial_train_size=40, eval_size=5))
    with pytest.raises(RuntimeError):
        run_backtest(df, "QQQ_close", splits,
                     {"random_walk": lambda: RandomWalk("QQQ_close")}, horizon=10)


def test_h1_default_is_noop_vs_explicit():
    df = _frame(120)
    splits = list(expanding_walk_forward(df.index, initial_train_size=40, eval_size=20))
    a = run_backtest(df, "QQQ_close", splits,
                     {"random_walk": lambda: RandomWalk("QQQ_close")})
    b = run_backtest(df, "QQQ_close", splits,
                     {"random_walk": lambda: RandomWalk("QQQ_close")}, horizon=1)
    assert a["random_walk"].n_eval_obs == b["random_walk"].n_eval_obs
    assert a["random_walk"].crps == b["random_walk"].crps
