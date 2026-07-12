"""Conformal calibration wrapper (helen D18): scales dispersion so train
coverage ~= 0.80, mean untouched, no-op when calibration data is thin."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from pythia.backtest.harness import _target_returns
from pythia.backtest.protocols import Model, ProbForecast
from pythia.models.conformal import ConformalScaledModel


class _FakeBase(Model):
    """Predicts mean=0, a fixed sigma — a deliberately mis-sized cone the
    conformal layer must right-size."""

    encoder_length = 0

    def __init__(self, sigma: float):
        self.sigma = sigma

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.zeros(n), index=eval_index),
            sigma=pd.Series(np.full(n, self.sigma), index=eval_index),
        )


def _frame(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2026-06-08 09:30", periods=n, freq="10min")
    steps = np.array([((i * 41) % 23 - 11) / 900.0 for i in range(n)])
    px = 400.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"QQQ_close": px}, index=idx)


def test_conformal_makes_train_coverage_80():
    df = _frame(200)
    y = _target_returns(df, "QQQ_close", 1).dropna()
    m = ConformalScaledModel(_FakeBase(0.01), "QQQ_close", horizon=1, coverage=0.80)
    m.fit(df)
    z80 = float(norm.ppf(0.90))
    scaled_sigma = m._s * 0.01
    cov = float((np.abs(y.to_numpy()) <= z80 * scaled_sigma).mean())
    assert abs(cov - 0.80) < 0.03  # calibrated by construction on train


def test_conformal_mean_is_untouched_and_sigma_scaled():
    df = _frame(200)
    m = ConformalScaledModel(_FakeBase(0.02), "QQQ_close", horizon=1)
    m.fit(df)
    fc = m.predict(df.index[50:60])
    assert (fc.mean == 0).all()                      # dispersion-only
    assert np.allclose(fc.sigma.to_numpy(), m._s * 0.02)


def test_conformal_forward_horizon_target():
    df = _frame(200)
    m3 = ConformalScaledModel(_FakeBase(0.01), "QQQ_close", horizon=3, coverage=0.80)
    m3.fit(df)
    # scale should calibrate the 3-bar (forward) target, not the 1-bar one.
    y3 = _target_returns(df, "QQQ_close", 3).dropna()
    z80 = float(norm.ppf(0.90))
    cov = float((np.abs(y3.to_numpy()) <= z80 * m3._s * 0.01).mean())
    assert abs(cov - 0.80) < 0.04


def test_conformal_thin_calibration_is_noop():
    df = _frame(30)
    m = ConformalScaledModel(_FakeBase(0.01), "QQQ_close", horizon=1, min_cal_rows=1000)
    m.fit(df)
    assert m._s == 1.0
    fc = m.predict(df.index[:5])
    assert np.allclose(fc.sigma.to_numpy(), 0.01)


def test_conformal_target_fn_calibrates_custom_target():
    """P5a: with target_fn, conformal calibrates train coverage on THAT target
    (e.g. realized-range), not the price return."""
    from pythia.features.targets import realized_range_target

    class _RangeBase(Model):
        encoder_length = 0
        def fit(self, train): pass
        def predict(self, idx):
            n = len(idx)
            return ProbForecast(mean=pd.Series(np.full(n, 0.02), index=idx),
                                sigma=pd.Series(np.full(n, 0.001), index=idx))

    df = _frame(200)
    # give it high/low so realized_range_target computes
    df = df.assign(QQQ_high=df["QQQ_close"] + 1.0, QQQ_low=df["QQQ_close"] - 1.0)
    def rfn(f):
        return realized_range_target(f["QQQ_high"], f["QQQ_low"]).reindex(f.index)
    m = ConformalScaledModel(base=_RangeBase(), target_col="QQQ_close",
                             horizon=1, coverage=0.80, target_fn=rfn)
    m.fit(df)
    y = rfn(df).dropna()
    z80 = float(norm.ppf(0.90))
    scaled = m._s * np.maximum(0.001, m._floor)
    cov = float((np.abs(y.to_numpy() - 0.02) <= z80 * scaled).mean())
    assert abs(cov - 0.80) < 0.05  # calibrated on the RANGE target by construction
