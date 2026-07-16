"""P3 raptor baselines — hermetic (injected p_move / tilt fixtures, no DB).

Covers the calibrated-dispersion p_move baseline (mean=0, sigma=c*p_move with
c fit to ~0.80 train coverage) and the OLS-tilt direction baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pythia.baselines import RaptorDirection, RaptorPMove, RaptorPMoveStub
from pythia.baselines.raptor_p_move import _Z80, _forward_log_returns


def _bars(n: int = 120):
    idx = pd.date_range("2026-06-08 09:30", periods=n, freq="10min")
    steps = np.array([((i * 29) % 13 - 6) / 800.0 for i in range(n)])
    px = 400.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"QQQ_close": px}, index=idx), idx


def _pmove(idx) -> pd.Series:
    # p_move loosely tracks |local move|, kept in (0,1).
    vals = np.array([0.05 + 0.5 * (((i * 17) % 7) / 7.0) for i in range(len(idx))])
    return pd.Series(vals, index=idx)


def _tilt(idx) -> pd.Series:
    vals = np.array([(((i * 11) % 5) - 2) / 4.0 for i in range(len(idx))])
    return pd.Series(vals, index=idx)


def test_pmove_predict_is_zero_mean_positive_sigma():
    df, idx = _bars()
    m = RaptorPMove("QQQ_close", _pmove(idx), horizon=3)
    m.fit(df.iloc[:90])
    fc = m.predict(idx[90:110])
    assert (fc.mean == 0).all()
    assert (fc.sigma > 0).all()
    assert len(fc.mean) == 20


def test_pmove_calibrates_to_80pct_train_coverage():
    df, idx = _bars()
    train = df.iloc[:100]
    m = RaptorPMove("QQQ_close", _pmove(idx), horizon=1)
    m.fit(train)
    # Reconstruct train sigma and check empirical P10-P90 coverage ≈ 0.80.
    r = _forward_log_returns(train["QQQ_close"], 1)
    pm = _pmove(idx).reindex(train.index)
    d = pd.concat([r.rename("r"), pm.rename("pm")], axis=1).dropna()
    d = d[d["pm"] > 1e-6]
    sigma = m._c * d["pm"]
    covered = (d["r"].abs() <= _Z80 * sigma).mean()
    assert abs(covered - 0.80) < 0.06


def test_pmove_missing_eval_pmove_uses_fallback():
    df, idx = _bars()
    pm = _pmove(idx).copy()
    m = RaptorPMove("QQQ_close", pm, horizon=1)
    m.fit(df.iloc[:90])
    # Eval bars with NO p_move history at all.
    future = pd.date_range(idx[-1] + pd.Timedelta("10min"), periods=5, freq="10min")
    fc = m.predict(future)
    assert (fc.sigma > 0).all()  # fell back to train-median p_move, not NaN


def test_pmove_nearzero_tail_does_not_explode_sigma():
    """A fat near-zero p_move tail must NOT blow up the c calibration (the live
    CRPS-45x bug). With calib_floor, near-zero rows are dropped from fit +
    floored in predict, so sigma stays bounded."""
    df, idx = _bars(160)
    pm = _pmove(idx).copy()
    # Inject a near-zero tail on a third of the bars.
    pm.iloc[::3] = 1e-5
    m = RaptorPMove("QQQ_close", pm, horizon=3, calib_floor=0.02)
    m.fit(df.iloc[:120])
    fc = m.predict(idx[120:150])
    # Sigma must be finite and in a sane band (not the exploded 0.1+ scale).
    assert np.isfinite(fc.sigma).all()
    assert fc.sigma.max() < 0.1


def test_pmove_sparse_pmove_degrades_to_flat_sigma_not_crash():
    """When a train window lacks enough p_move>=floor rows, RaptorPMove must
    DEGRADE to a flat RW-style sigma (never crash the whole backtest — the live
    GPU-run blocker: first split had only 16 qualifying rows)."""
    df, idx = _bars(n=80)
    # min_train_rows unreachable → forces the degrade path.
    m = RaptorPMove("QQQ_close", _pmove(idx), horizon=1, min_train_rows=1000)
    m.fit(df.iloc[:60])  # must NOT raise
    assert m._c is None  # degraded mode
    fc = m.predict(idx[60:75])
    assert (fc.sigma > 0).all()
    # Flat mode → constant sigma across eval bars.
    assert float(fc.sigma.std()) == 0.0


def test_pmove_stub_still_raises():
    m = RaptorPMoveStub("QQQ_close")
    m.fit(pd.DataFrame({"QQQ_close": [1.0, 2.0]}))
    try:
        m.predict(pd.Index([0]))
        assert False
    except NotImplementedError:
        pass


def test_direction_beta_and_predict():
    df, idx = _bars()
    m = RaptorDirection("QQQ_close", _tilt(idx), horizon=3)
    m.fit(df.iloc[:90])
    fc = m.predict(idx[90:105])
    assert np.isfinite(m._beta)
    assert (fc.sigma > 0).all()
    # mean = beta * tilt on eval
    exp = m._beta * _tilt(idx).reindex(idx[90:105]).to_numpy()
    assert np.allclose(fc.mean.to_numpy(), exp)


def test_direction_zero_tilt_gives_zero_beta():
    df, idx = _bars()
    flat = pd.Series(0.0, index=idx)
    m = RaptorDirection("QQQ_close", flat, horizon=1)
    m.fit(df.iloc[:90])
    assert m._beta == 0.0
    fc = m.predict(idx[90:100])
    assert (fc.mean == 0).all()
