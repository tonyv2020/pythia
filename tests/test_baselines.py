"""Baselines: fit → predict shape + honesty of the RaptorPMoveStub."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.baselines import LastReturn, RandomWalk, RaptorPMoveStub


@pytest.fixture
def synthetic_price_frame() -> pd.DataFrame:
    # Deterministic drift-plus-noise so RandomWalk fit gets a stable sigma.
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    r = rng.normal(loc=0.0002, scale=0.01, size=len(idx))
    px = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame({"QQQ_close": px}, index=idx)


def test_random_walk_fits_and_returns_zero_mean(synthetic_price_frame) -> None:
    m = RandomWalk("QQQ_close")
    m.fit(synthetic_price_frame)
    eval_idx = synthetic_price_frame.index[-10:]
    fc = m.predict(eval_idx)
    assert (fc.mean == 0.0).all()
    assert (fc.sigma > 0).all()
    # P10 should be < 0 and P90 > 0 under Normal(0, sigma).
    p10 = fc.quantile(0.10)
    p90 = fc.quantile(0.90)
    assert (p10 < 0).all()
    assert (p90 > 0).all()


def test_last_return_predicts_last_observed(synthetic_price_frame) -> None:
    m = LastReturn("QQQ_close")
    m.fit(synthetic_price_frame)
    eval_idx = synthetic_price_frame.index[-5:]
    fc = m.predict(eval_idx)
    # All means identical (persistence forecast).
    assert (fc.mean == fc.mean.iloc[0]).all()


def test_random_walk_rejects_short_train() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    frame = pd.DataFrame({"QQQ_close": [100.0] * 10}, index=idx)
    with pytest.raises(RuntimeError):
        RandomWalk("QQQ_close", min_train_rows=30).fit(frame)


def test_raptor_stub_raises_on_predict(synthetic_price_frame) -> None:
    m = RaptorPMoveStub("QQQ_close")
    m.fit(synthetic_price_frame)  # ok
    with pytest.raises(NotImplementedError):
        m.predict(synthetic_price_frame.index[-3:])
