"""Return + realized-range target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.features.targets import realized_range_target, return_target


def test_return_target_is_log_diff() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    px = pd.Series([100.0, 101.0, 100.5, 102.0, 101.0], index=idx, name="QQQ")
    r = return_target(px)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(101.0 / 100.0))
    assert r.iloc[3] == pytest.approx(np.log(102.0 / 100.5))


def test_realized_range_positive_and_zero_on_flat() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    high = pd.Series([102.0, 100.0, 105.0], index=idx, name="QQQ")
    low = pd.Series([98.0, 100.0, 101.0], index=idx, name="QQQ")
    r = realized_range_target(high, low)
    assert r.iloc[0] == pytest.approx(np.log(102.0 / 98.0))
    assert r.iloc[1] == pytest.approx(0.0)
    assert (r >= 0).all()


def test_realized_range_masks_inverted_bars() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    high = pd.Series([100.0, 100.0], index=idx)
    low = pd.Series([101.0, 99.0], index=idx)  # first is inverted
    r = realized_range_target(high, low)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(100.0 / 99.0))
