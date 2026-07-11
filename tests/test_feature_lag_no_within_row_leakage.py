"""HARD GATE (helen, P1): assert no within-row observed-covariate leakage.

The existing ``test_splits_no_lookahead`` proves the temporal SPLIT boundary
is clean — a model fit on rows up to ``train_end`` cannot see any row past
``train_end``. That is necessary but NOT SUFFICIENT.

Within a single row t, if we join today's price of a covariate (SPY_close_t)
with today's target return (QQQ_return_t, computed from close_{t-1} → close_t),
we've handed the model a covariate that WAS NOT AVAILABLE when the target's
realisation was decided — a same-bar leak. This test asserts every feature
value at row t is sourced from row <= t - lag (default lag=1).

Method: build a wide frame with a KNOWN AT-VALUE per column (each cell is
a monotone counter). Ask ``build_features`` for the feature frame. For every
observed column, assert that its value at date t equals the value of the
SAME column in the raw frame at date t-lag. Any deviation is a within-row
leak.

We also assert:
  - Known-future calendar columns pass through UNSHIFTED (so ``dow`` at t
    is the actual day-of-week at t).
  - An unclassified stray column raises rather than silently including it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pythia.features.lag import (
    DEFAULT_KNOWN_FUTURE,
    LagPolicy,
    build_features,
    default_policy_for,
)


def _tagged_frame(days: int = 30) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Wide frame with monotone integer tags per column so any lag/no-lag
    behaviour is trivially verifiable."""
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    df = pd.DataFrame(
        {
            "QQQ_close": np.arange(days) * 1.0,
            "QQQ_volume": np.arange(days) * 10.0,
            "SPY_close": np.arange(days) * 0.1,
            "SPY_volume": np.arange(days) * 5.0,
            "dow": [d.weekday() for d in idx],
            "days_to_fomc": np.arange(days) + 0.5,
        },
        index=idx,
    )
    return df, idx


def test_observed_columns_are_lagged_by_one() -> None:
    df, idx = _tagged_frame()
    policy = default_policy_for(df.columns, target_cols={"QQQ_close"})
    # Target column is dropped from features.
    assert "QQQ_close" in policy.targets
    # dow / days_to_fomc are calendar (exempt).
    assert {"dow", "days_to_fomc"}.issubset(policy.known_future)
    # Everything else must be observed.
    assert {"QQQ_volume", "SPY_close", "SPY_volume"}.issubset(policy.observed)

    feat = build_features(df, policy, lag=1)

    # For every observed column, feat's value at date t must equal df's
    # value at date t-1 for the SAME base column.
    for col in policy.observed:
        lag_col = f"{col}_lag1"
        assert lag_col in feat.columns, f"missing lagged column {lag_col}"
        # Compare every row that survived the drop:
        for t in feat.index:
            t_prev = t - pd.tseries.offsets.BDay()
            if t_prev not in df.index:
                continue
            actual = feat.at[t, lag_col]
            expected = df.at[t_prev, col]
            assert actual == pytest.approx(expected), (
                f"WITHIN-ROW LEAK: {lag_col}@{t} = {actual}, expected {expected} "
                f"(= raw {col}@{t_prev})"
            )


def test_known_future_columns_are_not_lagged() -> None:
    df, idx = _tagged_frame()
    policy = default_policy_for(df.columns, target_cols={"QQQ_close"})
    feat = build_features(df, policy, lag=1)

    for col in ("dow", "days_to_fomc"):
        for t in feat.index:
            assert feat.at[t, col] == pytest.approx(df.at[t, col]), (
                f"KNOWN-FUTURE column {col} at {t} was mutated by build_features"
            )


def test_unclassified_column_raises() -> None:
    df, _ = _tagged_frame()
    df["ROGUE_FEATURE"] = 1.0
    # Craft policy that omits ROGUE_FEATURE — the build must refuse.
    policy = LagPolicy(
        observed=frozenset({"QQQ_volume", "SPY_close", "SPY_volume"}),
        known_future=frozenset({"dow", "days_to_fomc"}),
        targets=frozenset({"QQQ_close"}),
    )
    with pytest.raises(ValueError, match="ROGUE_FEATURE"):
        build_features(df, policy)


def test_target_column_is_dropped_from_features() -> None:
    df, _ = _tagged_frame()
    policy = default_policy_for(df.columns, target_cols={"QQQ_close"})
    feat = build_features(df, policy, lag=1)
    # QQQ_close (the target) must not appear as a feature — not lagged, not
    # unlagged. But QQQ_volume_lag1 (a lag of a non-target observed col) is
    # fine.
    assert "QQQ_close" not in feat.columns
    assert "QQQ_close_lag1" not in feat.columns


def test_first_lag_rows_are_dropped() -> None:
    df, idx = _tagged_frame()
    policy = default_policy_for(df.columns, target_cols={"QQQ_close"})
    feat = build_features(df, policy, lag=2)
    # First 2 rows have NaN lagged columns; must be dropped.
    assert feat.index.min() >= idx[2]


def test_policy_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="multiple LagPolicy buckets"):
        LagPolicy(
            observed=frozenset({"X"}),
            known_future=frozenset({"X"}),  # collision
            targets=frozenset(),
        )


def test_calendar_features_all_recognized() -> None:
    """The calendar features produced by pythia.data.calendar_features must
    all be listed in DEFAULT_KNOWN_FUTURE (else they'd get lagged and leak
    time-of-day info into a feature that pretends to be about tomorrow)."""
    from pythia.data.calendar_features import add_calendar_features

    empty = pd.DataFrame(
        {"QQQ_close": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    with_cal = add_calendar_features(empty)
    added = set(with_cal.columns) - {"QQQ_close"}
    missing = added - DEFAULT_KNOWN_FUTURE
    assert not missing, (
        f"pythia.data.calendar_features added columns {missing} not listed in "
        f"DEFAULT_KNOWN_FUTURE — add them so build_features doesn't lag them"
    )
