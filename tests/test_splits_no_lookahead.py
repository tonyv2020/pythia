"""Every generated split must be leak-free: train_end <= eval_start."""

from __future__ import annotations

import pandas as pd
import pytest

from pythia.backtest.splits import (
    WalkForwardSplit,
    expanding_walk_forward,
    rolling_walk_forward,
    slice_train_eval,
)


@pytest.fixture
def daily_index() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=300, freq="D")


def test_split_dataclass_rejects_reversed_windows(daily_index) -> None:
    with pytest.raises(ValueError):
        WalkForwardSplit(
            train_start=daily_index[10],
            train_end=daily_index[5],
            eval_start=daily_index[5],
            eval_end=daily_index[15],
        )


def test_expanding_splits_are_leak_free(daily_index) -> None:
    splits = list(expanding_walk_forward(daily_index, initial_train_size=100, eval_size=20))
    assert splits, "expected at least one split"
    for s in splits:
        assert s.train_start == daily_index[0]
        assert s.train_end == s.eval_start
        assert s.eval_end > s.eval_start


def test_rolling_splits_have_fixed_train_size(daily_index) -> None:
    splits = list(rolling_walk_forward(daily_index, train_size=100, eval_size=20))
    assert splits, "expected at least one split"
    for s in splits:
        # 100 rows between train_start and train_end (approx by day count).
        assert (s.train_end - s.train_start).days == 100


def test_slice_train_eval_returns_disjoint_frames(daily_index) -> None:
    frame = pd.DataFrame({"y": range(len(daily_index))}, index=daily_index)
    (split,) = list(expanding_walk_forward(daily_index, initial_train_size=200, eval_size=100))[:1]
    train, ev = slice_train_eval(frame, split)
    assert train.index.max() < ev.index.min()
    assert set(train.index).isdisjoint(set(ev.index))


def test_empty_when_frame_too_short() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    assert list(expanding_walk_forward(idx, initial_train_size=100, eval_size=1)) == []


def test_non_monotonic_index_rejected() -> None:
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
    with pytest.raises(ValueError):
        list(expanding_walk_forward(idx, initial_train_size=1, eval_size=1))
