"""Walk-forward split generators (expanding & rolling).

A split is a triple ``(train_end, eval_start, eval_end)`` where
``eval_start = train_end`` (exclusive on the training side, inclusive on
the eval side). A model fit on rows ``< train_end`` MAY NOT observe any row
in ``[eval_start, eval_end]``. This is the ONLY defence against look-ahead
leakage; the harness enforces the invariant at runtime.

Expanding: every split re-uses all history <= train_end.
Rolling: every split uses a window of length ``train_size`` ending at
train_end (older data is discarded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single walk-forward split (both expanding and rolling variants use
    this shape)."""

    train_start: pd.Timestamp     # inclusive
    train_end: pd.Timestamp       # exclusive
    eval_start: pd.Timestamp      # inclusive; == train_end
    eval_end: pd.Timestamp        # exclusive

    def __post_init__(self) -> None:
        if not (self.train_start < self.train_end == self.eval_start < self.eval_end):
            raise ValueError(
                f"invalid split ordering: "
                f"train_start={self.train_start} train_end={self.train_end} "
                f"eval_end={self.eval_end}"
            )


# ``RollingSplit`` is an alias — the shape is identical. Naming lets callers
# be explicit about intent in a report.
RollingSplit = WalkForwardSplit


def _slice_bounds(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    """Half-open [start, end) slice endpoints in a sorted DatetimeIndex."""
    lo = int(index.searchsorted(start, side="left"))
    hi = int(index.searchsorted(end, side="left"))
    return lo, hi


def expanding_walk_forward(
    index: pd.Index,
    initial_train_size: int,
    eval_size: int,
    step: int | None = None,
) -> Iterator[WalkForwardSplit]:
    """Expanding-window walk-forward splits.

    ``initial_train_size`` is how many rows the first split trains on.
    ``eval_size`` is the width of each eval window (in rows).
    ``step`` (default = ``eval_size``) is how far the eval window slides.

    Each split's ``train_start`` == the frame's first date; only ``train_end``
    slides forward.
    """
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be monotonically increasing")
    if len(idx) < initial_train_size + eval_size:
        return
    step = step or eval_size
    if step <= 0:
        raise ValueError("step must be > 0")

    train_start = idx[0]
    train_end_pos = initial_train_size
    n = len(idx)

    while train_end_pos + eval_size <= n:
        train_end = idx[train_end_pos]
        eval_end_pos = train_end_pos + eval_size
        eval_end = idx[eval_end_pos] if eval_end_pos < n else idx[-1] + pd.Timedelta(days=1)
        yield WalkForwardSplit(
            train_start=train_start,
            train_end=train_end,
            eval_start=train_end,
            eval_end=eval_end,
        )
        train_end_pos += step


def rolling_walk_forward(
    index: pd.Index,
    train_size: int,
    eval_size: int,
    step: int | None = None,
) -> Iterator[WalkForwardSplit]:
    """Rolling-window walk-forward splits.

    Every split trains on exactly ``train_size`` most-recent rows before the
    eval window. Suitable when you believe the process has a finite memory.
    """
    idx = pd.DatetimeIndex(index)
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be monotonically increasing")
    if len(idx) < train_size + eval_size:
        return
    step = step or eval_size
    if step <= 0:
        raise ValueError("step must be > 0")

    train_start_pos = 0
    n = len(idx)

    while train_start_pos + train_size + eval_size <= n:
        train_end_pos = train_start_pos + train_size
        eval_end_pos = train_end_pos + eval_size
        eval_end = idx[eval_end_pos] if eval_end_pos < n else idx[-1] + pd.Timedelta(days=1)
        yield WalkForwardSplit(
            train_start=idx[train_start_pos],
            train_end=idx[train_end_pos],
            eval_start=idx[train_end_pos],
            eval_end=eval_end,
        )
        train_start_pos += step


def slice_train_eval(
    frame: pd.DataFrame, split: WalkForwardSplit
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, eval)`` frames obeying the split's half-open bounds."""
    idx = pd.DatetimeIndex(frame.index)
    tlo, thi = _slice_bounds(idx, split.train_start, split.train_end)
    elo, ehi = _slice_bounds(idx, split.eval_start, split.eval_end)
    return frame.iloc[tlo:thi], frame.iloc[elo:ehi]
