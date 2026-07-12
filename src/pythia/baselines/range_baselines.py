"""Range-target baselines (P5a multi-target).

The realized-range target is ``log(high/low)`` — a POSITIVE dispersion measure,
not a symmetric return. Its honest floors are persistence and a rolling window
(the range analogues of random-walk / last-value):

- ``LastRange``: mean = the last train realized-range (persistence); sigma =
  train range std. "Tomorrow's range ≈ today's."
- ``RollingRange``: mean = rolling-mean of the last ``window`` ranges; sigma =
  their rolling std. The mean-reversion floor.

A model earns its keep only if its per-row conditional range forecast beats
these constant-per-split floors on CRPS / pinball (realized range is
empirically more forecastable than direction, so this is a real bar — but a
real target too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.protocols import Model, ProbForecast
from ..features.targets import realized_range_target


def _range_series(frame: pd.DataFrame, high_col: str, low_col: str) -> pd.Series:
    return realized_range_target(frame[high_col], frame[low_col]).dropna()


class LastRange(Model):
    """Persistence: forecast every eval row at the last train realized-range."""

    def __init__(self, high_col: str, low_col: str, min_train_rows: int = 20,
                 sigma_floor: float = 1e-9) -> None:
        self.high_col = high_col
        self.low_col = low_col
        self.min_train_rows = min_train_rows
        self.sigma_floor = sigma_floor
        self._mean: float | None = None
        self._sigma: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        r = _range_series(train, self.high_col, self.low_col)
        if len(r) < self.min_train_rows:
            raise RuntimeError(f"LastRange needs >= {self.min_train_rows} rows, got {len(r)}")
        self._mean = float(r.iloc[-1])
        self._sigma = max(float(r.std(ddof=1)), self.sigma_floor)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        assert self._mean is not None
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.full(n, self._mean), index=eval_index),
            sigma=pd.Series(np.full(n, self._sigma), index=eval_index),
        )


class RollingRange(Model):
    """Rolling-window mean±std of realized-range."""

    def __init__(self, high_col: str, low_col: str, window: int = 20,
                 min_train_rows: int = 20, sigma_floor: float = 1e-9) -> None:
        self.high_col = high_col
        self.low_col = low_col
        self.window = window
        self.min_train_rows = min_train_rows
        self.sigma_floor = sigma_floor
        self._mean: float | None = None
        self._sigma: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        r = _range_series(train, self.high_col, self.low_col)
        if len(r) < self.min_train_rows:
            raise RuntimeError(f"RollingRange needs >= {self.min_train_rows} rows, got {len(r)}")
        tail = r.iloc[-self.window:]
        self._mean = float(tail.mean())
        self._sigma = max(float(tail.std(ddof=1)) if len(tail) > 1 else 0.0,
                          self.sigma_floor)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        assert self._mean is not None
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.full(n, self._mean), index=eval_index),
            sigma=pd.Series(np.full(n, self._sigma), index=eval_index),
        )
