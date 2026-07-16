"""Persistence / last-observed-return baseline.

Forecast for every eval row: ``mean`` = the last observed log-return in the
training set (a "momentum" prior at the crudest level). ``sigma`` = train
stdev, as with RandomWalk.

If this beats RandomWalk on hit-rate but not on CRPS/coverage, the
message is "there's a tiny persistent drift but you can't call it usefully."
The harness reports both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.protocols import Model, ProbForecast


class LastReturn(Model):
    """Persistence baseline: forecast mean = last observed train log-return, sigma = train-return stdev."""
    def __init__(self, target_col: str, min_train_rows: int = 30) -> None:
        self.target_col = target_col
        self.min_train_rows = min_train_rows
        self._mean: float | None = None
        self._sigma: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        """Set mean to the last train log-return and sigma to the train-return stdev; raise if too few rows."""
        px = train[self.target_col].astype(float)
        r = np.log(px / px.shift(1)).dropna()
        if len(r) < self.min_train_rows:
            raise RuntimeError(
                f"LastReturn needs >= {self.min_train_rows} train rows, got {len(r)}"
            )
        self._mean = float(r.iloc[-1])
        self._sigma = max(float(r.std(ddof=1)), 1e-9)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Constant persisted-last-return (mean, sigma) across ``eval_index``."""
        assert self._mean is not None and self._sigma is not None, "LastReturn not fit"
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.full(n, self._mean), index=eval_index),
            sigma=pd.Series(np.full(n, self._sigma), index=eval_index),
        )
