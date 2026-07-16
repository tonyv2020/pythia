"""Zero-mean Gaussian random-walk baseline.

Forecast for every eval row: ``mean=0`` (log-return), ``sigma`` = historical
stdev of training log-returns of the target series. This is the honest
"nothing to see here" prior — if a proposed model can't beat it on CRPS,
pinball, and MAE, the model is not useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.protocols import Model, ProbForecast


class RandomWalk(Model):
    """Zero-mean Gaussian with historical σ. Sigma is fit-time constant."""

    def __init__(self, target_col: str, min_train_rows: int = 30) -> None:
        self.target_col = target_col
        self.min_train_rows = min_train_rows
        self._sigma: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        """Estimate sigma as the stdev of train log-returns (floored > 0); raise if too few rows."""
        px = train[self.target_col].astype(float)
        r = np.log(px / px.shift(1)).dropna()
        if len(r) < self.min_train_rows:
            raise RuntimeError(
                f"RandomWalk needs >= {self.min_train_rows} train rows, got {len(r)}"
            )
        s = float(r.std(ddof=1))
        # Floor to avoid degenerate sigma=0 forecasts on flat train windows.
        self._sigma = max(s, 1e-9)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Zero-mean, constant-sigma Gaussian forecast across ``eval_index``."""
        assert self._sigma is not None, "RandomWalk not fit"
        n = len(eval_index)
        return ProbForecast(
            mean=pd.Series(np.zeros(n), index=eval_index),
            sigma=pd.Series(np.full(n, self._sigma), index=eval_index),
        )
