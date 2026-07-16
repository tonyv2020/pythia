"""Raptor direction baseline (P3).

raptor's ``staging.qqq_direction`` emits ``p_up`` / ``p_dn`` each intraday bar.
Per helen's D13 call, this is a SEPARATE directional baseline (distinct from the
p_move dispersion baseline): the directional TILT ``p_up - p_dn`` drives the
forecast MEAN sign.

Mapping (per-train-window calibrated, no fabricated magnitude):

    mean_t  = beta * tilt_t          tilt = p_up - p_dn
    sigma_t = residual std of train  (constant, RW-style)

``beta`` is a no-intercept OLS of the realized (horizon-consistent) train move
on the tilt — so the tilt's scale is learned from data, not assumed. If the
tilt has no explanatory variance, beta collapses to 0 and this degenerates to a
zero-mean RW-style forecast (honest: no directional edge claimed).

``tilt`` is injected as a Series indexed on bar timestamps, so tests use a
fixture and never touch raptor Postgres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.protocols import Model, ProbForecast
from .raptor_p_move import _forward_log_returns


class RaptorDirection(Model):
    """P3 directional baseline: mean = beta*(p_up - p_dn) tilt, sigma = train residual stdev (helen D13)."""
    def __init__(
        self,
        target_col: str,
        tilt: pd.Series,
        horizon: int = 1,
        min_train_rows: int = 30,
        sigma_floor: float = 1e-9,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.target_col = target_col
        self.tilt = tilt.astype(float)
        self.horizon = horizon
        self.min_train_rows = min_train_rows
        self.sigma_floor = sigma_floor
        self._beta: float | None = None
        self._sigma: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        """OLS-fit the no-intercept tilt->return beta and residual sigma; degrade to zero-drift RW when train is direction-sparse."""
        r = _forward_log_returns(train[self.target_col], self.horizon)
        tilt = self.tilt.reindex(train.index)
        df = pd.concat([r.rename("r"), tilt.rename("tilt")], axis=1).dropna()

        # Graceful fallback (parallel to RaptorPMove PR #13): if not enough
        # aligned train rows, degrade to a flat zero-drift RW-style forecast
        # (mean=0 via beta=0, sigma = train forward-return std). Direction-sparse
        # early windows carry no usable directional signal; the harness must
        # not crash, and no baseline should silently miss a split.
        if len(df) < self.min_train_rows:
            # Sigma from the target returns in the raw train window.
            r_only = r.dropna().to_numpy()
            s = float(np.std(r_only, ddof=1)) if len(r_only) > 1 else 0.0
            self._beta = 0.0
            self._sigma = max(s, self.sigma_floor)
            return

        x = df["tilt"].to_numpy()
        y = df["r"].to_numpy()
        denom = float(np.dot(x, x))
        self._beta = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        resid = y - self._beta * x
        s = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        self._sigma = max(s, self.sigma_floor)

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Forecast mean = beta*tilt (0 where tilt is missing) with constant residual sigma."""
        assert self._beta is not None and self._sigma is not None, "not fit"
        tilt = self.tilt.reindex(eval_index).fillna(0.0)
        mean = self._beta * tilt.to_numpy()
        return ProbForecast(
            mean=pd.Series(mean, index=eval_index),
            sigma=pd.Series(np.full(len(eval_index), self._sigma), index=eval_index),
        )
