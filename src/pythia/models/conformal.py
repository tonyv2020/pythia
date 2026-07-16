"""Per-train-window conformal calibration wrapper (helen D18).

A single TFT run's cov80 is SEED-NOISY (0.586 → 0.705 → 0.781 on identical
code+data), so chasing a lucky-seed calibration is cherry-picking. Instead we
right-size the predictive spread by CONSTRUCTION, per train window, using the
same calibration machinery already proven on the p_move baseline:

    fit, per train window:   s = quantile_p(|y - mean| / sigma) / z_p
    apply, on eval:          sigma_calibrated = s * sigma

where p = ``coverage`` (0.80) and z_p = Phi^-1(0.5 + p/2) is the two-sided
z for that central mass (z_0.80 = 1.2816). By construction the TRAIN central-p
interval covers ~p; ``eval`` coverage is then the honest out-of-sample check —
seed-independent, because the residual-distribution scale is far more stable
across seeds than any single run's raw coverage.

Calibration != skill: the wrapper scales only the DISPERSION. ``mean`` is
untouched, so MAE / directional metrics and the null-vs-RW verdict are
unchanged — it only makes the cone honestly-sized. Works on ANY base
``Model`` (daily TFTLiteModel + the intraday subclass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-not-found]

from ..backtest.harness import _target_returns
from ..backtest.protocols import Model, ProbForecast


@dataclass
class ConformalScaledModel(Model):
    """Wrap a base model; scale its predictive sigma so train coverage ~= p."""

    base: Model
    target_col: str = "QQQ_close"
    horizon: int = 1
    coverage: float = 0.80
    min_cal_rows: int = 20
    # Base sigma is FLOORED at ``sigma_floor_frac`` × a robust (MAD-based)
    # estimate of the residual scale before the ratio is taken. Without this, a
    # collapsed / degenerate base sigma (near-zero, common under a noisy seed or
    # low epochs) makes |resid|/sigma explode and blows the scale up — the same
    # near-zero-denominator failure mode fixed in the p_move baseline.
    sigma_floor_frac: float = 0.25
    sigma_floor_abs: float = 1e-9
    # P5a: calibrate against an arbitrary target (e.g. realized-range) instead of
    # the price return — must MATCH the target the harness scores for this model.
    # None → the default price target (_target_returns), unchanged.
    target_fn: "Callable[[pd.DataFrame], pd.Series] | None" = None

    _s: float = field(default=1.0, init=False)
    _floor: float = field(default=1e-9, init=False)

    def fit(self, train: pd.DataFrame) -> None:
        """Fit the wrapped model, then calibrate a conformal scaling factor from held-out train residuals."""
        self.base.fit(train)
        # The SAME target the harness scores for this model (price return by
        # default; a range/other target via target_fn — P5a).
        if self.target_fn is not None:
            y = self.target_fn(train).dropna()
        else:
            y = _target_returns(train, self.target_col, self.horizon).dropna()
        # Skip the base's encoder warm-up rows — there it returns a fallback
        # forecast (flat mean, fixed wide sigma) that would bias the scale.
        warm = int(getattr(self.base, "encoder_length", 0) or 0)
        cal_index = y.index[warm:] if warm < len(y) else y.index
        if len(cal_index) < self.min_cal_rows:
            self._s = 1.0  # not enough calibration rows → no-op
            self._floor = self.sigma_floor_abs
            return
        fc = self.base.predict(cal_index)
        resid = np.abs(y.loc[cal_index].to_numpy() - fc.mean.to_numpy())
        # Robust residual scale → a data-driven floor so no row's sigma is
        # pathologically small relative to the typical error.
        mad = float(np.median(resid))
        resid_scale = 1.4826 * mad if mad > 0 else float(np.std(resid) or 0.0)
        self._floor = max(self.sigma_floor_frac * resid_scale, self.sigma_floor_abs)
        sigma = np.maximum(fc.sigma.to_numpy(), self._floor)
        e = resid / sigma
        z = float(norm.ppf(0.5 + self.coverage / 2.0))
        q = float(np.quantile(e, self.coverage))
        self._s = max(q / z, self.sigma_floor_abs) if q > 0 else 1.0

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        """Emit the wrapped model's forecast with sigma scaled by the conformal calibration factor."""
        fc = self.base.predict(eval_index)
        sigma = np.maximum(fc.sigma.to_numpy(), self._floor)
        return ProbForecast(
            mean=fc.mean,
            sigma=pd.Series((self._s * sigma).clip(min=self.sigma_floor_abs), index=eval_index),
        )
