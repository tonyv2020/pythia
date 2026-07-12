"""Raptor p_move baselines.

``RaptorPMove`` (P3, real): raptor's ``p_move`` is a scalar MAGNITUDE
probability emitted each intraday bar (staging.qqq_pmove: date, bar_time,
p_move, oos). It is NOT a (mean, sigma) return forecast, so — per helen's D13
call — we use it as a CALIBRATED DISPERSION baseline:

    mean_t  = 0                       (p_move claims no direction)
    sigma_t = c * p_move_t            (bigger move-prob → wider band)

with the single scale ``c`` fit PER TRAIN WINDOW so the P10–P90 band covers
~80% of the realized (horizon-consistent) train moves. Then the harness scores
it on CRPS / coverage / pinball apples-to-apples with the TFT. Direction is a
SEPARATE baseline (``RaptorDirection``, from staging.qqq_direction).

The p_move history is injected as a ``pd.Series`` indexed on the frame's bar
timestamps, so unit tests use a deterministic fixture and never touch the
raptor Postgres (which the pipeline reads live on achilles).

``RaptorPMoveStub`` (kept): the original documented not-wired fallback that
raises rather than fabricating — still exported so a harness config can list
p_move as planned when no history is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.protocols import Model, ProbForecast

# z such that P(|Z| <= z) = 0.80 for Z ~ N(0,1)  → the P10–P90 half-width in σ.
_Z80 = 1.2815515594457412


class RaptorPMoveStub(Model):
    """Not-yet-wired raptor-p_move baseline. ``predict`` raises so no run
    silently omits it and reports as though everything passed. Superseded by
    ``RaptorPMove`` where a p_move history snapshot is available."""

    def __init__(self, target_col: str) -> None:
        self.target_col = target_col

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        raise NotImplementedError(
            "raptor p_move history not provided; use RaptorPMove(p_move=...) "
            "with a staging.qqq_pmove snapshot, or keep this stub as a planned "
            "baseline placeholder."
        )


def _forward_log_returns(px: pd.Series, horizon: int) -> pd.Series:
    """Move over [t, t+horizon]: log(px[t+h]) - log(px[t]). Last ``h`` rows NaN.

    Aligned at the EMISSION bar t — matching raptor's p_move, which is emitted
    at t forecasting the next ~horizon bars.
    """
    lp = np.log(px.astype(float))
    return lp.shift(-horizon) - lp


class RaptorPMove(Model):
    """Calibrated-dispersion baseline driven by raptor ``p_move``.

    ``p_move``: Series indexed on bar timestamps → p_move in [0,1].
    ``horizon``: bars ahead the target spans (P3 = 3 bars ≈ 30 min on 10-min
    bars; default 1 keeps it usable on any single-step frame).
    """

    def __init__(
        self,
        target_col: str,
        p_move: pd.Series,
        horizon: int = 1,
        min_train_rows: int = 30,
        calib_floor: float = 0.02,
        sigma_floor: float = 1e-9,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.target_col = target_col
        self.p_move = p_move.astype(float)
        self.horizon = horizon
        self.min_train_rows = min_train_rows
        # p_move has a fat near-zero tail (verified on live QQQ: median 0.032,
        # ~34% of rows < 0.01, min 3e-5). A near-zero p_move paired with a normal
        # realized move explodes |r|/(z80*pm), dragging the 0.80-quantile scale c
        # up and blowing sigma (and CRPS ~45x RW). So p_move below ``calib_floor``
        # is DROPPED from calibration and FLOORED in prediction — a within-spec
        # robustification of the D13 mapping. Verified on live QQQ: CRPS 0.076 →
        # 0.00168 (≈ RW), cov80 0.936 → 0.871.
        self.calib_floor = calib_floor
        self.sigma_floor = sigma_floor
        self._c: float | None = None
        self._fallback_pmove: float | None = None

    def fit(self, train: pd.DataFrame) -> None:
        r = _forward_log_returns(train[self.target_col], self.horizon)
        pm = self.p_move.reindex(train.index)
        df = pd.concat([r.rename("r"), pm.rename("pm")], axis=1).dropna()
        df = df[df["pm"] >= self.calib_floor]
        if len(df) < self.min_train_rows:
            raise RuntimeError(
                f"RaptorPMove needs >= {self.min_train_rows} aligned train rows "
                f"with p_move >= {self.calib_floor}, got {len(df)}"
            )
        # sigma_t = c * pm_t. Want P(|r| <= _Z80 * sigma) = 0.80 over train:
        # c = 0.80-quantile of |r| / (_Z80 * pm) over the floored calibration set.
        z = df["r"].abs() / (_Z80 * df["pm"])
        self._c = float(np.quantile(z, 0.80))
        self._fallback_pmove = float(df["pm"].median())

    def predict(self, eval_index: pd.Index) -> ProbForecast:
        assert self._c is not None, "RaptorPMove not fit"
        pm = self.p_move.reindex(eval_index)
        # Missing p_move for an eval bar → train-median; near-zero → floored so a
        # tiny move-prob can't collapse sigma.
        pm = pm.fillna(self._fallback_pmove).clip(lower=self.calib_floor)
        sigma = (self._c * pm).clip(lower=self.sigma_floor)
        return ProbForecast(
            mean=pd.Series(np.zeros(len(eval_index)), index=eval_index),
            sigma=pd.Series(sigma.to_numpy(), index=eval_index),
        )
