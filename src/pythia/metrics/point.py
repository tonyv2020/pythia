"""Point metrics: MAE, RMSE, and MAE skill vs a baseline (typically RW)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true={yt.shape} y_pred={yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error over finite (y_true, y_pred) pairs; NaN if none."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square error over finite (y_true, y_pred) pairs; NaN if none."""
    yt, yp = _pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae_skill_vs(model_mae: float, baseline_mae: float) -> float:
    """Skill score in (-inf, 1]. 1 = perfect vs baseline; 0 = same as baseline;
    negative = worse than baseline. NaN if baseline_mae <= 0.
    """
    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return float("nan")
    return 1.0 - (model_mae / baseline_mae)
