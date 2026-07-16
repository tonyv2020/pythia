"""Directional hit-rate.

Fraction of predictions with the same sign as the realised return, excluding
observations where both are zero (which no model gets credit for)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def directional_hit_rate(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of forecasts whose sign matches the realised return, skipping rows where both are exactly zero; NaN if none are scorable."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true={yt.shape} y_pred={yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp) & ~((yt == 0) & (yp == 0))
    yt = yt[mask]
    yp = yp[mask]
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.sign(yt) == np.sign(yp)))
