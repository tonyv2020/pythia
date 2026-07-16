"""Calibration: empirical coverage of a predictive interval.

For a well-calibrated 80% interval, ~80% of realised y_true values should
fall between P10 and P90. If coverage collapses to 40%, the model is
overconfident and any downstream trade sizing based on σ is wrong.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Fraction of ``y_true`` falling in the closed interval ``[lower, upper]``.

    Returns NaN on empty input. Requires lower <= upper on every entry;
    swapped/invalid intervals are treated as coverage-of-0 for that entry
    (so a broken model can't get free credit).
    """
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: y_true={yt.shape} lower={lo.shape} upper={hi.shape}")
    mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
    if not mask.any():
        return float("nan")
    inside = (yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])
    return float(np.mean(inside))
