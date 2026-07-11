"""Pinball (quantile) loss.

For quantile q ∈ (0, 1) and forecast quantile ``q_hat``:

    L_q(y, q_hat) = max( q*(y - q_hat), (q-1)*(y - q_hat) )

Reduces to |y - median| when q = 0.5 (up to a factor of 2, absorbed).
Lower is better.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def pinball_loss(y_true: ArrayLike, q_hat: ArrayLike, q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    yt = np.asarray(y_true, dtype=float)
    qh = np.asarray(q_hat, dtype=float)
    if yt.shape != qh.shape:
        raise ValueError(f"shape mismatch: y_true={yt.shape} q_hat={qh.shape}")
    mask = np.isfinite(yt) & np.isfinite(qh)
    if not mask.any():
        return float("nan")
    yt = yt[mask]
    qh = qh[mask]
    diff = yt - qh
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
