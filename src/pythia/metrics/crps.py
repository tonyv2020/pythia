"""Continuous Ranked Probability Score under a Normal predictive.

Closed-form (Gneiting & Raftery 2007):

    CRPS(N(mu, sigma), y)
        = sigma * ( z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) )
    where z = (y - mu) / sigma.

Lower is better. Reduces to |y - mu| as sigma → 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm  # type: ignore[import-not-found]


def crps_normal(y_true: ArrayLike, mean: ArrayLike, sigma: ArrayLike) -> float:
    """Mean CRPS of a Normal(mean, sigma) predictive vs ``y_true`` (closed form; lower is better); non-finite or sigma<=0 rows dropped, NaN if none remain."""
    yt = np.asarray(y_true, dtype=float)
    mu = np.asarray(mean, dtype=float)
    sd = np.asarray(sigma, dtype=float)
    if not (yt.shape == mu.shape == sd.shape):
        raise ValueError(f"shape mismatch: y_true={yt.shape} mean={mu.shape} sigma={sd.shape}")
    mask = np.isfinite(yt) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    if not mask.any():
        return float("nan")
    yt = yt[mask]
    mu = mu[mask]
    sd = sd[mask]

    z = (yt - mu) / sd
    crps = sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))
