"""Target constructors — return + realized range/vol.

Both targets are aligned so that the value at row ``t`` is what the model,
FORECASTING FROM ROW t-1, is trying to predict about the interval
[t-1, t]. This is symmetric with the covariate-lag gate: features at row
t come from t-1 (or earlier for wider horizons), target at row t is the
realised outcome at t.

- ``return_target``: log(px_t) - log(px_{t-1}). NaN on the first row.
- ``realized_range_target``: log(high_t / low_t). Range (log-scale, always
   >= 0) is a robust proxy for realised volatility and is empirically more
   forecastable than direction — helen made it first-class in P1 for that
   reason.

Both return a pandas Series indexed by date so the harness can join them
against the feature frame produced by ``build_features``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_target(px: pd.Series) -> pd.Series:
    px = px.astype(float).sort_index()
    return np.log(px / px.shift(1)).rename(f"{px.name}_ret" if px.name else "ret")


def realized_range_target(
    high: pd.Series, low: pd.Series, name: str | None = None
) -> pd.Series:
    """Log(high / low). Positive semi-definite; 0 only for a flat bar."""
    h = high.astype(float).sort_index()
    lo = low.astype(float).sort_index()
    if not (h >= lo).all():
        # Silently mask nonsense rows rather than raise — bad ticks happen
        # and the training loop's NaN filter will drop them.
        mask = h >= lo
        h = h.where(mask)
        lo = lo.where(mask)
    r = np.log(h / lo)
    r.name = name or (f"{high.name}_range" if high.name else "range")
    return r
