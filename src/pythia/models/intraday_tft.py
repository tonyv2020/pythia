"""Intraday TFT-lite (P3) — horizon-consistent target.

The daily ``TFTLiteModel`` builds a ONE-STEP return target
(``return_target`` = log(px[t]/px[t-1])), which matches the daily harness
(``run_backtest`` horizon=1). The intraday harness scores the FORWARD h-bar
return log(px[t+h]/px[t]) (horizon=3 ≈ 30 min on 10-min bars), so the model
MUST train to predict that same forward-h quantity — otherwise it forecasts a
1-bar move while being scored on a 3-bar move (a silent correctness bug).

This subclass overrides ONLY the target construction to the forward-h return;
everything else (features, lag gate, quantile heads, predict path) is inherited
from the reviewed daily adapter unchanged. No edit to the P1 model file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .adapter import TFTLiteModel


@dataclass
class IntradayTFTLiteModel(TFTLiteModel):
    """TFT-lite variant trained on the forward h-bar return (P3 intraday horizon=3), fixing the 1-bar/3-bar target mismatch."""
    horizon: int = 3

    def _build_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        px = frame[self.target_col].astype(float)
        lp = np.log(px)
        # Forward h-bar return, aligned at the emission bar t (matches the
        # harness y_true and the raptor p_move emission convention). Last h
        # rows are NaN and drop out in fit's target join.
        r = (lp.shift(-self.horizon) - lp).rename("y_return")
        # Range head is auxiliary (multi-task regulariser); intraday QQQ bars
        # carry no high/low columns, so use the forward-h move magnitude.
        rng = r.abs().rename("y_range")
        return pd.concat([r, rng], axis=1)
