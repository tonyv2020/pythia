"""Breakout detection (P5b) — a calibration diagnostic, NOT an alpha signal.

A "breakout" = the realized move landed OUTSIDE the model's forecast P10-P90
band (a tail event). If the model is calibrated (P10-P90 covers 80%), the
expected breakout rate is ~20% (10% per tail). The panel's scorecard is the
ROLLING breakout rate vs that 20% expected — over-rate = bands too tight,
under-rate = too wide. It is a diagnostic of the forecast's honesty, not a
trade signal.

Pure functions here (detect + rate) so they are fully hermetic-testable; the
walk-forward population loop + the /breakouts serve handler build on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EXPECTED_RATE = 0.20  # P10-P90 band → ~20% of moves fall outside if calibrated

# DDL for the registry-side table (created idempotently by the population job).
BREAKOUTS_DDL = """
CREATE TABLE IF NOT EXISTS pythia_breakouts (
    id             BIGSERIAL PRIMARY KEY,
    model_version  TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    ts             TIMESTAMP NOT NULL,
    horizon        INTEGER NOT NULL DEFAULT 1,
    direction      TEXT NOT NULL,          -- 'up' | 'down' | 'none'
    realized       DOUBLE PRECISION,
    p10            DOUBLE PRECISION,
    p90            DOUBLE PRECISION,
    exceeded       BOOLEAN NOT NULL,
    magnitude      DOUBLE PRECISION,       -- distance past the band (0 if none)
    oos            BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (model_version, symbol, ts, horizon)
);
"""


@dataclass(frozen=True)
class BreakoutRate:
    breakout_rate: float
    expected: float
    n: int
    window: int


def detect_breakouts(
    realized: pd.Series, p10: pd.Series, p90: pd.Series
) -> pd.DataFrame:
    """Per-bar breakout flags. Rows aligned on the shared index (inner-joined,
    NaNs dropped). ``direction``: 'up' if realized > p90, 'down' if < p10, else
    'none'. ``magnitude`` = signed distance past the breached edge (0 if none).
    """
    df = pd.concat(
        [realized.rename("realized"), p10.rename("p10"), p90.rename("p90")],
        axis=1,
    ).dropna()
    if df.empty:
        return pd.DataFrame(
            columns=["ts", "realized", "p10", "p90", "direction", "exceeded", "magnitude"]
        )
    up = df["realized"] > df["p90"]
    down = df["realized"] < df["p10"]
    direction = np.where(up, "up", np.where(down, "down", "none"))
    magnitude = np.where(
        up, df["realized"] - df["p90"],
        np.where(down, df["p10"] - df["realized"], 0.0),
    )
    out = pd.DataFrame(
        {
            "ts": df.index,
            "realized": df["realized"].to_numpy(),
            "p10": df["p10"].to_numpy(),
            "p90": df["p90"].to_numpy(),
            "direction": direction,
            "exceeded": (up | down).to_numpy(),
            "magnitude": magnitude,
        }
    ).reset_index(drop=True)
    return out


def breakout_rate(breakouts: pd.DataFrame, window: int = 20) -> BreakoutRate:
    """Rolling breakout rate over the last ``window`` rows vs the 20% expected."""
    if breakouts.empty:
        return BreakoutRate(breakout_rate=float("nan"), expected=EXPECTED_RATE, n=0, window=window)
    tail = breakouts.tail(window)
    rate = float(tail["exceeded"].mean())
    return BreakoutRate(
        breakout_rate=rate, expected=EXPECTED_RATE, n=int(len(tail)), window=window
    )
