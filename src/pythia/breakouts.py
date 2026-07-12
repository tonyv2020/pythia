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
from typing import Callable

import numpy as np
import pandas as pd

from .backtest.harness import _target_returns
from .backtest.protocols import Model
from .backtest.splits import expanding_walk_forward, slice_train_eval

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


ModelFactory = Callable[[], Model]


def run_breakout_scan(
    frame: pd.DataFrame,
    model_factory: ModelFactory,
    target_col: str = "QQQ_close",
    symbol: str = "QQQ",
    model_version: str = "unknown",
    initial_train: int = 252,
    eval_size: int = 63,
    horizon: int = 1,
) -> pd.DataFrame:
    """Walk-forward POPULATION collector: replay the served price model's
    OOS forecasts across expanding splits and flag every bar whose realized
    move breached the P10-P90 band.

    This is the SAME walk-forward geometry the report uses (a model fit on
    rows < train_end never sees eval), so every row is genuinely out-of-sample
    (``oos=True``). Returns one row per scored eval bar with the pythia_breakouts
    columns (model_version, symbol, ts, horizon, direction, realized, p10, p90,
    exceeded, magnitude, oos) — ready to UPSERT into the audit table.

    ``model_factory`` returns a FRESH model per split (no state bleed = no
    look-ahead). Any ``Model`` works: a CPU baseline for a cheap/hermetic scan,
    or the served TFT for the real band. It is a calibration diagnostic — the
    breach rate vs the 20% expected — NOT a trade signal.
    """
    returns = _target_returns(frame, target_col, horizon)
    parts: list[pd.DataFrame] = []
    for split in expanding_walk_forward(frame.index, initial_train, eval_size):
        train_frame, eval_frame = slice_train_eval(frame, split)
        if eval_frame.empty:
            continue
        # Same leak-free purge/cut as the harness for horizon>1.
        purge = max(horizon - 1, 0)
        if purge > 0 and len(train_frame) > purge:
            train_frame = train_frame.iloc[:-purge]
        y_true = returns.loc[eval_frame.index].dropna()
        if purge > 0:
            y_true = y_true.iloc[:-purge] if len(y_true) > purge else y_true.iloc[:0]
        if y_true.empty:
            continue

        model = model_factory()
        model.fit(train_frame)
        fc = model.predict(y_true.index)
        p10 = fc.quantile(0.10)
        p90 = fc.quantile(0.90)
        bo = detect_breakouts(y_true, p10, p90)
        if bo.empty:
            continue
        bo.insert(0, "model_version", model_version)
        bo.insert(1, "symbol", symbol)
        bo["horizon"] = horizon
        bo["oos"] = True
        parts.append(bo)

    cols = ["model_version", "symbol", "ts", "horizon", "direction",
            "realized", "p10", "p90", "exceeded", "magnitude", "oos"]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    return out[cols]


def build_breakouts_response(
    scan: pd.DataFrame, window: int = 20, recent: int = 30
) -> dict:
    """The /breakouts serve block: the rolling breach rate vs the 20% expected
    + the most recent breakout events + an honest calibration verdict.

    ``badge``: green when the rolling rate sits near expected (0.10-0.30 band),
    amber otherwise (bands systematically too tight → over-rate, or too wide →
    under-rate). Disclosure, not a trade signal (matches the D25 range badge).
    """
    rate = breakout_rate(scan, window=window)
    r = rate.breakout_rate
    in_band = 0.10 <= r <= 0.30 if r == r else False  # r==r → not NaN
    if r != r:
        verdict = "no scan rows yet"
    elif r > 0.30:
        verdict = f"breach rate {r:.0%} >> {EXPECTED_RATE:.0%} expected — bands too TIGHT (under-dispersed)."
    elif r < 0.10:
        verdict = f"breach rate {r:.0%} << {EXPECTED_RATE:.0%} expected — bands too WIDE (over-dispersed)."
    else:
        verdict = f"breach rate {r:.0%} ~ {EXPECTED_RATE:.0%} expected — band calibrated."

    events: list[dict] = []
    if not scan.empty:
        tail = scan[scan["exceeded"]].tail(recent)
        for _, row in tail.iterrows():
            ts = row["ts"]
            events.append({
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "direction": row["direction"],
                "realized": float(row["realized"]),
                "p10": float(row["p10"]),
                "p90": float(row["p90"]),
                "magnitude": float(row["magnitude"]),
            })

    return {
        "expected_rate": EXPECTED_RATE,
        "window": window,
        "rate": None if r != r else float(r),
        "n": rate.n,
        "n_scan": int(len(scan)),
        "badge": "green" if in_band else "amber",
        "verdict": verdict,
        "events": events,
        "note": ("rolling P10-P90 breach rate vs the ~20% expected under "
                 "calibration. A diagnostic of forecast honesty (band too "
                 "tight/wide), NOT a trade signal — do not size trades from "
                 "breakout flags."),
    }
