"""Intraday bar assembler (P3 foundation).

The daily assembler rolls raptor ticks up to one bar per trading DAY. P3
forecasts the next 30–60 minutes, so it needs the same board rolled up to
fixed intraday BARS (default 30 min), timestamp-indexed, with the same wide
schema (`{symbol}_close` / `{symbol}_volume`) so the *existing* horizon-agnostic
harness / splits / baselines / metrics all apply unchanged — only the index
granularity changes.

Design choices exposed as parameters (helen owns the methodology call — see the
P3 design note):

- ``bar_minutes`` (default 30): bar width. "next-bar return" = the forecast
  horizon, so 30 → next-30-min, 60 → next-60-min.
- ``session_only`` (default True): keep only regular-session bars
  (09:30–16:00 by raptor's clock). Extended-hours ticks are thin and noisy.
- ``is_session_open`` column: marks the FIRST bar of each session. The
  bar-to-bar return at that row spans the OVERNIGHT gap, which is a different
  (and much larger-variance) forecasting problem than an intraday move. The
  backtest masks those rows so the intraday model/baselines are scored on
  within-session moves only. Exposed as a column (not silently dropped) so the
  choice is visible and reversible.

Covariate-lag gate is unchanged: features at bar t come from bar <= t-1, the
target at bar t is the realised [t-1, t] move — identical to the daily
contract, just at bar granularity. Known-future intraday calendar features
(minute-of-day, minutes-to-close) are added via
``calendar_features.intraday_calendar_features`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Callable, Iterable

import pandas as pd

from ..config import BOARD_SYMBOLS

# Regular US equity session by raptor's clock (see calendar_features).
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

# Injectable tick source for hermetic tests. Real path uses the assembler's
# Postgres ``_fetch_intraday``; tests pass a fixture frame.
TicksFn = Callable[[Iterable[str], date, date], pd.DataFrame]


@dataclass(frozen=True)
class IntradayAssemblyResult:
    bars: pd.DataFrame            # wide, DatetimeIndex (bar timestamp)
    symbols_included: tuple[str, ...]
    symbols_missing: tuple[str, ...]
    bar_minutes: int


def _floor_to_bar(t: pd.Series, bar_minutes: int) -> pd.Series:
    """Floor a time-of-day (as minutes past midnight) to the bar's left edge."""
    return (t // bar_minutes) * bar_minutes


def bars_from_ticks(ticks: pd.DataFrame, bar_minutes: int = 30,
                    session_only: bool = True) -> pd.DataFrame:
    """Roll intraday ticks up to per-symbol fixed ``bar_minutes`` OHLCV bars.

    Ticks need columns: ``symbol, date, time (HH:MM:SS), price, volume``.
    Returns long bars: ``symbol, bar_ts, session_date, open, high, low, close,
    volume, is_session_open``. Determinism: ties broken by tick order (input is
    ORDER BY symbol,date,time), so open/close are first/last within the bar.
    """
    cols = ["symbol", "bar_ts", "session_date", "open", "high", "low",
            "close", "volume", "is_session_open"]
    if ticks is None or ticks.empty:
        return pd.DataFrame(columns=cols)

    t = ticks.copy()
    t["price"] = pd.to_numeric(t["price"], errors="coerce")
    t["volume"] = pd.to_numeric(t["volume"], errors="coerce").fillna(0)
    t = t.dropna(subset=["price"])
    t = t[t["price"] > 0]

    tod = pd.to_datetime(t["time"], format="%H:%M:%S", errors="coerce")
    t["minute"] = tod.dt.hour * 60 + tod.dt.minute
    t["sec"] = tod.dt.second
    if session_only:
        open_min = SESSION_OPEN.hour * 60 + SESSION_OPEN.minute
        close_min = SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute
        t = t[(t["minute"] >= open_min) & (t["minute"] < close_min)]
    if t.empty:
        return pd.DataFrame(columns=cols)

    t["bar_min"] = _floor_to_bar(t["minute"], bar_minutes)
    t["session_date"] = pd.to_datetime(t["date"]).dt.date
    # Stable order so first/last within a bar are the true open/close.
    t = t.sort_values(["symbol", "session_date", "bar_min", "minute", "sec"])

    grp = t.groupby(["symbol", "session_date", "bar_min"], sort=True)
    bars = grp.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    # Bar timestamp = session_date + bar's left-edge minute.
    bars["bar_ts"] = pd.to_datetime(bars["session_date"]) + pd.to_timedelta(
        bars["bar_min"], unit="m"
    )
    # First bar of each (symbol, session) = the overnight-gap bar.
    bars = bars.sort_values(["symbol", "bar_ts"])
    bars["is_session_open"] = (
        bars.groupby("symbol")["session_date"].transform(
            lambda s: s != s.shift(1)
        )
    )
    return bars[cols].reset_index(drop=True)


def _pivot_intraday_wide(bars: pd.DataFrame) -> pd.DataFrame:
    """Long bars -> wide, index = bar_ts, cols ``{symbol}_close`` /
    ``{symbol}_volume`` (+ a frame-level ``is_session_open`` marker)."""
    if bars.empty:
        return pd.DataFrame()
    close = (bars.pivot(index="bar_ts", columns="symbol", values="close")
             .add_suffix("_close").sort_index())
    volume = (bars.pivot(index="bar_ts", columns="symbol", values="volume")
              .add_suffix("_volume").sort_index())
    wide = close.join(volume, how="outer")
    wide = wide[sorted(wide.columns)]

    # A bar_ts is a session-open row if ANY symbol marks it so (they share the
    # session calendar). Kept as one frame-level column for overnight masking.
    open_flags = (bars.groupby("bar_ts")["is_session_open"].max()
                  .reindex(wide.index).fillna(False).astype(bool))
    wide["is_session_open"] = open_flags.values

    # Known-future intraday calendar features from the bar timestamp.
    idx = pd.to_datetime(wide.index)
    minute_of_day = idx.hour * 60 + idx.minute
    close_min = SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute
    wide["minute_of_day"] = minute_of_day
    wide["minutes_to_close"] = close_min - minute_of_day
    wide["dow"] = idx.weekday
    return wide


def assemble_intraday_dataset(
    start: date,
    end: date,
    engine=None,
    bar_minutes: int = 30,
    session_only: bool = True,
    symbols: Iterable[str] | None = None,
    ticks_fn: TicksFn | None = None,
) -> IntradayAssemblyResult:
    """Assemble the intraday board dataset for ``[start, end]``.

    ``ticks_fn`` overrides the tick source (the test seam). Default pulls from
    raptor Postgres via the daily assembler's ``_fetch_intraday`` — so the
    intraday path shares the exact same source query and symbol-omit policy.
    """
    wanted = list(symbols) if symbols is not None else list(BOARD_SYMBOLS)

    if ticks_fn is not None:
        ticks = ticks_fn(wanted, start, end)
    else:
        # Lazy import to avoid a hard SQLAlchemy engine requirement in tests.
        from .assembler import _fetch_intraday
        from .source import get_engine

        ticks = _fetch_intraday(engine or get_engine(), wanted, start, end)

    present = set(ticks["symbol"].unique()) if not ticks.empty else set()
    missing = tuple(s for s in wanted if s not in present)
    included = tuple(s for s in wanted if s in present)

    bars = bars_from_ticks(ticks, bar_minutes=bar_minutes, session_only=session_only)
    wide = _pivot_intraday_wide(bars)
    return IntradayAssemblyResult(
        bars=wide,
        symbols_included=included,
        symbols_missing=missing,
        bar_minutes=bar_minutes,
    )


def overnight_mask(bars_wide: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where the next-bar (TRAILING) target should be KEPT
    (i.e. NOT an overnight-gap row). First bar of each session → False. Use this
    for a 1-bar horizon; for a forward multi-bar horizon use
    ``forward_session_mask``."""
    if "is_session_open" not in bars_wide.columns:
        return pd.Series(True, index=bars_wide.index)
    return ~bars_wide["is_session_open"].astype(bool)


def forward_session_mask(bars_wide: pd.DataFrame, horizon: int) -> pd.Series:
    """Boolean Series for a FORWARD ``horizon``-bar target: True where bar t and
    bar t+horizon fall in the SAME session (calendar date), so the forecast
    interval [t, t+h] does NOT cross an overnight gap. End-of-session bars whose
    forward target would span overnight → False (excluded from scoring). This is
    the P3 within-session mask for the 30-min = 3-bar horizon.
    """
    idx = pd.to_datetime(bars_wide.index)
    dates = pd.Series([ts.date() for ts in idx], index=bars_wide.index)
    same = dates.values[:-horizon] == dates.values[horizon:] if horizon > 0 else []
    keep = pd.Series(False, index=bars_wide.index)
    if horizon > 0 and len(bars_wide) > horizon:
        keep.iloc[: len(bars_wide) - horizon] = same
    return keep
