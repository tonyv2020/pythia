"""Assemble the P0 dataset from ``staging.quote_raw``.

Pipeline:
    1. Pull intraday ticks for the board (target + covariates) from Postgres,
       between ``start`` and ``end``.
    2. Roll them up to per-symbol per-day OHLCV (session close = last non-zero
       price, session open = first, high/low, sum of volume).
    3. Pivot to a wide (date × [symbol_close, symbol_volume, ...]) frame.
    4. Attach calendar features via ``calendar_features.add_calendar_features``.
    5. Emit tidy Parquet.

Everything is date-indexed to a single trading-day granularity; intraday is
retained as its own Parquet if the caller asks for it (P0 doesn't need it,
but the API is stable so P1+ models can request it without refactoring).

Determinism: given identical ``staging.quote_raw`` state, ``assemble_dataset``
produces byte-identical Parquet. Ordering is fixed: rows by (date), columns
by (symbol asc within each measure asc). No timestamps in output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import Engine, text

from ..config import BOARD_SYMBOLS, QUOTE_TABLE, RATE_SYMBOLS, VIX_SYMBOLS
from .calendar_features import add_calendar_features
from .historical import ProviderFn, combine_daily, fetch_historical_daily_bars
from .source import get_engine


@dataclass(frozen=True)
class AssemblyResult:
    """Return type from ``assemble_dataset``. ``daily`` is the P0 dataset."""

    daily: pd.DataFrame
    intraday: pd.DataFrame | None
    symbols_included: tuple[str, ...]
    symbols_missing: tuple[str, ...]
    # D8 backfill: symbols whose OLD bars were filled from the historical
    # provider. Empty on the default (Postgres-only) path.
    symbols_backfilled: tuple[str, ...] = ()


def _fetch_intraday(
    engine: Engine,
    symbols: Iterable[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    q = text(
        f"""
        SELECT symbol,
               date::date          AS date,
               time                AS time,
               last_trade_price    AS price,
               volume              AS volume,
               open_price          AS open_price
        FROM {QUOTE_TABLE}
        WHERE symbol = ANY(:syms)
          AND date BETWEEN :start AND :end
          AND last_trade_price IS NOT NULL
          AND last_trade_price > 0
        ORDER BY symbol, date, time
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(
            q, conn, params={"syms": list(symbols), "start": start, "end": end}
        )


def daily_bars_from_intraday(intraday: pd.DataFrame) -> pd.DataFrame:
    """Roll intraday ticks up to per-symbol per-date OHLCV bars.

    - ``open``  = first non-zero ``price`` of the day (or fall back to
                  ``open_price`` if that column has any values).
    - ``close`` = last non-zero ``price`` of the day.
    - ``high`` / ``low`` = intraday max / min of ``price``.
    - ``volume`` = sum (guard NaN → 0).

    Rows with a single tick still produce a bar (open == high == low == close).
    """
    if intraday.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )

    intraday = intraday.copy()
    intraday["price"] = pd.to_numeric(intraday["price"], errors="coerce")
    intraday["volume"] = pd.to_numeric(intraday["volume"], errors="coerce").fillna(0)

    grouped = intraday.groupby(["symbol", "date"], sort=True)

    daily = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    # If open_price column is populated (some vendors ship it) prefer that
    # for the day's open — the first tick may be mid-session.
    if "open_price" in intraday.columns:
        vendor_open = (
            intraday.dropna(subset=["open_price"])
            .groupby(["symbol", "date"], sort=True)["open_price"]
            .first()
            .reset_index()
        )
        if not vendor_open.empty:
            daily = daily.merge(
                vendor_open, on=["symbol", "date"], how="left", suffixes=("", "_vendor")
            )
            daily["open"] = daily["open_price"].where(
                daily["open_price"].notna() & (daily["open_price"] > 0), daily["open"]
            )
            daily = daily.drop(columns=["open_price"])

    return daily.reset_index(drop=True)


def _pivot_wide(
    daily: pd.DataFrame, hl_symbols: "set[str] | None" = None
) -> pd.DataFrame:
    """Long → wide: index = date, cols = ``{symbol}_close`` / ``{symbol}_volume``.

    ``close`` and ``volume`` propagate for every symbol — enough for the price
    baselines + metrics. ``high`` / ``low`` are added ONLY for symbols in
    ``hl_symbols`` (P5a multi-target: the realized-range target = log(high/low)
    needs them). Kept opt-in + target-only because the model adapter treats a
    symbol's high/low as TARGET columns (excluded from features); adding them
    for every symbol would instead balloon the covariate set and change the
    price model. Default ``None`` → close+volume only (existing behaviour).
    """
    if daily.empty:
        return pd.DataFrame()

    close = (
        daily.pivot(index="date", columns="symbol", values="close")
        .add_suffix("_close")
        .sort_index()
    )
    volume = (
        daily.pivot(index="date", columns="symbol", values="volume")
        .add_suffix("_volume")
        .sort_index()
    )
    wide = close.join(volume, how="outer")

    if hl_symbols:
        for measure in ("high", "low"):
            piv = daily[daily["symbol"].isin(hl_symbols)].pivot(
                index="date", columns="symbol", values=measure
            ).add_suffix(f"_{measure}").sort_index()
            wide = wide.join(piv, how="outer")

    # Deterministic column ordering (measure grouped by symbol asc).
    return wide[sorted(wide.columns)]


def assemble_dataset(
    start: date,
    end: date,
    engine: Engine | None = None,
    include_intraday: bool = False,
    symbols: Iterable[str] | None = None,
    historical_provider: str | None = None,
    historical_adjust: bool = True,
    historical_start: date | None = None,
    historical_provider_fn: ProviderFn | None = None,
    hl_symbols: Iterable[str] | None = None,
) -> AssemblyResult:
    """Pull the board + rate/vol proxies and return daily bars pivoted wide.

    Symbols not present in ``staging.quote_raw`` (including possibly VIX and
    the rate proxies) are OMITTED with a note in ``symbols_missing``. This is
    deliberate: the harness must not fail on absent covariates because
    Pythia's coverage is expected to change as raptor's ingestion grows.

    ``symbols=None`` → the full board + all VIX/RATE candidates. Callers can
    pass an explicit set to test on a subset.

    D8 backfill (opt-in): when ``historical_provider`` is set (e.g. "yfinance"),
    historical daily bars for ``[historical_start or start, end]`` are fetched
    and stitched UNDER the raptor feed — raptor stays the truth for dates it
    covers, historical fills all the older bars. This fattens the thin ~214-obs
    sample. The wide output SCHEMA is unchanged (``{symbol}_close`` /
    ``{symbol}_volume`` + calendar features), so the covariate-lag gate and
    ffill-past-only logic downstream are untouched. With no provider the path is
    exactly as before and stays byte-deterministic. See docs/d8-backfill.md.
    """
    engine = engine or get_engine()
    if symbols is None:
        wanted = list(BOARD_SYMBOLS) + list(VIX_SYMBOLS) + list(RATE_SYMBOLS)
    else:
        wanted = list(symbols)

    intraday = _fetch_intraday(engine, wanted, start, end)
    daily_long = daily_bars_from_intraday(intraday)

    backfilled: tuple[str, ...] = ()
    if historical_provider is not None or historical_provider_fn is not None:
        hist_start = historical_start or start
        historical_long = fetch_historical_daily_bars(
            wanted,
            hist_start,
            end,
            provider=historical_provider or "yfinance",
            adjust=historical_adjust,
            provider_fn=historical_provider_fn,
        )
        if not historical_long.empty:
            backfilled = tuple(
                s for s in wanted if s in set(historical_long["symbol"].unique())
            )
            daily_long = combine_daily(daily_long, historical_long, prefer="raptor")

    # After backfill, "present" reflects everything now in the combined frame.
    present_combined = (
        set(daily_long["symbol"].unique()) if not daily_long.empty else set()
    )
    missing = tuple(s for s in wanted if s not in present_combined)
    included = tuple(s for s in wanted if s in present_combined)

    wide = _pivot_wide(daily_long, set(hl_symbols) if hl_symbols else None)

    # Calendar features (time-agnostic; keyed on date) go on the wide frame.
    wide = add_calendar_features(wide)

    return AssemblyResult(
        daily=wide,
        intraday=intraday if include_intraday else None,
        symbols_included=included,
        symbols_missing=missing,
        symbols_backfilled=backfilled,
    )


def write_dataset(result: AssemblyResult, out: Path) -> Path:
    """Persist the wide daily frame as Parquet. Deterministic file bytes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(out, engine="pyarrow", compression="zstd", index=True)
    return out
