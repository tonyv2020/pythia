"""Historical daily-bars backfill source (D8).

raptor's ``staging.quote_raw`` only covers the macro board since ~2026-06-05,
so the P0 walk-forward runs on ~214 thin observations and the daily verdict is
under-powered. This module backfills YEARS of historical daily OHLCV for the
board from a free public source, as the OLD bars; the live raptor feed remains
the truth for RECENT bars. The two are stitched in ``combine_daily``.

Provider
--------
``yfinance`` (Yahoo) is the working default. A ``stooq`` CSV adapter is also
provided, but Yahoo is primary: stooq serves a JS anti-bot interstitial to
headless clients (verified 2026-07-11), so it is a best-effort fallback only.

Adjustment
----------
Prices are SPLIT- and DIVIDEND-adjusted by default (``adjust=True``). Over
multi-year history several board names split (NVDA 10:1 2024, TSLA 3:1 2022,
GOOG & AMZN 20:1 2022, AAPL 4:1 2020); using RAW prices would inject fake
~90% split-day "returns" into the covariates and corrupt every returns-based
feature. Adjustment factors are ~1.0 near the 2026-06-05 raptor cutover, so the
adjusted history joins raptor's RAW recent feed with no material seam. Pass
``adjust=False`` to match raptor's raw convention exactly — a documented
tradeoff; helen owns the methodology call (see docs/d8-backfill.md).

Determinism
-----------
Unlike the pure-Postgres assembler (byte-deterministic given fixed
``quote_raw``), output here depends on the external provider's current
snapshot. Backfill is therefore OPT-IN; the default ``assemble_dataset`` path
is unchanged and stays deterministic.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Iterable

import pandas as pd

# Long-format columns, IDENTICAL to ``daily_bars_from_intraday`` output, so the
# backfilled bars flow through ``_pivot_wide`` + ``add_calendar_features``
# unchanged — the assembler's wide output schema is preserved exactly.
DAILY_COLUMNS: list[str] = ["symbol", "date", "open", "high", "low", "close", "volume"]

# provider(symbol, start, end, adjust) -> DataFrame indexed/keyed by date with
# columns {open, high, low, close, volume}. Kept as a small callable so tests
# inject a deterministic fake and never touch the network.
ProviderFn = Callable[[str, date, date, bool], pd.DataFrame]


def _normalize_provider_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider's per-symbol frame to ``[date, open, high, low, close,
    volume]`` (no ``symbol`` column yet). Tolerant of yfinance's MultiIndex
    columns and mixed capitalisation."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = raw.copy()

    # yfinance returns a (measure, ticker) MultiIndex for a single ticker —
    # flatten to the measure level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Date lives in the index for yfinance; make it a column.
    if "date" not in {str(c).lower() for c in df.columns}:
        df = df.reset_index()

    df.columns = [str(c).lower() for c in df.columns]
    # yfinance names the index "date" after reset; some providers use "datetime".
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    wanted = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"provider frame missing columns {missing}; got {list(df.columns)}")

    df = df[wanted].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows with no close (holidays / provider gaps).
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def _yfinance_provider(symbol: str, start: date, end: date, adjust: bool) -> pd.DataFrame:
    """Yahoo daily bars via yfinance. ``end`` is inclusive (yfinance's is
    exclusive, so we add a day)."""
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=adjust,
        actions=False,
        progress=False,
        threads=False,
    )
    return _normalize_provider_frame(raw)


def _stooq_provider(symbol: str, start: date, end: date, adjust: bool) -> pd.DataFrame:
    """Best-effort stooq CSV adapter. NOTE: stooq serves a JS interstitial to
    headless clients, so this typically returns empty outside a browser; kept
    for environments (or a future proxy) where the CSV endpoint is reachable.
    stooq daily is split+div adjusted already, so ``adjust`` is advisory."""
    import io
    import urllib.request

    s = symbol.lower()
    if "." not in s:
        s = f"{s}.us"
    url = (
        f"https://stooq.com/q/d/l/?s={s}"
        f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        body = r.read().decode("utf-8", errors="replace")
    if not body.lstrip().lower().startswith("date"):
        # JS interstitial / error page, not CSV.
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return _normalize_provider_frame(pd.read_csv(io.StringIO(body)))


_PROVIDERS: dict[str, ProviderFn] = {
    "yfinance": _yfinance_provider,
    "stooq": _stooq_provider,
}


def fetch_historical_daily_bars(
    symbols: Iterable[str],
    start: date,
    end: date,
    provider: str = "yfinance",
    adjust: bool = True,
    provider_fn: ProviderFn | None = None,
) -> pd.DataFrame:
    """Fetch historical daily OHLCV for ``symbols`` in ``[start, end]``.

    Returns a long frame with ``DAILY_COLUMNS``. Symbols the provider cannot
    return are skipped (recorded by the caller via set difference); a partial
    backfill is better than a hard failure, matching the assembler's
    omit-missing philosophy.

    ``provider_fn`` overrides the named provider — the test seam, so unit tests
    inject deterministic fixtures and never hit the network.
    """
    fn = provider_fn or _PROVIDERS[provider]
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            one = fn(sym, start, end, adjust)
        except Exception:
            # Unreachable symbol / transient provider error — skip, don't crash
            # the whole backfill. Caller sees it as missing.
            continue
        one = _normalize_provider_frame(one) if "symbol" not in getattr(one, "columns", []) else one
        if one is None or one.empty:
            continue
        one = one.copy()
        one["symbol"] = sym
        frames.append(one[DAILY_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def combine_daily(
    raptor_long: pd.DataFrame,
    historical_long: pd.DataFrame,
    prefer: str = "raptor",
) -> pd.DataFrame:
    """Stitch raptor (recent, live truth) and historical (old backfill) long
    daily frames into one, deduped on ``(symbol, date)``.

    ``prefer='raptor'`` (default): where both sources have a bar for the same
    ``(symbol, date)``, keep raptor's — it is the deployed source of truth and
    matches production. Historical fills every ``(symbol, date)`` raptor lacks
    (all the old history, and any recent symbols raptor hasn't ingested yet).
    """
    if prefer not in ("raptor", "historical"):
        raise ValueError("prefer must be 'raptor' or 'historical'")

    raptor = raptor_long if raptor_long is not None else pd.DataFrame(columns=DAILY_COLUMNS)
    hist = historical_long if historical_long is not None else pd.DataFrame(columns=DAILY_COLUMNS)
    for frame in (raptor, hist):
        for col in DAILY_COLUMNS:
            if col not in frame.columns:
                frame[col] = pd.Series(dtype="object")

    # Order the concat so the PREFERRED source is kept by drop_duplicates.
    ordered = [hist[DAILY_COLUMNS], raptor[DAILY_COLUMNS]]
    if prefer == "historical":
        ordered = ordered[::-1]

    combined = pd.concat(ordered, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    combined = combined.drop_duplicates(subset=["symbol", "date"], keep="last")
    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)
