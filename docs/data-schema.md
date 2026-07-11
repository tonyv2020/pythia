# P0 dataset schema

## Source

`postgresql://…/raptor?schema=staging`, table `staging.quote_raw`. Columns
Pythia consumes: `symbol`, `date`, `time`, `last_trade_price`, `volume`,
`open_price`. Ingestion cadence: intraday ticks — sparse-sampled (~20/day
verified 2026-07-11). Coverage varies by symbol (see `symbols_missing` in
the assembler manifest).

## Assembly

The assembler rolls intraday ticks up to per-symbol per-date OHLCV bars
(open = first tick / vendor open if present, close = last tick, high/low =
intraday max/min, volume = sum), then pivots to wide:

    index = date
    columns = QQQ_close, QQQ_volume, SPY_close, SPY_volume, ..., dow, month,
              dom, is_monday, is_friday, is_month_end, is_quarter_end,
              days_to_fomc, is_earnings_season

## Symbol set

Target: **QQQ**.

Covariates (macro board, 19 of them):
`SPY, DIA, IWM, AAPL, MSFT, NVDA, GOOG, AMZN, META, TSLA, GLD, SLV, GDX,
USO, UGA, UNG, DBE, CORN, WEAT`

Optional (looked up but omitted if raptor doesn't ingest them yet):
`^VIX`/`VIX`/`VIXY` (vol proxy), `^TNX`/`TNX`/`IEF`/`TLT` (rates proxy).

## FOMC calendar

Hardcoded in `pythia.data.calendar_features.FOMC_DATES`, spanning
2023-01-01 → 2026-12-31 as published by the FRB. To extend beyond 2026,
add the newly-published dates to that tuple and bump the version — never
scrape live, because the dataset must be replayable.

## Earnings season

Heuristic: True for a date `d` when days-since-most-recent-quarter-end is
in `[15, 49]`. Not a per-stock earnings calendar; that would require
company-level truth and would fatten P0 well past its scope. Refined
per-company earnings dates are a P2+ concern.

## Deterministic bytes

Same DB state → same Parquet bytes. Row order is `date` ascending; column
order is lexicographic within each measure. No timestamps embedded in the
Parquet metadata (via `pd.to_parquet(compression='zstd')`).
