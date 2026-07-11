# D8 — Historical daily-bars backfill

**Task:** `caa2b7f5` (Ariadne WBS P1.1). **Author:** agent-2 (achilles).
**Decision owner (methodology):** helen.

## Why

raptor's `staging.quote_raw` has only carried the macro board since
~2026-06-05, so the P0/P1 daily walk-forward ran on ~214 thin observations —
too few for a defensible calibration/skill verdict (the null-skill result was
under-powered, not necessarily wrong). D8 backfills **years** of historical
daily OHLCV for the board so the daily model + honest verdict firm up on a real
sample.

Live proof (2018-01-01 → 2026-07-10, full 20-symbol board, yfinance): **2141
trading days/symbol**, and the baseline walk-forward goes from a handful of
splits to **89 splits / n=1869**.

## What it does

`pythia.data.historical.fetch_historical_daily_bars(symbols, start, end,
provider="yfinance", adjust=True)` returns long daily bars in the **exact same
schema** as `daily_bars_from_intraday` (`symbol, date, open, high, low, close,
volume`). `assemble_dataset(..., historical_provider="yfinance",
historical_start=<years-back>)` stitches them **under** the raptor feed via
`combine_daily(prefer="raptor")`:

- **raptor stays the truth for every `(symbol, date)` it covers** (the
  deployed, production source).
- **historical fills every bar raptor lacks** — all the old history, plus any
  recent symbol raptor hasn't ingested yet.

The wide output schema (`{symbol}_close`, `{symbol}_volume` + calendar
features) is **unchanged**, so the P1 covariate-lag gate and the
ffill-past-only feature logic downstream are **untouched** — backfill adds
*rows*, never *columns*. A test asserts this (`test_backfill_preserves_wide_
schema_and_adds_rows`).

## Decisions for helen

### 1. Split/dividend adjustment (default: ON)

Over multi-year history several board names split: **NVDA 10:1 (2024-06),
TSLA 3:1 (2022), GOOG & AMZN 20:1 (2022), AAPL 4:1 (2020)**. RAW prices would
inject fake ~90% split-day "returns" into the covariates and corrupt every
returns-based feature. So the default is **split+dividend adjusted**
(`adjust=True`).

- Adjustment factors are ≈1.0 near the 2026-06-05 raptor cutover, so the
  adjusted history joins raptor's RAW recent feed **with no material seam**.
- Verified: NVDA adjusted max |daily return| over 2018–2026 = **0.244** (i.e.
  no fake split jump; raw would show ≈ −0.90 on 2024-06-10).
- Pass `--no-adjust` to match raptor's raw convention exactly if you prefer
  that tradeoff. **Your call — flag me if you want the default flipped.**

### 2. Determinism

The pure-Postgres assembler is byte-deterministic given fixed `quote_raw`.
With an external provider, output depends on the provider's current snapshot,
so **backfill is opt-in** and the default path is unchanged. For a frozen,
reproducible research set, run the backfill once and commit the resulting
Parquet's SHA-256 to the model registry (the P1 `compute_dataset_hash` already
does this).

### 3. Provider

`yfinance` (Yahoo) is the working default. A `stooq` CSV adapter is included
but Yahoo is primary — stooq serves a JS anti-bot interstitial to headless
clients (verified 2026-07-11), so it is a best-effort fallback only. The
provider is a small injectable callable, so tests use a deterministic fake and
never touch the network.

## How to run (on achilles, with raptor DB reachable)

```sh
pip install -e '.[dev,backfill]' scipy

# Fatten the dataset: historical from 2018, raptor feed for recent.
PYTHIA_DB_DSN=postgresql://…/raptor \
  python -m scripts.assemble_dataset \
    --start 2026-06-01 --end 2026-07-10 \
    --historical yfinance --historical-start 2018-01-01 \
    --out data/board_backfilled.parquet
# manifest prints symbols_included / symbols_backfilled / historical_adjusted

# Re-run the honest walk-forward on the fattened set.
python -m scripts.score_baselines \
  --dataset data/board_backfilled.parquet --target QQQ_close \
  --initial-train 252 --eval-size 21 --report data/report_backfilled.json

# Then the P1 TFT retrain on the fattened set (2080 Ti) for the firmed verdict.
```

Note `--start` bounds the raptor (recent) pull; `--historical-start` bounds the
backfill. Set `--historical-start` years back to fatten; the raptor feed still
overrides wherever it has bars.
