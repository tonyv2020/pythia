# pythia

**Decoupled probabilistic forecasting harness for QQQ and the macro board.**

Pythia is the *evaluation foundation* — a dataset + a walk-forward backtest
harness + honest baselines + calibration-aware metrics. **P0 ships no model.**
Every model that arrives later (P1+, any repo) is graded against these
baselines on this harness, out-of-sample, or the number does not count.

Raptor is upstream (it produces the data and, later, a competing `p_move`);
raptor-intel is downstream (it will *render* pythia's outputs when a model
exists). Pythia does not depend on either at runtime beyond the source DB.

## Guarantees (P0 acceptance)

- **Reproducible dataset.** `scripts/assemble_dataset.py` pulls QQQ + the 19
  other macro-board symbols + calendar features from `staging.quote_raw`,
  writes tidy Parquet under `data/`. Same inputs → identical bytes.
- **Walk-forward, no look-ahead.** `pythia.backtest.splits` yields
  `(train_end, eval_start, eval_end)` tuples — expanding or rolling. No fit
  ever sees a bar past `train_end`.
- **Honest baselines.** `pythia.baselines` ships three: `random_walk` (zero
  mean, historical σ), `last_return` (drift = last observed return), and a
  documented `raptor_p_move` **adapter stub** (P0 does not fetch raptor's
  live p_move — that is a P1 wiring task; the stub raises so no baseline
  quietly returns fabricated numbers).
- **Calibration-first metrics.** `pythia.metrics` reports directional
  hit-rate AND P10–P90 coverage AND CRPS AND pinball loss AND MAE-vs-RW
  skill. A model that "beats direction 55%" but has 40% P10–P90 coverage
  will FAIL the eval — miscalibrated forecasts are worse than none.
- **Nothing that isn't measured.** No plotting, no dashboard, no model.
  P0 is deliberately unglamorous.

## Honesty rails (all phases, forever)

1. Walk-forward out-of-sample only. NO look-ahead. Splits are enforced in
   code, not by convention.
2. Every model is scored against `random_walk` and `last_return` (and, once
   P1 wires it, `raptor_p_move`).
3. A model does NOT have to beat random-walk on direction — that outcome
   is *allowed and reported*. Direction is nearly-unpredictable; the harness
   does not pretend otherwise.
4. Forecasts MUST be **calibrated** — P10–P90 coverage should be 80% ± 5pp
   on the eval window. If not calibrated, the run reports "MISCALIBRATED"
   and the model is not fit to trade against.
5. Never overclaim. If skill is 0, we ship "skill is 0."

helen owns the eval methodology + calibration review.

## Layout

```
src/pythia/
  data/         — DB pull + calendar features
  backtest/     — splits + harness
  baselines/    — random-walk, last-return, raptor-p_move stub
  metrics/      — MAE, hit-rate, calibration, CRPS, pinball
scripts/
  assemble_dataset.py    # DB → data/*.parquet
  score_baselines.py     # harness on baselines → report
tests/          — unit tests for splits + metrics (calibration is critical)
docs/
  methodology.md         # how a fair score is computed
  data-schema.md         # what's in the dataset and where it came from
  p0-acceptance.md       # what "done" means and how to verify it
data/           # gitignored — parquet lands here
```

## Quickstart

```sh
python -m pythia.scripts.assemble_dataset --start 2024-01-01 --out data/board_2024_onwards.parquet
python -m pythia.scripts.score_baselines --dataset data/board_2024_onwards.parquet --report report.json
pytest -q
```

Postgres source: `postgres.hollywood.svc.cluster.local:5432/raptor`,
schema `staging.quote_raw`. Set `PYTHIA_DB_DSN` for anything else.

## Status

- [x] P0 dataset assembler
- [x] P0 walk-forward harness
- [x] P0 baselines (random_walk, last_return, raptor_p_move stub)
- [x] P0 metrics (MAE, hit-rate, coverage, CRPS, pinball)
- [ ] P1 raptor `p_move` live adapter (out of scope for P0)
- [ ] P1+ actual models (out of scope for P0)
