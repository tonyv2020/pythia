# Pythia P0 methodology

The scoring pipeline for any model — baseline or otherwise.

## The one invariant

**A model NEVER sees a data row past ``train_end`` of the split it is being
fit on.** Every other rule flows from this. The harness enforces it by
handing the model a `(train_frame, eval_index)` pair — the eval frame's y
values are computed by the harness, not the model, so there is no source
of eval-time truth to leak.

## Target definition

The target is the one-step log-return of ``QQQ`` (or whatever
``--target`` column is passed):

    y_t = log(px_t) - log(px_{t-1})

The first row's y is NaN (no prior close) and is dropped from scoring —
never zero-imputed.

## Splits

`expanding_walk_forward(index, initial_train_size=252, eval_size=21)` yields
the P0 default: each split trains on ALL history up to `train_end`, evaluates
on the next 21 rows, then slides by 21 rows. On a 3-year daily series with
`--initial-train 252 --eval-size 21` that is ~24 splits worth of scoring.

`rolling_walk_forward(index, train_size=756, eval_size=21)` is available for
models that assume a finite memory.

## Metrics — in ranked order of importance

1. **Coverage (P10–P90)** — target 0.80 ± 0.05. If a model is outside this
   band it is MISCALIBRATED and the report says so; downstream sizing (any
   trade allocation using σ) is invalid.
2. **CRPS** — proper score for the full predictive distribution under Normal.
   Lower is better. This is the single number to compare distinct models.
3. **Pinball loss @ q ∈ {0.10, 0.50, 0.90}** — how tight the quantile-specific
   forecasts are. Pinball @ 0.5 == MAE (up to constant).
4. **MAE and MAE-skill vs random-walk** — point-forecast comparison.
5. **Directional hit-rate** — reported for completeness. Not decisive. A
   50% hit-rate is the fair prior for a martingale target.

## Interpreting a result

- A model that beats random-walk on CRPS *and* is calibrated (P10–P90 ∈
  [0.75, 0.85]) is a real improvement. Report it.
- A model that beats random-walk on directional hit-rate but is
  miscalibrated is a hazard. The report will flag it and the recommended
  action is to fix calibration before believing the hit-rate.
- A model that ties random-walk on everything is a null result — report it
  as such. That is a scientifically valid outcome and honest.

## What Pythia does not do (P0)

- Fit any model. RandomWalk / LastReturn are baselines, not "models under
  test." They exist to be the floor.
- Plot anything. `report.json` is machine-readable; visualisation is
  raptor-intel's job later.
- Scrape live FOMC or earnings dates. The FOMC list is a frozen snapshot in
  `pythia.data.calendar_features.FOMC_DATES`; update it in a PR.
- Beat the market. It doesn't claim to.
