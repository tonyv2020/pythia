# P0 acceptance — what "done" means and how to verify

## Ariadne task
**Pythia P0 — dataset + walk-forward backtest harness + baselines**
`task_id 591a4f12-eb8c-428a-b8de-250e3623fa6b`

## Acceptance criteria (task text)

> reproducible dataset + a harness that scores ANY model vs baselines
> out-of-sample with calibration+skill. helen reviews for leakage +
> honest baselines.

## Objective checklist

- [x] Repo `tonyv2020/pythia` exists (decoupled — own repo, own pipeline).
- [x] Data assembler pulls target + macro board + calendar features from
      `staging.quote_raw` (+ optional VIX/rates, omitted if absent).
- [x] Walk-forward splits: expanding + rolling, both prohibit look-ahead
      via `WalkForwardSplit.__post_init__`.
- [x] Baselines: `RandomWalk`, `LastReturn`, and a **stub** `RaptorPMoveStub`
      that raises rather than fabricates (documented deliberate choice).
- [x] Metrics: MAE, RMSE, MAE-skill-vs-baseline, directional hit-rate,
      P10-P90 coverage (calibration), CRPS-under-Normal, pinball @ q ∈
      {0.10, 0.50, 0.90}.
- [x] Harness `run_backtest` fits+scores every model on every split and
      returns per-model `Report` with `warnings` populated on
      miscalibration.
- [x] CLIs: `assemble_dataset.py` (DB → Parquet), `score_baselines.py`
      (Parquet → report.json).
- [x] Tests: splits leak-free, metrics honest, baselines rejected on short
      train, harness end-to-end.
- [x] Docs: methodology, data-schema, this file.

## Non-goals (P0)

- Any actual model. Baselines only.
- Live raptor `p_move` scoring — deferred to P1 (see `RaptorPMoveStub`).
- Plotting or UI.
- Live scraping of FOMC / earnings dates.

## How helen verifies

1. `git clone tonyv2020/pythia && cd pythia`
2. `pip install -e '.[dev]'`
3. `pytest -q` — all tests should pass.
4. `PYTHIA_DB_DSN=postgresql://…/raptor python -m scripts.assemble_dataset \
       --start 2024-01-01 --end 2026-07-10 \
       --out data/board_2024_onwards.parquet`
5. `python -m scripts.score_baselines \
       --dataset data/board_2024_onwards.parquet \
       --report report.json`
6. Read `report.json` and confirm:
   - Both baselines produce reports with finite metrics.
   - `random_walk.coverage_80` is in `[0.75, 0.85]` on the eval window
     (calibrated). If not, that's a diagnostic finding about the target's
     tail behaviour, not a harness bug.
   - `last_return.mae_skill_vs_rw` exists and is close to 0 (persistence
     rarely helps on log-returns; a large positive value would be
     suspicious and worth investigating).

## Leakage review checklist (for helen)

- `slice_train_eval` returns disjoint frames — proven by
  `test_slice_train_eval_returns_disjoint_frames`.
- `_target_returns` computes on the whole frame BUT the harness picks
  `y_true = returns.loc[eval_frame.index]` — no leakage; eval y's are
  fixed derived-from-the-truth values.
- Models receive `train` frame only in `.fit()`; the eval index is passed
  to `.predict()` but the frame is not. It is structurally impossible for
  a Pythia model to observe an eval-window price under the current
  interface.
- Calendar features are computed from the date alone, no lookahead by
  construction.
