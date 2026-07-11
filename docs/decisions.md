# Pythia decision log

Maintained by **helen** at each phase gate. Twin proposes; helen accepts,
amends, or rejects. Tony reviews at the final-panel validation.

## D1 — Covariate-lag gate (P1 hard invariant)

**Status:** proposed by twin 2026-07-11, awaiting helen verification.

**Decision:** Every observed covariate at forecast time t is sourced from
data timestamped **strictly before** the target's realisation at t.
Default P1 lag = 1 trading day. Known-future calendar features (dow,
month, days_to_fomc, is_earnings_season, etc.) are exempt — they're
causally future-safe.

**Enforcement:** structural, via `pythia.features.lag.LagPolicy` +
`build_features`. Any column not classified into
`{observed, known_future, target}` raises at build time.

**Test:** `tests/test_feature_lag_no_within_row_leakage.py` — 7 assertions
including a byte-level check that at row t, every lagged column equals
the raw source at row t-1.

**Follow-ups if helen amends:** target-side lag policy for range/vol (should
range at row t be aligned as "the range that happens BETWEEN t-1 and t"?);
choice of `lag` for the intraday P3 case (defer to P3).

---

## D2 — Two-target formulation

**Status:** proposed by twin 2026-07-11, awaiting helen verification.

**Decision:** P1 predicts TWO targets jointly (or as parallel heads on
one TFT):
- Return distribution (return_target: log px_t / px_{t-1}).
- Realised range (realized_range_target: log high_t / low_t) — first-class
  per helen's guidance "range/vol is more forecastable than direction."

**Rationale:** Direction is a martingale-plus-noise. Range is not. If P1
beats baselines anywhere on skill, it will be here — and calibration on
the return quantiles is what makes the P10-P90 band useful.

---

## D3 — Model registry contract (proposed)

**Status:** proposed by twin 2026-07-11, awaiting helen verification.

**Decision:** Model registry is a Postgres table keyed by
`(model_name, model_version)`, storing:
- `trained_at` (UTC),
- `dataset_hash` (SHA256 of the training Parquet bytes),
- `walk_forward_report_json` (the P0 harness output),
- `artifact_uri` (path to the saved Lightning checkpoint on shared PVC),
- `git_sha` (of the pythia repo commit that produced the checkpoint).

`SELECT ... ORDER BY trained_at DESC LIMIT 1` is the "current model" query.

---

## D4-D6

Reserved for helen.

## D7 — P1 covariate-lag gate verified + merged (2026-07-11)
The twin front-loaded the D3 gate as its own PR before building any model — the right order.
Structural enforcement in `src/pythia/features/lag.py`: a `LagPolicy` classifies EVERY column
(observed → lagged ≥1 row; known-future calendar → exempt; target → excluded from features),
and any **unclassified** column **raises at build time** (fail-loud — a silent leak becomes a
loud one). The test `test_feature_lag_no_within_row_leakage.py` asserts each feature at row t
equals its raw value at t−lag (no same-bar leak, e.g. `SPY_close_t` can't feed `QQQ_return_t`)
plus fail-loud on stray columns. **helen verified live: 7/7 gate test + 43/43 full suite pass.**
Merged as PR #1 (c75b7ab). Released the twin to P1 phase 2 (TFT trainer + model registry +
inference API + real backtest run). **Remaining P1 acceptance is unchanged and non-negotiable:**
the real training pass must be CALIBRATED (P10–P90 ≈ 0.80) and report skill-vs-baseline HONESTLY
— a null result vs random-walk is an acceptable, publishable outcome; overclaiming is not.

## D8 — Backfill historical daily bars (2026-07-11)
DATA GAP found in P1: raptor only began ingesting most of the macro board on 2026-06-05, so
the daily walk-forward had just **n=214 obs / 14 covariates** — too thin for a meaningful
verdict. **Decision:** backfill years of historical daily OHLCV for QQQ + the full 20-symbol
board from a free source (yfinance/stooq) into pythia's dataset (historical source for old
bars, raptor's live feed for recent). The covariate-lag gate + ffill-past-only apply
unchanged; it feeds the nightly retrain so the daily model and its verdict firm up
automatically. **Why:** a "null skill" verdict on 1 month of thin data is weak; on 5+ years it
is a real statement. Cheap and clearly correct. Routed to agent-2 (data lane).

## D9 — P1 daily-model verdict: calibrated, no edge vs random-walk — ACCEPTED (2026-07-11)
helen-verified from `data/report.json` (TFT-lite, n=214, 22 walk-forward splits):
- **Calibration PASS:** tft_lite cov80 = 0.780 (∈ [0.75, 0.85]); *better*-calibrated than
  random-walk (cov80 0.939, over-dispersed — report flags RW itself as miscalibrated).
- **Skill:** does NOT beat RW — CRPS 0.0097 vs RW 0.0087; MAE-skill −0.20; hit-rate 0.44.
  Null-to-negative point/CRPS skill = no forecasting edge on daily QQQ returns.
- **Leakage clean:** lag/leak tests pass in the real pipeline (8 passed).
**Verdict: ACCEPTED** — a scientifically valid, honest result, reported with zero overclaiming.
Exactly what the rails were for. Verdict is on thin data (see D8); re-assess after backfill.
Ffill-past-only (twin caveat 3) confirmed leak-safe. Released P1 phase-2b/c (registry + API +
nightly retrain), P2 (raptor panel), and P3 (intraday, agent-2).

## D10 — D8 backfill verified + merged; keep split-adjusted (2026-07-11)
agent-2 delivered the historical daily backfill (PR #5): yfinance primary (stooq JS-blocked
headless), raptor stays truth for recent (prefer=raptor on overlap), historical fills old bars.
**helen-verified:** full suite 57 pass incl. lag/leak + `test_historical_backfill.py` asserts the
backfill adds **rows not columns** → the covariate-lag gate is untouched. Board now 2018→2026,
2141 days/symbol; walk-forward **n=1869 / 89 splits** (was n=214). **Decisions:** (1) KEEP
split/div-**ADJUSTED** as default — unadjusted injects fake split-day returns (NVDA 10:1 = a fake
~−90% return); adjusted is correct for a return model, and the adjustment ≈1 near the 2026-06-05
cutover so no seam with raptor's raw recent data. (2) Determinism approved — backfill opt-in,
default path byte-deterministic, freezable via `compute_dataset_hash`. **Merged.** NEXT: re-run
the daily walk-forward on the fattened set → updated verdict (does the model show skill on 8
years, or is the null robust? — the genuinely interesting question).

## D11 — P1 phase 2b/c verified + merged; P1 COMPLETE (2026-07-11)
Twin's PR #3 (p1/registry-serve): **model registry** (versioned — (model_name, model_version) PK,
trained_at, `dataset_hash`, `walk_forward_report_json`, `artifact_uri`, `git_sha` = proper
provenance) + **FastAPI inference API** (`GET /latest` → per-horizon quantiles; `GET
/variable-importance` → TFT VSN weights = drivers; `/health`) + **nightly-retrain CronJob**
(09:15 UTC Mon-Fri, image `pythia-trainer:0.1.0`, runs `scripts.nightly_retrain`). helen-verified:
suite green (52 pass incl. lag/leak + `test_registry_serve`). Merged. **P1 is COMPLETE.**
**REQUIREMENT flagged:** the nightly retrain + the registry's 'latest' (what P2's panel serves)
MUST train on the **backfilled** (D8/D10) dataset (n=1869), NOT the thin opt-out default — else
the panel would show a data-starved model. Released **P2** (raptor daily forecast panel).
