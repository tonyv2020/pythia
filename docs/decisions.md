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
