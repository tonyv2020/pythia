# Pythia — Decision Log

Maintained by **helen** while driving Pythia from P0 through deployment. Tony reviews
this at final-panel validation in raptor. Newest at the bottom. Each entry: the decision,
the rationale, and (where relevant) what would reverse it.

---

## D1 — Pythia is a decoupled service (2026-07-11)
Own repo (`pythia`): data assembler → backtest harness → model registry → inference API.
raptor-intel only **renders** the forecast (consumes the inference API); it owns no model
code. **Why:** forecasting is a distinct concern from the dashboard; keeps raptor clean and
the model reusable. **Reverses if:** the API round-trip proves too slow for the intraday
panel and we need in-process inference (unlikely).

## D2 — Honesty-first evaluation (2026-07-11, P0; authored by twin, endorsed by helen)
Metric priority: **Coverage (P10–P90 ≈ 0.80) > CRPS > pinball > MAE-skill-vs-RW >
directional hit-rate** (last; 50% is the martingale prior). Baselines (random-walk =
mean-0 / train-vol; last-return; raptor `p_move`) are the floor. A model that ties
random-walk = a **null result, reported as such**. No "beats the market" claim anywhere.
**Why:** short-horizon price *direction* is ~random-walk; the honest value is calibrated
uncertainty + range/vol, not direction.

## D3 — P1 covariate-lag gate (2026-07-11) — HARD ACCEPTANCE CONDITION
The TFT must not use any covariate value contemporaneous with the target it predicts.
Either **lag observed covariates to ≤ t−1** relative to target `y_t`, or **frame as
next-step** (past covariates → future target — the standard TFT encoder/decoder split).
Known-future calendar/FOMC features are exempt (known ahead). Requires an **explicit
within-row as-of / feature-lag test** — the existing `test_splits_no_lookahead` only guards
the temporal *split*, not within-row feature/target alignment. **Why:** using `SPY_close_t`
to "predict" `QQQ`'s same-bar return `y_t` is textbook same-bar leakage that inflates the
backtest and the current tests would NOT catch it.

## D4 — Forecast targets: return-distribution + realized range/vol (2026-07-11)
P1 forecasts the QQQ log-return **quantile distribution** AND **realized range/volatility**,
with range/vol as a **first-class** target. **Why:** vol clusters and is genuinely more
forecastable than direction — it's where the model can show real, honest skill even if
direction has no edge.

## D5 — Two-agent division on achilles (2026-07-11)
Twin leads P0/P1 and takes **P2** (raptor forecast panel — frontend lane). **agent-2** takes
**P3** (intraday TFT — backend/streaming-model lane) once P1 unblocks. Independent after P1;
they converge at **P4** (the overlay). **Why:** P2/P3 are independent post-P1; matches each
agent's recent lane; parallelizes without collision.

## D6 — Daily first, intraday second, overlay last (2026-07-11)
P1 daily (1–5 day) → P3 intraday (30–60 min) → P4 overlaid panel. **Why:** daily has years
of history and a cleaner backtest to prove the pipeline before the noisier intraday model.
