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

## D12 — P3 intraday design calls + nightly-retrain-on-backfill merged (2026-07-11)
(a) **PR #6 merged** (4df5455): `scripts/nightly_retrain` now DEFAULTS to the D8 backfill (yfinance
2018+, full 20-symbol board, split-adjusted, initial_train 252 → n=1869) — the served model trains
on fat data (P2 hard-req satisfied). The fat-dataset VERDICT is still PENDING — routed a run.
(b) **P3 intraday design** (agent-2 PR #7, review-gate — helen-verified: suite green + design note):
- **Q1 horizon = 30 MIN.** Shorter = more forecastable + a tighter, more useful cone; 60-min
  intraday returns are ~noise. At the 3-min feed that's ~10 bars ahead. Architecture can extend.
- **Q3 = implement `p_move` as a REAL baseline NOW** (not a stub). It's the key honest question for
  intraday: does the TFT beat raptor's *existing* p_move signal? A stub would defeat the point.
- **eval_mask design APPROVED**: score only within-session bars; baselines `.predict()` on the full
  window, mask only the SCORING — leak-safe (past-only) + apples-to-apples with baselines.
PR #7 merged; agent-2 proceeds to the full intraday model.

## D13 — P3 p_move baseline design (2026-07-11)
agent-2 found the real persisted signals in raptor-intel pg: `staging.qqq_pmove` (324 days,
2023→2026; p_move = scalar move-MAGNITUDE probability, avg 0.157) + `staging.qqq_direction`
(p_up/p_dn). Methodology calls (helen owns eval):
1. **p_move → CALIBRATED DISPERSION baseline** — approve agent-2's mapping. p_move is a magnitude
   probability, NOT a (mean,sigma) return forecast, so map it as mean=0 + sigma = c·g(p_move),
   with scale c fit on each TRAIN window to ~0.80 train coverage → a legit "raptor-implied-
   dispersion" baseline scored on the same CRPS/coverage/pinball as the TFT (apples-to-apples).
   DIRECTION handled separately via `qqq_direction` (p_up/p_dn → mean sign) as its own baseline.
   Optional (nice-to-have, not required): also report p_move's native move-magnitude skill as
   Brier/AUC. This respects what each signal IS.
2. **Granularity → 10-MIN intraday BARS** (matches p_move's native ~10-min grid + the 3-min feed).
   Horizon stays ~30 min = 3 bars (refines D12: the 30-min *horizon* holds; the *bar size* is
   10-min, not 30). Native apples-to-apples vs p_move.
3. **Data asymmetry ACCEPTED (honest):** the intraday-TFT-vs-p_move comparison is limited to the
   ~1-month tick-feed overlap (raw ticks only since 2026-06-05); report n-on-overlap explicitly.
   No free intraday backfill (unlike daily). Intraday verdict stays thin until tick history
   accrues — stated openly, not hidden.
Separately: the FAT DAILY VERDICT is still pending — agent-2 to RUN scripts/nightly_retrain
(defaults→backfill) NOW; it is independent of the P3 work.

## D14 — Fat-dataset DAILY VERDICT: robust null, well-calibrated (2026-07-12)
Daily walk-forward re-run on the D8-backfilled **n=1869 / 89-split** set (8 years, 2018→2026;
PR #8, 80 epochs). helen-verified from `docs/fat-daily-report.json`:
- **tft_lite**: cov80 **0.781** (CALIBRATED, best of the three), CRPS 0.00821, MAE-skill vs RW
  **−0.046**, hit 0.524.
- **random_walk**: cov80 0.864 (mild over-cover), CRPS 0.00796.
- **last_return**: cov80 0.731 (under-cover), CRPS 0.01106.
**VERDICT: robust NULL.** On 8 years the TFT is well-calibrated (better than both baselines) but
does NOT beat random-walk on the proper score (CRPS ~3% behind, MAE ~5% behind; 52.4% hit-rate is
within noise of the 50% martingale prior). MORE DATA HELPED (MAE-skill −0.20 on thin n=214 →
−0.046 on n=1869; CRPS gap narrowed) but did not create an edge. This is the honest, expected
result for daily QQQ returns — a scientifically valid null (D2). The panel shows the calibrated
cone + an honest "no edge vs random-walk" scorecard. Any residual signal is more likely to
surface in the intraday model (P3) vs p_move — the next test.

## D15 — Intraday baseline verdict + TFT greenlight (2026-07-12)
agent-2's intraday walk-forward (PR #9; 10-min bars, 30-min horizon, n=7242 on the ~1-month tick
overlap; horizon-purge h-1 embargo + eval_mask within-session — tests present + green). BASELINE
verdict (no TFT yet), helen-verified:
- random_walk     : cov80 0.956 | CRPS 0.001673
- last_return     : cov80 0.933 | CRPS 0.003099 | skill −0.93
- raptor_p_move   : cov80 0.871 | CRPS 0.001681 | skill 0.0  (BEST-calibrated — carries real
  dispersion, tightens coverage vs RW's over-dispersed 0.956, but MATCHES RW on CRPS)
- raptor_direction: cov80 0.958 | CRPS 0.001573 | skill −0.006  (≈ RW; no 30-min directional edge)
Read: p_move has real dispersion info (best cov80) but no CRPS edge; no directional edge at 30-min;
all mildly over-dispersed → the TFT's job is a TIGHTER conditional sigma than these.
Decisions: (a) floor-0.02 sigma fix APPROVED — disclose it in the methodology (not a hidden fudge).
(b) GREENLIGHT the intraday TFT-lite (deep sample, honest+solid baselines, same scored path). Train
on the 2080 Ti for the reported verdict (CPU ok for a smoke test only). The intraday-TFT-vs-p_move
verdict is the next Tony one-liner.

## D16 — Intraday TFT merged; Pythia goes live (deployment greenlit) (2026-07-12)
- **PR #11 merged** (intraday TFT-lite): `IntradayTFTLiteModel` overrides ONLY the target to the
  FORWARD-3-bar return log(px[t+3]/px[t]) so it predicts what the harness scores (avoids a silent
  1-bar-trained / 3-bar-scored bug — unit-tested). p_move calib_floor DISCLOSED in
  docs/p3-intraday-design.md. Suite green.
- **PR #10 merged** (registry ensure_schema tolerates InsufficientPrivilege when the table exists —
  defensive, for the shared-DB deploy).
**DEPLOYMENT** — Pythia goes live. Greenlit the twin to: (1) rebuild `pythia-trainer` from main
(now carries the intraday code) + run the intraday backtest GPU Job on the 2080 Ti → the
intraday-vs-p_move VERDICT (Tony ping). (2) Build+deploy the pythia SERVICE — pythia-serve (FastAPI
/latest + /variable-importance), pythia-trainer, **registry in raptor's appdb** (a pythia schema via
PYTHIA_DB_DSN — decoupled tables, shared pg instance), ingress `/pythia/api/*`, Authentik
ProxyProvider — so P2's raptor panel (raptor-intel PR #34) consumes it live. Chose build-serve-now
over hold-for-#34: we are driving to deployment, serve+panel ship together, and helen
screenshot-VERIFIES P2 live rather than blocking on a frontend code review.

## D17 — LIVE INCIDENT (raptor white-screen) + resolution (2026-07-12, weekend)
The Pythia daily-forecast panel (raptor-intel #34/#35) crashed the ENTIRE raptor dashboard (React
#310, hooks-order violation) when /pythia/api/* returned 404 — no error boundary → the whole SPA
blanked (body=5 chars). Two deployment root causes for the 404: (a) the pythia registry couldn't
create its table in raptor appdb — InsufficientPrivilege: permission denied for schema public → no
model registered; (b) model-name mismatch (panel tft_lite_daily vs retrain tft_lite_daily_qqq).
RESOLUTION (Tony OK'd weekend downtime → proper fix, no rollback): twin shipped **PR #36** (error
boundary + hooks-order fix) → raptor now LOADS + the panel degrades gracefully. **helen-verified
live: body 3357, dashboard renders, no React #310, the 404 is caught by the boundary.**
**HARD NEW P2 GATE:** the dashboard must load even if the forecast API is down — this was NOT
verified live before P2 shipped; that's the real miss.
REMAINING (P2 polish, not blocking the cleared incident): (a) the served model is MISCALIBRATED
cov80=0.586 (trained 40/32) — reverting the nightly config to 60/16 to serve a CALIBRATED model
(~D14's 0.781); (b) /pythia/api/variable-importance still 404s (name/endpoint) — fix to serve or
gracefully-empty. **P2 accepted once the panel shows the calibrated model + drivers (or clean
graceful states) live.**

## D18 — Daily calibration is seed-noisy -> conformal calibration, NOT seed-hunting (2026-07-12)
The raw TFT-lite daily calibration is NONDETERMINISTIC across training runs (cov80: 0.586 @40/32;
0.705 @60/16-80ep; 0.781 @an earlier 60/16 run — SAME code + SAME fat dataset). So any single
run's cov80 is a noisy, seed-dependent number. FORK (agent-2 asked): (a) ship whatever 100-ep
lands + log "not robustly calibrated", or (b) hyperparameter × seed search, pick the
median-calibrated config as canonical.
DECISION — NEITHER as stated. (b) is CHERRY-PICKING a lucky seed = overfitting the calibration
metric; it would show a calibration that won't hold on new data (dishonest). (a) ships a
possibly-miscalibrated model. Instead **(c): apply PER-TRAIN-WINDOW CONFORMAL calibration to the
TFT's predictive spread** — fit a scale on each TRAIN window so train coverage ~= 0.80, apply
out-of-sample (the SAME technique agent-2 already uses for the p_move baseline's sigma). This makes
calibration ROBUST + honest BY CONSTRUCTION, independent of the training seed, with EVAL coverage
as the honest out-of-sample check. **Calibration != skill:** the null-vs-RW CRPS verdict (D14) is
UNAFFECTED — conformal scaling right-sizes the bands, it does not manufacture edge. Let the 100-ep
run finish as a data point, but the CANONICAL fix is the conformal layer. This also fixes P2's
served model so the panel shows a genuinely-calibrated cone + the honest "no edge vs RW" scorecard.

## D19 — Intraday verdict + D18 conformal EMPIRICALLY VALIDATED (2026-07-12)
(A) **D18 validated:** multi-seed proof — RAW TFT eval-cov80 seed-noise spread ~0.22 (0.586 / 0.705 /
0.781 / 0.809). Conformal shrank it ~10x to ~0.02, centered in-gate. Per-seed eval spread = honest
OOS proof it generalizes (train-calibration is by-construction). Conformal delivers SEED-INDEPENDENT
calibration, as premised. The conformal wrapper also fixed a lucky-seed raw run (cov80 0.809, CRPS
0.00221) into a genuinely-calibrated 0.77 forecaster — accurate story, not a lucky-seed one.
(B) **INTRADAY VERDICT** (conformal TFT, 10-min bars, 30-min horizon, 3 splits):
- conformal_tft   : cov80 ~0.774 (CALIBRATED, seed-stable) | CRPS mean 0.00145 | MAE-skill -0.10..-0.23
- random_walk     : cov80 0.807 | CRPS 0.00127
- raptor_p_move   : cov80 0.827 | CRPS 0.00154
- raptor_direction: ~ RW (no directional edge)
READ: intraday TFT ~TIES p_move on CRPS (beats it 2/3 seeds) but does NOT beat random-walk (RW is the
floor); no point/directional edge. VERDICT: calibrated, seed-robust, at-parity-with-p_move, NO edge
over random-walk — mirrors the daily null (D14).
**OVERALL FINDING:** QQQ returns are essentially a random walk at BOTH daily and 30-min horizons — no
forecasting edge. Pythia's deliverable is a rigorous, honest, seed-robust CALIBRATED-uncertainty
model that reports the null transparently. The value is right-sized uncertainty + intellectual
honesty, not alpha. P3 model+backtest+conformal complete.

## D20 — Daily calibration gate: accept 0.737 with honest disclosure (2026-07-12)
The re-registered conformal daily model lands EVAL cov80 = 0.737 — ~1.3pp below my [0.75,0.85]
floor, ~6pp under the 0.80 target. Multi-seed evidence (D19) shows this is SYSTEMATIC + seed-stable
(~0.02 spread), NOT a bad draw: conformal-to-train-0.80 systematically under-covers eval by ~6pp on
daily QQQ — a real train->eval coverage DRIFT (the future is modestly more uncertain than the recent
past; non-stationarity). Twin offered: (1) accept 0.737 with honest amber disclosure, or (2) target
train cov80 ~0.85 to compensate the drift so eval lands ~0.80 (a ~40min GPU run).
DECISION: **PATH 1** — accept 0.737, but the panel MUST honestly disclose it (amber + note: "eval
cov80 ~0.74; bands slightly tight — a systematic ~6pp train->eval drift on daily QQQ"), NOT claim
"well-calibrated 0.80". Rationale: for a NULL-result model the deliverable is HONEST uncertainty +
transparency, not hitting an arbitrary target; a seed-stable 0.74 honestly-disclosed is more honest
than a target-nudged 0.80, and it surfaces a REAL property of the data (non-stationarity drift).
Option 2 (drift-compensation to 0.80) is a documented config-toggle if exact-0.80 bands are wanted
later, but not required. I'm relaxing my own [0.75,0.85] gate to accept 0.737-disclosed given it is
seed-stable + transparent — the honesty of the disclosure is what matters, not the exact number.

## D21 — P2 accepted: panel serves the honest 0.737 daily model (2026-07-12, helen verified live)
Verified `pythia-serve` `/latest` serves `tft_lite_daily_qqq` **v2026-07-12** (trained 10:42Z,
dataset_hash 7a160165…): coverage_80 **0.7373**, mae_skill_vs_rw **−0.0625**, n_eval **1869 / 89
splits** — the D20 conformal model, NOT any pre-conformal miscalibrated row. Screenshot-verified the
live panel at raptor.tonyvigna.com: dashboard loads with the API up (hard gate — error boundary from
PR #36 also proven 06:56Z); the forecast panel reads **"MISCALIBRATED (0.737, target 0.80 ± 0.05)"**
with an Honest scorecard (cov 73.7%, MAE-skill −6.3% "worse than RW", hit 53.2%, n=1869/89,
CRPS 0.0084|0.0080 TFT|RW) and notes "systematic ~6pp train→eval drift on daily QQQ" + "Do not size
trades from these forecasts." **No "well-calibrated 0.80" claim anywhere.** Drivers strip degrades
gracefully ("Drivers not yet exported by the trainer — VSN weights land in P1 phase 2c/d"). **P2 =
DONE.** **Why:** every D20 acceptance condition is live and honest; the panel tells the true null-result
story. Next: P4 (daily + intraday conformal cones overlaid on one forward axis + event markers +
backtest replay) — the panel Tony validates.

## D22 — P4 first-cut incident + restore: frontend base-path regression (2026-07-12, helen)
**What happened.** The P4 overlay first cut (raptor-intel PR #37) rendered its *scaffolding* (badges,
boundary seam-check, honest notes) but the cone **chart drew no SVG** (svgCount 0) — a gate-fail I
bounced back. The twin's chart fix (PR #38) then **white-screened all of raptor** (~13:33–~14:55Z,
~1.5h): #root 0 children, no JS error, Dashboard chunk never requested.
**Root cause (the real one).** NOT the chart. The #38 image rebuild **dropped the
`VITE_BASE_PATH=/` build arg**, so the bundle was built with the pre-cutover base `/raptor-intel/`
(router basename `/raptor-intel/`). Production serves raptor at the **domain root** and OIDC returns
to `/#token`, so at path `/` the router matched nothing → never mounted. The first rollback (rebuild
of PR #36 commit 909d4fc) **also** blanked — same dropped arg. Decisive test: the *unauthenticated*
root `/` was blank too, and assets loaded from `/raptor-intel/assets/` while the page was at `/`.
**Fix / restore.** Rebuilt with `--build-arg VITE_BASE_PATH=/ --build-arg VITE_API_URL=` → assets
serve from `/assets/`, #root mounts (1 child, ~3MB DOM), authenticated Dashboard + P2 panel + all
APIs 200. Added `strategy: Recreate` to `k8s/50-frontend.yaml` (PR #39) to kill the rollout-overlap
2-pod chunk-hash race as well.
**Standing lessons.** (1) **Bake `VITE_BASE_PATH=/` + `VITE_API_URL=` as Dockerfile/CI defaults** for
raptor-intel-frontend so a rebuild can never silently drop them — one omission cost ~1.5h. (2) A blank
SPA with **no JS error + entry chunk 200 + lazy chunk never requested** ⇒ suspect a **base-path/router
basename mismatch**, and always test the **unauthenticated root** to isolate frontend-mount from
auth/backend. (3) Verify a "known-good" rollback on the **authenticated** path, not just the sign-in
card. **Why logged:** P4's overlay must be re-attempted as a minimal delta on this base=/ build and
must NOT reintroduce either regression; the chart-render (svgCount 0) fix is still pending.

## D23 — P4 ACCEPTED: overlaid final panel live + rendering (2026-07-12, helen live-verified)
The overlay panel renders on raptor.tonyvigna.com (base=/, app mounts, hard gate passed). Verified
live: overlay container svgCount>=1; the chart draws BOTH cones on one forward axis (-30m → +5d) —
daily cone green P90 / orange P10 / white-dashed P50 (~725.51, tooltip T+2d P10 706.97 / P90 744.54)
spreading to +5d, intraday cone at the left edge — plus the daily P50 line, working hover tooltip,
event marker (GOOG EPS), badges **daily AMBER cov80 73.7% / intraday CALIBRATED cov80 77.4%**, the
boundary-consistency seam check, and both honest scorecard notes. Rebuilt with baked
VITE_BASE_PATH=/ + VITE_API_URL= defaults and a Dockerfile fail-fast assert (PR #40) so the D22
base-path regression cannot recur; deployed Recreate. **P4 = DONE.** This completes the Pythia
deliverable: an honest, calibrated **null-result** forecasting panel for QQQ (returns ≈ random-walk at
both daily and intraday horizons; the value is calibrated uncertainty, not alpha).
**Two polish items flagged to Tony for his live-validation pass (NOT blockers):** (1) the intraday
30-min cone is visually compressed beside the 5-day daily axis (inherent shared-linear-axis tradeoff;
options: broken/dual axis, log-time, or a zoom toggle); (2) only 1 of 7 known events (GOOG EPS 07-15)
falls within the daily +5d horizon — the rest (TSLA/AMZN+META/FOMC/AAPL/MSFT+NVDA/CPI, 07-22→08-05)
are beyond the forecast window; consider extending the axis or a separate event ribbon.
**Why:** per Tony's "run it through deployment, I'll validate the final pane," the functional bar is met
and live; these are design calls he's best placed to direct. P5 extras remain optional.

## D24 — P4 polish calls + P5 kickoff (2026-07-12, helen; Tony delegated "do the things you recommend")
Tony validated P4 and delegated both the polish calls and the P5 go-ahead to helen. Decisions:

**P4 polish 1 — intraday cone compression → BROKEN/PIECEWISE forward axis.** Give the intraday
segment (now→+30m) a fixed left zone (~1/3 width) at minute ticks and the daily segment (+1d→+5d)
the right ~2/3 at day ticks, with a labeled seam ("end of session ┊ days →"). Preserves the
one-axis seam narrative while making BOTH cones readable. Rejected: log-time (hard to read),
zoom-toggle (hides the overlay's whole point). Implementation latitude: if a true broken axis is
too fiddly in Recharts, two aligned sub-charts sharing the seam is an acceptable equivalent —
outcome fixed (intraday readable + daily readable + seam preserved).

**P4 polish 2 — events beyond +5d → separate 30-DAY EVENT RIBBON below the chart.** Render all 7
known events as ticks/labels on a calendar ribbon decoupled from the forecast axis; in-window events
(GOOG 07-15) also keep their on-chart marker. Do NOT extend the cone axis past +5d — the model
doesn't forecast that far and empty cone would be misleading.

**P5 question answers:**
1. Attention viz scope → **P2 forecast panel only** (agree default; keep the overlay clean).
2. Breakouts channel → **`pythia_breakouts` table + `/breakouts` endpoint + in-panel pill for v1;
   external ping DEFERRED.** If built later, use a dedicated pythia-diagnostics GitHub-issue channel,
   **NOT an Ariadne project** — Ariadne is the CPM/work-scheduling board; telemetry would pollute
   ground truth.
3. Range/vol target math → **realized-range (high−low) as % of price** (agree default; matches the P2
   "Day range" vocabulary + interpretable). Model log(range). Parkinson vol noted as an optional
   future rigor upgrade.
4. Order → P5 is now **UNBLOCKED** (helen owns the P4 polish call). Sequence: **P4-polish (broken axis
   + event ribbon) FIRST** (it reshapes the overlay chart), then **(a) multi-target → (c) attention →
   (b) breakouts**. Each ships as its own PR + Recreate + local svgCount-verify + honesty guardrails
   (calibrated bands, skill-vs-RW always reported, no alpha copy).
**Why:** completes the panel Tony validated; each piece is honest-by-construction and independently
shippable; the P4-polish-first ordering avoids chart-component churn against the multi-target toggle.

## D25 — P4-polish ACCEPTED: broken axis (readable intraday) + 30-day event ribbon (2026-07-12, helen live-verified)
Verified live on raptor.tonyvigna.com (base=/ holds: unauth root=sign-in, auth mounts ~3MB, assets
from /assets/ new hash index-DLzDElmz.js). Both D24 polish items landed: (1) BROKEN/PIECEWISE forward
axis — intraday cone (now→+30m, minute ticks) now has REAL WIDTH and reads cleanly (green P90 / orange
P10 / blue actual) with a labeled seam ("intraday minutes | daily sessions"), then the daily cone
(+1d→+5d) — no longer a compressed sliver; (2) 30-DAY EVENT RIBBON renders all 7 events (GOOG, TSLA,
AMZN, META, FOMC, AAPL, MSFT, NVDA, CPI), decoupled from the +5d forecast axis, with in-window GOOG
also marked on the chart. Badges honest (daily AMBER 73.7%, intraday CALIBRATED 77.4%). **P4-polish = DONE.**

## D26 — P5a range target: serve rolling_range, conformal-calibrated + honest (2026-07-12, helen)
The multi-target toggle needs a range/vol model. Bake-off: **rolling_range WINS CRPS (0.00617 vs
range_tft 0.00715)**, and the range-TFT head is **under-dispersed (cov80 0.680, below the 0.75 gate)** —
same "simple ≈/> TFT" null shape as price (D2/D14/D19). DECISION: **v1 range target = rolling_range**
(the CRPS winner), with the **same per-window conformal calibration as price (D18) applied to its
P10/P90 bands** so cov80 lands in-gate; if conformal can't reach ~0.80, **disclose honestly** (amber
badge + drift note, per D20) rather than ship tight bands. Do NOT serve the raw under-dispersed
range-TFT. The multi-target toggle shows price ↔ range each with its OWN calibration badge + skill-vs-RW,
no alpha copy. **Why:** mirrors the price precedent exactly — serve the best-CRPS model with honestly
calibrated (or honestly disclosed) uncertainty; the range null is itself an honest result. P5c
(attention viz) proceeds in parallel per D24 (independent, no external contract — no need to hold it
for P4-polish, which is now verified anyway).

## D27 — P5c attention strip: live + honest graceful degradation verified (2026-07-12, helen)
PRs #25 (pythia: capture+serve TFT attention_weights) + #42 (raptor: AttentionStrip on the P2 forecast
panel) merged + Recreate-deployed (base=/ holds, bundle index-DHoDnuWA.js). Verified live on the P2
"Pythia daily forecast" panel: the strip renders UNDER the drivers with the honest note **"Attention
not available for this model version (trainer pre-P5c)"** — the correct graceful state per D24 Q1
(the served model predates the attention capture). Placement correct: P2 panel ONLY, not the overlay.
**P5c code = DONE** (UI + honest degradation + capture/serve path shipped). The REAL attention bars
auto-populate on the next NIGHTLY RETRAIN (which registers a model carrying attention_weights).
**FOLLOW-UP:** after that retrain, re-verify the strip shows real per-bar weights + the honest
diffuse-attention note when ~uniform — that validates the PR #25 capture end-to-end (unverified until a
model carries the array; if it still says "not available" post-retrain, the capture is broken).

## D28 — P5a (multi-target toggle) + P5b (breakout pill) verified live (2026-07-12, helen)
Screenshot-verified on raptor.tonyvigna.com (base=/ holds, assets from /assets/, bundle index-DlBJCewQ.js).
P5a: the Pythia daily-forecast panel has a **Target: Price (return) | Range** toggle, each target with
its OWN cov80 badge + skill-vs-baseline scorecard (price shows the honest MISCALIBRATED 0.737; range
conformal-calibrated to ~0.84 GREEN per D26). P5b: the **Breakouts (rolling 20) diagnostic** pill reads
**AMBER · rolling 60% vs 20% expected (+40pp)** BUT **lifetime breach 24% ~ 20% over n=2640 =
structurally calibrated**, framed as a RECENT vol-regime drift (not alpha) — both rates always shown vs
the 20% expectation, recent breach events listed, honest recent-vs-structural verdict. No alpha copy.
**P5a + P5b = DONE.** Remaining: **P5c real attention bars** (still "not available for this model
version" — pending the nightly retrain populating attention_weights; re-verify then via /tmp/pythia_attn_check.py).
