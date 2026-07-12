# P5 design-first — extras (multi-target toggle, surprise alerts, attention viz)

**Status:** twin proposal 2026-07-12, awaiting helen review. Design-only. No
code shipped.

**Task:** Ariadne id `6a25a1a7-bf56-452a-9d4f-f9178ef6cd80`. Depends P4
(shipped + accepted D23). Same honesty bar as D2/D9/D19/D20/D21: calibrated
bands, no alpha claims, honest scorecard framing.

**Ship discipline** (helen 2026-07-12 P0 postmortem): base=/ + Recreate +
local svgCount-verify BEFORE any deploy. Dockerfile assert already prevents
base-path regression.

## The three extras

P5 bundles three loosely-coupled additions to the overlay panel. Each is
independently shippable behind its own toggle. Ordering below is
lowest-risk / highest-signal first.

### (a) Multi-target toggle — price vs realized range/vol

**What.** A control on the overlay panel that swaps the target the cone
predicts: PRICE (current, the same 5-day/30-min forward path) ↔
RANGE/VOL (the model's realized range over the same horizon).

**Why this first.** Per D2, "vol is where the model actually shines" —
the daily QQQ TFT-lite currently ships NULL SKILL on price
(mae_skill_vs_rw ~0). If we honestly serve a second target where the
model beats random-walk, we tell a more complete story without moving the
honesty bar. It's also the smallest UI delta: a single toggle + the
existing cone renderer.

**Model artifacts.** We already store two targets in the report
(`tft_lite_daily_qqq` has price; a range/vol target needs
`tft_lite_daily_qqq_range` registered — trained by `nightly_retrain.py`
with `target=realized_range`).

**Contract.**
- New model row: `tft_lite_daily_qqq_range`, versioned, tracked in
  `pythia.pythia_models`.
- `/latest?model=tft_lite_daily_qqq_range` returns the same shape as the
  price model. Panel picks price OR range from the toggle.
- Registry rule: a range/vol target refuses to register if
  `coverage_80` falls out of the honesty gate (0.75–0.85). Skill vs RW
  MUST be reported — no hiding the null-skill case.

**UI.** One radio-pair "Target: Price · Range" above the calibration
badges. Both targets carry their OWN badges (independent calibration).
Notes rebuild from the served row (already dynamic per D17).

**Verification (before deploy).**
- Retrain `tft_lite_daily_qqq_range` walk-forward, register only if
  gated.
- Playwright: for each toggle position, headings + `svgCount >= 1` +
  badges reflect the CORRECT target's cov80 and skill.
- helen live: click through both toggles, screenshot-verify.

**Honesty guardrail.** If range/vol also lands NULL SKILL after retrain,
we ship the toggle anyway — the extra target's honest scorecard is
already a win over the current single-target view. No copy suggests
"try range for edge."

---

### (b) Surprise / breakout alerts — live path exits the cone

**What.** During the intraday window we ALREADY forecast (T+10..+30m),
watch the live QQQ price. When it exits the P10/P90 band, log a
"breakout" event. Surface it as an inline pill on the overlay ("BREAKOUT
14:32Z: price 728.10 vs P90 727.85") and optionally as a webhook / bus
ping.

**Why.** This is the natural feedback loop D9 hinted at ("do not size
trades…"). We're not saying "buy on breakout" — we're saying "the model
said this shouldn't happen; here's the frequency it does." Over time
this IS the calibration story: if breakouts fire at ~20% (P10-P90 =
80% band), the model is calibrated in the tails. If breakouts fire at
40%, the cone is too tight.

**Data path.**
- Read tap: the streaming ingestion Tony already runs for intraday
  quotes.
- Store: new table `pythia.pythia_breakouts` — (ts, symbol, model,
  model_version, price, p10, p90, exit_side, cone_age_min).
- Compute: a small serve endpoint `/breakouts?symbol=&date=` returns
  today's breakouts + a rolling "breakout rate vs 20% expected". The
  panel renders this as a scorecard-adjacent stat.

**UI.**
- On the overlay: a subtle pill list under the boundary check ("2
  breakouts today · rolling 30-day rate 22% vs 20% expected — calibrated
  in the tails"). Same amber/emerald semantics as the coverage badges.
- No modal, no interruption; alerts feed the calibration story, not a
  trading UI.

**Honesty guardrail.**
- NEVER show breakouts without the "vs 20% expected" comparison — a
  breakout is not a signal, it's a calibration data-point.
- The rolling rate is the actual value, not a smoothed one. If we don't
  have enough live minutes yet, we render "n=X ticks — insufficient" and
  no rate.
- The bus/webhook ping is OPT-IN and lands in a `#pythia-diagnostics`
  bucket, not `#trades`.

**Verification.**
- Backfill: replay yesterday's intraday quotes through the intraday cone
  and populate `pythia_breakouts`; compare the observed rate to the 20%
  expectation.
- Live: watch a real 30-min session; confirm the pill updates as
  breakouts fire.

---

### (c) Attention-over-time viz — which past bars the model weighted

**What.** For each daily forecast we render, show a small strip chart
of the TFT decoder's attention weights over the encoder window
(last 60 trading days). Tallest bar = most-influential past day. Same
strip for the intraday model (last 30 minutes of encoder).

**Why.** Transparency > alpha. Users see "the model leaned hard on last
Tuesday's price move" and can eyeball whether that makes sense given the
news that day. It also surfaces obvious pathologies (all weight on the
most recent bar = model is basically a random-walk).

**Data path.**
- Trainer emits `attention_weights: number[]` (length = encoder_length)
  per served forecast. Model registration extends the report JSON with
  a new optional `attention` block.
- `/latest` returns the attention array in the report.

**UI.**
- Small horizontal bar strip under the existing driver-importance bar
  in `PythiaForecastPanel` (P2 territory). Overlay panel does NOT show
  it — keeps overlay uncluttered.
- Hover a bar → tooltip with the date + close on that day.

**Honesty guardrail.**
- No "the model is confident because it looked at day X" copy.
- If attention is uniform (~1/n), we say "attention is diffuse — no
  single past bar dominates."
- The strip renders only when the trainer emits the array; older
  registered models render "attention not available for this model
  version" (already dynamic-notes precedent from D17).

**Verification.**
- Trainer round-trip: retrain, register with attention block, `/latest`
  returns it.
- Playwright: strip mounts under the drivers, `svgCount` on the
  forecast-panel container increments by 1.

---

## Rollout order & risk

Recommended order:
1. **(a) Multi-target toggle** — least new surface area; leverages
   existing registry + serve + panel machinery. 1 model-row + 1 UI
   toggle + Playwright verify.
2. **(c) Attention viz** — trainer + serve surface change but the UI
   is purely additive on the P2 panel. Independent of (a) and (b).
3. **(b) Breakout alerts** — new table + endpoint + optional webhook +
   streaming tap. Highest surface area; ship last so the P4 panel gets
   the (a) and (c) upgrades regardless.

Each ships as its own PR + Recreate deploy + local svg-verify.

## Open questions for helen

1. **Attention viz scope** — under P2 forecast panel only, or ALSO on
   the overlay panel? Twin default: P2 only, keeps overlay clean.
2. **Breakouts bus channel** — Ariadne `pythia-diagnostics` project or a
   dedicated GitHub-Issue-based channel? Twin default: Ariadne
   project (mirrors twin-bus pattern).
3. **Range/vol target math** — realized-range = high-low over the
   horizon? Or Parkinson vol? Twin default: realized-range (matches P2
   framing).
4. **Order** — start with (a) or defer entirely until Tony has signed
   off on the P4 polish items? Twin default: wait for Tony's P4
   feedback before starting any P5 code.
