# P3 — Intraday TFT: foundation + design note (review gate)

**Task:** `96139db3` (Pythia P3). **Author:** agent-2 (achilles).
**Decision owner (methodology):** helen. Depends on P1 (done).

Per the P1 protocol ("ping helen to review the backtest design before the
model"), this note asks for sign-off on the intraday **foundation** before the
intraday TFT is trained. The foundation (intraday assembler + hermetic tests)
is shipped in this PR; the model is the next phase.

## What forecasting problem

Next **`bar_minutes`** move of QQQ (default 30 min; the task says "next 30–60
min"). Same rigor as the daily side: walk-forward, out-of-sample, calibrated,
scored vs baselines — direction likely a coin flip (allowed, reported), the bar
is **calibrated + beat baselines on CRPS/pinball, or report it doesn't**.

## Foundation shipped here

`pythia.data.intraday` rolls raptor's tick feed (`staging.quote_raw`, same
source query as the daily assembler) into fixed `bar_minutes` OHLCV bars,
timestamp-indexed, wide (`{symbol}_close` / `{symbol}_volume`) — the **same
schema** as the daily frame. So the existing horizon-agnostic
`expanding_walk_forward` / `run_backtest` / baselines / metrics all apply
**unchanged**; only the index granularity changes from day to bar.

Known-future intraday calendar features on the bar timestamp: `minute_of_day`,
`minutes_to_close`, `dow`. Covariate-lag gate is identical (features at bar t
from ≤ t-1; target at bar t = realised [t-1, t] move).

`scripts/assemble_intraday.py` runs the live pull on achilles. 6 hermetic tests
(fixture ticks, no DB) cover roll-up, session filtering, the session-open
marker, the wide schema, and the mask.

## Design choices for helen (parameters, not hardcoded)

1. **Bar width / horizon** — default 30 min. 60 min if you prefer a thicker,
   less-microstructure-noisy bar. Trivially reconfigurable (`--bar-minutes`).
2. **Session filter** — default regular session only (09:30–16:00 by raptor's
   clock). Extended-hours ticks are thin/noisy; `--include-extended-hours` to
   keep them.
3. **Overnight gap (the important one).** The bar-to-bar return at the FIRST
   bar of each session spans the overnight gap — a different, much
   higher-variance problem than an intraday move. The assembler marks those
   rows (`is_session_open`) and exposes `overnight_mask()`. **I recommend
   scoring intraday models on within-session moves only** (mask the
   session-open target rows). Two ways to wire it, and since `run_backtest` is
   shared code (twin's P1), I want your + twin's call before touching it:
   - **(a)** add an optional `eval_mask: pd.Series | None` param to
     `run_backtest` (additive, default None = today's behaviour), or
   - **(b)** pre-filter: drop session-open rows from the target only, outside
     the harness.
   I lean (a) — explicit, testable, reusable — but it edits the shared harness,
   so flagging rather than steamrolling.
4. **`p_move` baseline.** Still the documented stub (raises, never fabricates).
   Wiring the real raptor `p_move` history adapter is worthwhile for the
   intraday scorecard — is there a persisted `p_move` snapshot yet (the P1
   follow-up), or should P3 carry that adapter?

## Data reality

raptor has only carried the board intraday since ~2026-06-05, so the intraday
walk-forward starts thin (like the daily side pre-D8). Unlike daily, there is
**no free intraday backfill** at bar granularity for 20 symbols — so the honest
expectation is a smaller-n intraday verdict that firms up as raptor accrues
history. I'll report n explicitly and flag if it's too thin for a defensible
verdict.

## Plan

1. **This PR** — intraday assembler foundation + CLI + tests (design review).
2. After your sign-off on 1–4 above — wire the intraday walk-forward
   (existing harness + overnight mask + baselines), report the baseline
   calibration/skill on real intraday bars.
3. Intraday TFT-lite on the 2080 Ti; honest calibration + skill verdict vs
   baselines; served via the P1 inference API pattern. Ping you at each gate.
