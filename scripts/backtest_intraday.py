"""CLI: P3 intraday walk-forward backtest (live).

Assembles 10-min bars from raptor's tick feed, loads raptor's p_move +
direction history, runs the forward-horizon walk-forward against the baselines,
and writes report.json. Runs on achilles where the raptor Postgres is reachable
(quote_raw + staging.qqq_pmove + staging.qqq_direction all live in the same DB).

Usage:
    PYTHIA_DB_DSN=postgresql://appuser:***@postgres-0.raptor-intel:5432/appdb \
      python -m scripts.backtest_intraday \
        --start 2026-06-05 --end 2026-07-11 \
        --bar-minutes 10 --horizon 3 \
        --initial-train 200 --eval-size 39 \
        --report data/intraday_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pythia.backtest.intraday import run_intraday_backtest
from pythia.data.intraday import assemble_intraday_dataset
from pythia.data.pmove_history import load_direction_tilt, load_pmove_series


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="backtest_intraday",
        description="P3 intraday walk-forward vs baselines incl. raptor p_move.",
    )
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--out", "--report", dest="report", type=Path, required=True)
    p.add_argument("--bar-minutes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=3, help="bars ahead (3 x 10min = 30min)")
    p.add_argument("--initial-train", type=int, default=200)
    p.add_argument("--eval-size", type=int, default=39, help="~1 session of 10-min bars")
    p.add_argument("--target", type=str, default="QQQ_close")
    # Intraday TFT-lite (the REPORTED verdict must be a GPU pass; CPU = smoke).
    p.add_argument(
        "--with-tft", action="store_true", help="also train + score the intraday TFT-lite"
    )
    p.add_argument("--encoder-length", type=int, default=39)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--tft-batch-size", type=int, default=128)
    args = p.parse_args(argv)

    intr = assemble_intraday_dataset(args.start, args.end, bar_minutes=args.bar_minutes)
    p_move = load_pmove_series()
    tilt = load_direction_tilt()

    tft_kwargs = dict(
        encoder_length=args.encoder_length,
        hidden_size=args.hidden_size,
        max_epochs=args.max_epochs,
        batch_size=args.tft_batch_size,
    )
    reports = run_intraday_backtest(
        intr.bars,
        price_col=args.target,
        p_move=p_move,
        tilt=tilt,
        horizon=args.horizon,
        initial_train=args.initial_train,
        eval_size=args.eval_size,
        with_tft=args.with_tft,
        tft_kwargs=tft_kwargs,
    )

    out = {name: r.as_dict() for name, r in reports.items()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, sort_keys=True))

    manifest = {
        "bars": int(intr.bars.shape[0]),
        "bar_minutes": args.bar_minutes,
        "horizon": args.horizon,
        "symbols_included": list(intr.symbols_included),
        "p_move_rows": int(p_move.shape[0]),
        "direction_rows": int(tilt.shape[0]),
        "models": {
            name: {
                "n_eval_obs": r.n_eval_obs,
                "n_splits": r.n_splits,
                "coverage_80": r.coverage_80,
                "crps": r.crps,
                "mae_skill_vs_rw": r.mae_skill_vs_rw,
                "warnings": r.warnings,
            }
            for name, r in reports.items()
        },
        "report": str(args.report),
    }
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
