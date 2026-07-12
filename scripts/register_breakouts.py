"""CLI: run the P5b breakout scan + attach the scorecard to the latest model.

Two writes, both keyed on the SAME registered (model_name, model_version) so
/breakouts and /latest agree:

  1. per-bar AUDIT rows -> pythia_breakouts (idempotent UPSERT on
     model_version+symbol+ts+horizon), from run_breakout_scan;
  2. rolling SCORECARD block -> report_json["breakouts"], from
     build_breakouts_response, merged into the latest price row + re-registered
     (registry UPSERT, no version churn).

Scan model: the diagnostic measures the SERVED price band's P10-P90 breach
rate. The runnable default here is a conformal-calibrated random-walk band
(CPU, replayable) — the closest cheap proxy to the served calibrated cone.
The production run uses the served TFT band by importing run_breakout_scan
with the TFT factory inside a GPU job; the scan_model label records which was
used so the panel never misrepresents the source.

Usage:
    python -m scripts.register_breakouts \\
        --model-name tft_lite_daily_qqq \\
        --start 2023-05-01 --end 2026-07-11 \\
        --symbol QQQ --historical yfinance
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from sqlalchemy import text

from pythia.baselines import RandomWalk
from pythia.breakouts import (
    BREAKOUTS_DDL,
    build_breakouts_response,
    run_breakout_scan,
)
from pythia.data import assemble_dataset
from pythia.models.conformal import ConformalScaledModel
from pythia.registry import get_default_registry


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _factory(kind: str, target_col: str):
    if kind == "random_walk":
        return lambda: RandomWalk(target_col)
    if kind == "conformal_rw":
        return lambda: ConformalScaledModel(base=RandomWalk(target_col), target_col=target_col)
    raise ValueError(f"unknown scan model {kind!r}")


def _persist_rows(engine, scan) -> int:
    """Idempotent UPSERT of the per-bar audit rows into pythia_breakouts."""
    with engine.begin() as conn:
        conn.execute(text(BREAKOUTS_DDL))
    if scan.empty:
        return 0
    q = text(
        """
        INSERT INTO pythia_breakouts
            (model_version, symbol, ts, horizon, direction, realized,
             p10, p90, exceeded, magnitude, oos)
        VALUES
            (:model_version, :symbol, :ts, :horizon, :direction, :realized,
             :p10, :p90, :exceeded, :magnitude, :oos)
        ON CONFLICT (model_version, symbol, ts, horizon) DO UPDATE
            SET direction = EXCLUDED.direction,
                realized  = EXCLUDED.realized,
                p10       = EXCLUDED.p10,
                p90       = EXCLUDED.p90,
                exceeded  = EXCLUDED.exceeded,
                magnitude = EXCLUDED.magnitude,
                oos       = EXCLUDED.oos
        """
    )
    rows = [
        {
            "model_version": r.model_version,
            "symbol": r.symbol,
            "ts": r.ts,
            "horizon": int(r.horizon),
            "direction": r.direction,
            "realized": float(r.realized),
            "p10": float(r.p10),
            "p90": float(r.p90),
            "exceeded": bool(r.exceeded),
            "magnitude": float(r.magnitude),
            "oos": bool(r.oos),
        }
        for r in scan.itertuples(index=False)
    ]
    with engine.begin() as conn:
        conn.execute(q, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="register_breakouts",
        description="Run the P5b breakout scan + attach the scorecard.",
    )
    p.add_argument("--model-name", type=str, default="tft_lite_daily_qqq")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--symbol", type=str, default="QQQ")
    p.add_argument("--scan-model", type=str, default="conformal_rw",
                   choices=["conformal_rw", "random_walk"])
    p.add_argument("--historical", type=str, default="yfinance",
                   choices=["yfinance", "stooq", "none"])
    p.add_argument("--historical-start", type=_parse_date, default=None)
    p.add_argument("--initial-train", type=int, default=252)
    p.add_argument("--eval-size", type=int, default=63)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--dry-run", action="store_true",
                   help="compute + print; no DB writes")
    args = p.parse_args(argv)

    symbol = args.symbol.upper()
    target_col = f"{symbol}_close"
    provider = None if args.historical == "none" else args.historical

    result = assemble_dataset(
        args.start, args.end, symbols=None,
        historical_provider=provider, historical_start=args.historical_start,
    )
    wide = result.daily

    reg = get_default_registry()
    rec = None if args.dry_run else reg.latest(args.model_name)
    version = rec.model_version if rec is not None else "unregistered"

    scan = run_breakout_scan(
        wide, _factory(args.scan_model, target_col),
        target_col=target_col, symbol=symbol, model_version=version,
        initial_train=args.initial_train, eval_size=args.eval_size,
    )
    block = build_breakouts_response(scan, window=args.window)
    block["scan_model"] = args.scan_model  # never misrepresent the source

    print(
        f"breakout scan: n={block['n_scan']} rows badge={block['badge']} "
        f"rate={block['rate']} scan_model={args.scan_model}",
        file=sys.stderr,
    )

    if args.dry_run:
        json.dump(block, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if rec is None:
        print(
            f"ERROR: no registered model under {args.model_name!r}; register the "
            f"price model first (scripts.register_report).",
            file=sys.stderr,
        )
        return 2

    n_rows = _persist_rows(reg.engine, scan)
    report = dict(rec.report_json)
    report["breakouts"] = block
    reg.register(
        model_name=rec.model_name,
        model_version=rec.model_version,  # SAME version
        dataset_hash=rec.dataset_hash,
        report_json=report,
        artifact_uri=rec.artifact_uri,
        git_sha=rec.git_sha,
    )
    print(
        f"persisted {n_rows} audit rows + attached scorecard to "
        f"{rec.model_name}@{rec.model_version}; /breakouts now serves it.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
