"""CLI: run the walk-forward harness on baselines and emit a JSON report.

Usage:
    python -m scripts.score_baselines \
        --dataset data/board_2024_onwards.parquet \
        --target  QQQ_close \
        --initial-train 252 \
        --eval-size 21 \
        --report  report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.baselines import LastReturn, RandomWalk


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="score_baselines",
        description="Run walk-forward + baselines + emit JSON report.",
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--target", type=str, default="QQQ_close",
                   help="target column (default: QQQ_close)")
    p.add_argument("--initial-train", type=int, default=252,
                   help="initial expanding-window size in rows (default: 252 = ~1 trading yr)")
    p.add_argument("--eval-size", type=int, default=21,
                   help="eval window size in rows (default: 21 = ~1 trading mo)")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if args.target not in df.columns:
        raise SystemExit(
            f"target column {args.target!r} not in dataset columns: "
            f"{list(df.columns)[:10]}... (see data-schema.md for naming)"
        )

    splits = list(expanding_walk_forward(
        df.index,
        initial_train_size=args.initial_train,
        eval_size=args.eval_size,
    ))
    if not splits:
        raise SystemExit(
            f"no splits produced — dataset has {len(df)} rows, need "
            f">= {args.initial_train + args.eval_size} for at least one split"
        )

    reports = run_backtest(
        df,
        target_col=args.target,
        splits=splits,
        model_factories={
            "random_walk": lambda: RandomWalk(target_col=args.target),
            "last_return": lambda: LastReturn(target_col=args.target),
        },
    )

    out = {name: r.as_dict() for name, r in reports.items()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    # Also human-readable summary to stdout.
    for name, r in reports.items():
        sys.stdout.write(
            f"[{name}] splits={r.n_splits} n={r.n_eval_obs} "
            f"MAE={r.mae:.6g} hit={r.hit_rate:.3f} cov80={r.coverage_80:.3f} "
            f"CRPS={r.crps:.6g} pinball50={r.pinball_50:.6g}\n"
        )
        for w in r.warnings:
            sys.stdout.write(f"  ! {w}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
