"""P1 phase-2 walk-forward training pass.

Assembles the P0 dataset (or reads a pre-assembled Parquet), runs walk-forward
splits, trains TFTLite on each split, scores against RandomWalk + LastReturn
baselines, and writes ``report.json`` with per-model calibration + skill.

Honesty rails: this script is a thin wrapper around ``run_backtest`` — the
same harness the baselines use. Any change here that would relax the
covariate-lag gate has to touch the shared feature builder + tests first.

Usage:
    python scripts/train_p1_tft.py \\
        --dataset data/board.parquet \\
        --target QQQ_close \\
        --initial-train 252 --eval-size 21 \\
        --max-epochs 8 --encoder-length 60 \\
        --report report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pythia.backtest import expanding_walk_forward, run_backtest
from pythia.baselines import LastReturn, RandomWalk
from pythia.models import TFTLiteModel


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_p1_tft")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--target", type=str, default="QQQ_close")
    p.add_argument("--initial-train", type=int, default=252)
    p.add_argument("--eval-size", type=int, default=21)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=8)
    p.add_argument("--encoder-length", type=int, default=60)
    p.add_argument("--hidden-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--max-splits", type=int, default=None,
                   help="cap the number of walk-forward splits for a smoke run")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if args.target not in df.columns:
        raise SystemExit(f"target {args.target!r} not in dataset")

    splits = list(expanding_walk_forward(
        df.index,
        initial_train_size=args.initial_train,
        eval_size=args.eval_size,
        step=args.step,
    ))
    if args.max_splits is not None:
        splits = splits[: args.max_splits]
    if not splits:
        raise SystemExit("no splits produced — dataset too short")

    sys.stdout.write(f"walk-forward: {len(splits)} splits\n"); sys.stdout.flush()

    reports = run_backtest(
        df,
        target_col=args.target,
        splits=splits,
        model_factories={
            "random_walk": lambda: RandomWalk(args.target),
            "last_return": lambda: LastReturn(args.target),
            "tft_lite":    lambda: TFTLiteModel(
                target_col=args.target,
                encoder_length=args.encoder_length,
                hidden_size=args.hidden_size,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
            ),
        },
    )

    out = {name: r.as_dict() for name, r in reports.items()}
    # Strip per_split raw arrays from disk report — keep aggregates.
    for r in out.values():
        r.pop("per_split", None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for name, r in reports.items():
        sys.stdout.write(
            f"[{name}] splits={r.n_splits} n={r.n_eval_obs} "
            f"MAE={r.mae:.6g} hit={r.hit_rate:.3f} cov80={r.coverage_80:.3f} "
            f"CRPS={r.crps:.6g} pinball50={r.pinball_50:.6g} "
            f"skill_vs_rw={r.mae_skill_vs_rw}\n"
        )
        for w in r.warnings:
            sys.stdout.write(f"  ! {w}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
