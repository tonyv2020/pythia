"""End-to-end nightly retrain — assemble → train → register.

Wired into a k8s CronJob (see k8s/nightly-retrain-cronjob.yaml). One
execution:
    1. Assembles the P0 dataset from raptor Postgres for the *long-history*
       symbol subset (D8 backfill will grow this later).
    2. Runs the walk-forward TFT-lite backtest.
    3. Registers the report in ``pythia_models`` with today's UTC datestamp
       as the model_version.

Exits non-zero on any step failure — k8s handles alerting via the CronJob
failure condition; there's no ambient state on success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


LONG_HISTORY_SUBSET = (
    "QQQ,AAPL,AMZN,GOOG,CORN,DBE,GLD,"
    "BYND,CANE,DBB,DIS,DOCU,FXB,FXE,FXY"
)


def _run(cmd: list[str]) -> None:
    sys.stdout.write(f"$ {' '.join(cmd)}\n"); sys.stdout.flush()
    subprocess.check_call(cmd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nightly_retrain")
    p.add_argument("--data-dir", type=Path, default=Path("/data"))
    p.add_argument("--start", type=str, default="2023-05-19")
    p.add_argument("--model-name", type=str, default="tft_lite_daily_qqq")
    p.add_argument("--artifact-uri", type=str, default="pvc://pythia-data/report.json")
    p.add_argument("--symbols", type=str, default=LONG_HISTORY_SUBSET)
    p.add_argument("--initial-train", type=int, default=150)
    p.add_argument("--eval-size", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    args = p.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / "board.parquet"
    report = args.data_dir / "report.json"
    today = datetime.now(UTC).date().isoformat()
    # End = yesterday (today's bars may not be finalised yet).
    end = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    _run([
        sys.executable, "-m", "scripts.assemble_dataset",
        "--start", args.start,
        "--end", end,
        "--symbols", args.symbols,
        "--out", str(dataset),
    ])

    _run([
        sys.executable, "-m", "scripts.train_p1_tft",
        "--dataset", str(dataset),
        "--target", "QQQ_close",
        "--initial-train", str(args.initial_train),
        "--eval-size", str(args.eval_size),
        "--max-epochs", str(args.max_epochs),
        "--encoder-length", "40",
        "--hidden-size", "32",
        "--batch-size", "32",
        "--report", str(report),
    ])

    _run([
        sys.executable, "-m", "scripts.register_report",
        "--model-name", args.model_name,
        "--model-version", f"v{today}",
        "--dataset", str(dataset),
        "--report", str(report),
        "--artifact-uri", args.artifact_uri,
    ])

    sys.stdout.write(f"nightly retrain complete: {today}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
