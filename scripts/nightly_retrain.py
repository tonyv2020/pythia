"""End-to-end nightly retrain — assemble → train → register.

Wired into a k8s CronJob (see k8s/nightly-retrain-cronjob.yaml). One
execution:
    1. Assembles a **D8-backfilled** dataset — historical daily bars from
       yfinance (2018-01-01 onward by default) + the raptor feed for recent
       bars. This is the fat sample the P1-verdict + registry should run on;
       n=1869/89 splits vs the thin default's n=214/22 (D10, helen 2026-07-11).
    2. Runs the walk-forward TFT-lite backtest against the resulting frame.
    3. Registers the report in ``pythia_models`` with today's UTC datestamp
       as the model_version.

The default flags OPT INTO the backfill so ``kubectl apply`` of the
CronJob produces the fat verdict without extra config. Overrides via CLI
if a night's run needs the thin-only path for diagnostics.

Exits non-zero on any step failure — k8s handles alerting via the CronJob
failure condition; there's no ambient state on success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


# Full 20-symbol macro board — everything the raptor-intel Macro % Change
# panel renders. yfinance covers all of these with multi-year history, so
# the backfilled dataset is the FULL board, not the long-history subset the
# thin-only path is limited to.
FULL_BOARD = (
    "QQQ,SPY,DIA,IWM,AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,GLD,SLV,GDX,USO,UGA,UNG,DBE,CORN,WEAT"
)


def _run(cmd: list[str]) -> None:
    sys.stdout.write(f"$ {' '.join(cmd)}\n")
    sys.stdout.flush()
    subprocess.check_call(cmd)


def _run_soft(cmd: list[str], label: str) -> bool:
    """Run a non-critical step; WARN + continue on failure instead of aborting.

    Used for the P5 diagnostic blocks (range / breakouts): they enrich the
    freshly-registered row but must NOT fail the whole nightly if e.g. yfinance
    hiccups — the core model already registered and serves. The panel degrades
    gracefully to a null block and the next nightly retries. Returns True on
    success so the caller can log a summary.
    """
    sys.stdout.write(f"$ {' '.join(cmd)}\n")
    sys.stdout.flush()
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            f"WARNING: {label} failed (exit {e.returncode}); row stays without "
            f"that block, panel shows the null-safe state, next nightly retries.\n"
        )
        sys.stderr.flush()
        return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nightly_retrain")
    p.add_argument("--data-dir", type=Path, default=Path("/data"))
    p.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="earliest date for the assembled dataset. Default 2018-01-01 covers "
        "the D8 backfill window; the raptor feed still supplies recent bars.",
    )
    p.add_argument("--model-name", type=str, default="tft_lite_daily_qqq")
    p.add_argument("--artifact-uri", type=str, default="pvc://pythia-data/report.json")
    p.add_argument("--symbols", type=str, default=FULL_BOARD)
    p.add_argument(
        "--historical",
        type=str,
        default="yfinance",
        help="D8 backfill provider (default: yfinance). Pass empty string "
        "to disable the backfill and use raptor-only bars.",
    )
    p.add_argument(
        "--no-historical-adjust",
        action="store_true",
        help="Use raw (unadjusted) historical prices. Default is split/div-"
        "adjusted; unadjusted would inject fake ~90%% split-day returns "
        "into a multi-year sample (helen D10).",
    )
    p.add_argument(
        "--initial-train",
        type=int,
        default=252,
        help="~1 trading yr — right sized for the D8-fattened dataset "
        "(n=1869); the thin-only default was 150.",
    )
    p.add_argument("--eval-size", type=int, default=21)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument(
        "--skip-p5-blocks",
        action="store_true",
        help="skip attaching the P5a range + P5b breakout blocks to the new "
        "row (diagnostic runs). Default: attach them so /latest range + "
        "/breakouts survive the retrain (helen D28).",
    )
    args = p.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.data_dir / "board.parquet"
    report = args.data_dir / "report.json"
    today = datetime.now(UTC).date().isoformat()
    end = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()

    assemble_cmd = [
        sys.executable,
        "-m",
        "scripts.assemble_dataset",
        "--start",
        args.start,
        "--end",
        end,
        "--symbols",
        args.symbols,
        "--out",
        str(dataset),
    ]
    if args.historical:
        assemble_cmd += ["--historical", args.historical, "--historical-start", args.start]
        if args.no_historical_adjust:
            assemble_cmd += ["--no-adjust"]

    _run(assemble_cmd)

    _run(
        [
            sys.executable,
            "-m",
            "scripts.train_p1_tft",
            "--dataset",
            str(dataset),
            "--target",
            "QQQ_close",
            "--initial-train",
            str(args.initial_train),
            "--eval-size",
            str(args.eval_size),
            "--max-epochs",
            str(args.max_epochs),
            # D14 cov80 0.781 came from encoder=60 hidden=16; my earlier 40/32
            # miscalibrated to cov80 0.586. Revert to agent-2s calibrated config.
            "--encoder-length",
            "60",
            "--hidden-size",
            "16",
            "--batch-size",
            "32",
            "--report",
            str(report),
        ]
    )

    _run(
        [
            sys.executable,
            "-m",
            "scripts.register_report",
            "--model-name",
            args.model_name,
            "--model-version",
            f"v{today}",
            "--dataset",
            str(dataset),
            "--report",
            str(report),
            "--artifact-uri",
            args.artifact_uri,
        ]
    )

    # P5 multi-target blocks (helen D28: P5a range + P5b breakouts ACCEPTED).
    # register_report writes ONLY the price report (+ the trainer's attention)
    # onto the new vYYYY-MM-DD row, so without this step every nightly ORPHANS
    # the range + breakouts blocks and /latest range + /breakouts go null until
    # someone re-populates by hand. Attach them here, onto the row we just
    # registered (the populate scripts UPSERT the SAME latest row, same
    # model_version). They re-assemble with high/low (hl_symbols) since the
    # price board.parquet has none — a self-contained ~min of CPU each. Soft:
    # a yfinance blip must not fail the whole retrain; the panel degrades to the
    # null-safe block and the next nightly retries.
    if not args.skip_p5_blocks:
        hist = args.historical if args.historical else "none"
        block_args = [
            "--model-name",
            args.model_name,
            "--start",
            args.start,
            "--end",
            end,
            "--symbol",
            "QQQ",
            "--historical",
            hist,
        ]
        ok_range = _run_soft(
            [sys.executable, "-m", "scripts.register_range_block", *block_args],
            "register_range_block (P5a)",
        )
        ok_breakouts = _run_soft(
            [sys.executable, "-m", "scripts.register_breakouts", *block_args],
            "register_breakouts (P5b)",
        )
        sys.stdout.write(
            f"P5 blocks: range={'ok' if ok_range else 'FAILED'} "
            f"breakouts={'ok' if ok_breakouts else 'FAILED'}\n"
        )

    sys.stdout.write(f"nightly retrain complete: {today}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
