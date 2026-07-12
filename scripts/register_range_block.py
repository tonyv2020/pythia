"""CLI: attach the P5a realized-range block to the latest registered model.

The price model row (``tft_lite_daily_qqq``) is registered by
``scripts.register_report`` from the nightly TFT run. This step ADDS the
``range`` block to that SAME row's ``report_json`` so /latest can serve
``price`` and ``range`` keyed on ONE model_version (twin's serve contract,
helen D25). It:

  1. assembles the daily board WITH high/low for --symbol (hl_symbols),
     backfilling history from yfinance so the walk-forward verdict is honest;
  2. computes the conformal-rolling-range block (compute_range_block);
  3. reads the latest registry row for --model-name, merges
     report_json["range"] = block, and RE-registers under the same
     (model_name, model_version) — the registry UPSERT updates report_json
     in place, so no new row and no version churn.

Usage:
    python -m scripts.register_range_block \\
        --model-name tft_lite_daily_qqq \\
        --start 2023-05-01 --end 2026-07-11 \\
        --symbol QQQ --historical yfinance

Idempotent + replayable (D3): re-running with the same data reproduces the
same block. Prints the served coverage/badge to stdout (NO DB DSN echoed).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from pythia.data import assemble_dataset
from pythia.range_serve import compute_range_block
from pythia.registry import get_default_registry


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="register_range_block",
        description="Attach the P5a range block to the latest registered model row.",
    )
    p.add_argument("--model-name", type=str, default="tft_lite_daily_qqq")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--symbol", type=str, default="QQQ")
    p.add_argument(
        "--historical",
        type=str,
        default="yfinance",
        choices=["yfinance", "stooq", "none"],
        help="backfill provider to fatten the walk-forward sample (default: yfinance; "
        "'none' uses raptor's feed only)",
    )
    p.add_argument("--historical-start", type=_parse_date, default=None)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--initial-train", type=int, default=252)
    p.add_argument("--eval-size", type=int, default=63)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + print the block but do NOT write to the registry",
    )
    args = p.parse_args(argv)

    symbol = args.symbol.upper()
    provider = None if args.historical == "none" else args.historical

    # 1) assemble WITH high/low for the range target.
    result = assemble_dataset(
        args.start,
        args.end,
        symbols=None,
        historical_provider=provider,
        historical_start=args.historical_start,
        hl_symbols={symbol},
    )
    wide = result.daily

    # 2) compute the honest range block (verdict + latest cone).
    block = compute_range_block(
        wide,
        symbol=symbol,
        window=args.window,
        initial_train=args.initial_train,
        eval_size=args.eval_size,
    )

    # Human-readable summary (no secrets).
    cov = block.get("coverage_80")
    cov_s = f"{cov:.3f}" if isinstance(cov, float) and cov == cov else "n/a"
    print(
        f"range block: model={block['model']} badge={block['badge']} "
        f"cov80={cov_s} crps={block.get('crps')} n_eval={block.get('n_eval_obs')} "
        f"cone={block['cone']}",
        file=sys.stderr,
    )

    if args.dry_run:
        json.dump(block, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    # 3) merge into the latest row for --model-name + re-register (UPSERT).
    reg = get_default_registry()
    rec = reg.latest(args.model_name)
    if rec is None:
        print(
            f"ERROR: no registered model under {args.model_name!r}; register the "
            f"price model first (scripts.register_report).",
            file=sys.stderr,
        )
        return 2

    report = dict(rec.report_json)
    report["range"] = block
    reg.register(
        model_name=rec.model_name,
        model_version=rec.model_version,  # SAME version — key stays stable
        dataset_hash=rec.dataset_hash,
        report_json=report,
        artifact_uri=rec.artifact_uri,
        git_sha=rec.git_sha,
    )
    print(
        f"attached range block to {rec.model_name}@{rec.model_version} "
        f"(row id {rec.id}); /latest now serves report_json['range'].",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
