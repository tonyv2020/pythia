"""CLI: assemble the P0 dataset and write Parquet.

Usage:
    python -m scripts.assemble_dataset \
        --start 2024-01-01 \
        --end   2026-07-10 \
        --out   data/board_2024_onwards.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pythia.data import assemble_dataset
from pythia.data.assembler import write_dataset


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="assemble_dataset",
        description="Pull the P0 board + calendar features from raptor Postgres.",
    )
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="comma-separated symbol override (default: full board + VIX/rate proxies)",
    )
    # --- D8 historical backfill (opt-in) ---
    p.add_argument(
        "--historical",
        type=str,
        default=None,
        choices=["yfinance", "stooq"],
        help="backfill OLD daily bars from this provider (raptor feed stays truth for recent)",
    )
    p.add_argument(
        "--historical-start",
        type=_parse_date,
        default=None,
        help="earliest date to backfill from (default: --start). Use a years-back date to fatten the sample.",
    )
    p.add_argument(
        "--no-adjust",
        action="store_true",
        help="use RAW (unadjusted) historical prices to match raptor's convention "
        "(default: split/div-ADJUSTED to avoid fake split-day returns; see docs/d8-backfill.md)",
    )
    args = p.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    result = assemble_dataset(
        args.start,
        args.end,
        symbols=symbols,
        historical_provider=args.historical,
        historical_adjust=not args.no_adjust,
        historical_start=args.historical_start,
    )
    out = write_dataset(result, args.out)

    manifest = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "rows": int(result.daily.shape[0]),
        "cols": int(result.daily.shape[1]),
        "symbols_included": list(result.symbols_included),
        "symbols_missing": list(result.symbols_missing),
        "symbols_backfilled": list(result.symbols_backfilled),
        "historical_provider": args.historical,
        "historical_start": args.historical_start.isoformat() if args.historical_start else None,
        "historical_adjusted": (args.historical is not None) and (not args.no_adjust),
        "output": str(out),
    }
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
