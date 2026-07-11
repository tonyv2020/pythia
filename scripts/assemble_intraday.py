"""CLI: assemble the P3 intraday board dataset and write Parquet.

Usage:
    PYTHIA_DB_DSN=postgresql://…/raptor \
      python -m scripts.assemble_intraday \
        --start 2026-06-05 --end 2026-07-11 \
        --bar-minutes 30 \
        --out data/intraday_30m.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pythia.data.intraday import assemble_intraday_dataset


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="assemble_intraday",
        description="Roll raptor ticks into fixed intraday bars (P3 foundation).",
    )
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bar-minutes", type=int, default=30,
                   help="bar width in minutes = forecast horizon (default 30)")
    p.add_argument("--symbols", type=str, default=None,
                   help="comma-separated override (default: full board)")
    p.add_argument("--include-extended-hours", action="store_true",
                   help="keep pre/post-market bars (default: regular session only)")
    args = p.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    res = assemble_intraday_dataset(
        args.start, args.end,
        bar_minutes=args.bar_minutes,
        session_only=not args.include_extended_hours,
        symbols=symbols,
    )
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res.bars.to_parquet(out, engine="pyarrow", compression="zstd", index=True)

    manifest = {
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "bar_minutes": res.bar_minutes,
        "bars": int(res.bars.shape[0]),
        "cols": int(res.bars.shape[1]),
        "symbols_included": list(res.symbols_included),
        "symbols_missing": list(res.symbols_missing),
        "session_only": not args.include_extended_hours,
        "output": str(out),
    }
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
