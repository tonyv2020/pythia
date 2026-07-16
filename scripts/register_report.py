"""CLI: register a completed training run in the pythia registry.

Used by both the nightly retrain and the manual runs. Reads the report
JSON produced by scripts.train_p1_tft, computes the dataset SHA-256, tags
the row with the current git SHA + a caller-supplied artifact URI.

Usage:
    python -m scripts.register_report \\
        --model-name    tft_lite_daily_qqq \\
        --model-version v20260711-a \\
        --dataset       data/board.parquet \\
        --report        data/report.json \\
        --artifact-uri  s3://pythia/artifacts/2026-07-11.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pythia.registry import (
    compute_dataset_hash,
    current_git_sha,
    get_default_registry,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="register_report")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--model-version", type=str, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--artifact-uri", type=str, default="local://none")
    p.add_argument("--repo", type=Path, default=None)
    args = p.parse_args(argv)

    report = json.loads(args.report.read_text())
    dataset_hash = compute_dataset_hash(args.dataset)
    git_sha = current_git_sha(args.repo)

    reg = get_default_registry()
    row_id = reg.register(
        model_name=args.model_name,
        model_version=args.model_version,
        dataset_hash=dataset_hash,
        report_json=report,
        artifact_uri=args.artifact_uri,
        git_sha=git_sha,
    )
    print(
        json.dumps(
            {
                "row_id": row_id,
                "model_name": args.model_name,
                "model_version": args.model_version,
                "dataset_hash": dataset_hash,
                "git_sha": git_sha,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
