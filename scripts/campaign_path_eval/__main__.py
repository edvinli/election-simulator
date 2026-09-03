"""CLI for the rolling retrospective campaign-path evaluation."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from .evaluate import (
    DEFAULT_BASE_SEED,
    DEFAULT_ORIGIN_STRIDE_DAYS,
    DEFAULT_PATH_DAYS,
    DEFAULT_SAMPLES,
    DEFAULT_START,
    evaluate_campaign_paths,
    write_evaluation_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.campaign_path_eval")
    parser.add_argument("--path-days", type=int, default=DEFAULT_PATH_DAYS)
    parser.add_argument("--stride-days", type=int, default=DEFAULT_ORIGIN_STRIDE_DAYS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat())
    parser.add_argument(
        "--timeseries-file",
        type=Path,
        default=PROCESSED / "pollofpolls" / "pollofpolls_timeseries.csv",
    )
    parser.add_argument("--backtest-dir", type=Path, default=PROCESSED / "backtests")
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=REPO_ROOT / "diagnostics" / "campaign_paths"
    )
    args = parser.parse_args(argv)

    evaluation = evaluate_campaign_paths(
        timeseries_file=args.timeseries_file,
        path_days=args.path_days,
        stride_days=args.stride_days,
        samples=args.samples,
        base_seed=args.seed,
        start=date.fromisoformat(args.start),
    )
    written = write_evaluation_artifacts(
        evaluation,
        backtest_dir=args.backtest_dir,
        diagnostics_dir=args.diagnostics_dir,
    )
    print(json.dumps(evaluation.summary, indent=2, sort_keys=True))
    print(json.dumps(written, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
