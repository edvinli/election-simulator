"""End-to-end pipeline for historical party-election threshold events study.
"""
import argparse
from pathlib import Path
import sys
from typing import Sequence

from scripts.threshold_events.episodes import generate_and_save_canonical_datasets
from scripts.threshold_events.qa import run_all_threshold_qa


def run_pipeline() -> int:
    """Run full threshold events pipeline: generate canonical datasets -> run QA."""
    print("=== Step 1: Generating Canonical Threshold Events Datasets ===")
    generate_and_save_canonical_datasets()

    print("\n=== Step 2: Running QA Diagnostics and Robustness Checks ===")
    report = run_all_threshold_qa()

    if not report["assertions"]["all_assertions_passed"]:
        print("ERROR: One or more assertions failed!", file=sys.stderr)
        return 1

    print("\nPipeline completed successfully!")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Historical Party-Election Threshold Events Pipeline."
    )
    args = parser.parse_args(args_list)
    return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
