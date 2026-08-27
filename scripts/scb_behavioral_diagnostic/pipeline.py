"""Execution pipeline for SCB behavioral threshold diagnostic (Step 3).
"""
import argparse
from pathlib import Path
import sys
from typing import Sequence

from scripts.scb_behavioral_diagnostic.qa import run_full_scb_behavioral_qa


def run_pipeline() -> int:
    """Run full SCB behavioral threshold diagnostic pipeline."""
    print("=== Step 3: SCB Behavioral Threshold Diagnostic Pipeline ===")
    report = run_full_scb_behavioral_qa()

    if not report["assertions"]["all_assertions_passed"]:
        print("ERROR: One or more assertions failed!", file=sys.stderr)
        return 1

    print("\nStep 3 pipeline completed successfully!")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SCB Behavioral Threshold Diagnostic Pipeline (Step 3)."
    )
    args = parser.parse_args(args_list)
    return run_pipeline()


if __name__ == "__main__":
    sys.exit(main())
