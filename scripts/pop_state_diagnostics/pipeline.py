"""Orchestrator pipeline CLI for Step 4A PoP State-Dependence Diagnostic."""

import argparse
from pathlib import Path
import sys

from scripts.pop_state_diagnostics.config import PROCESSED_DATA_DIR
from scripts.pop_state_diagnostics.qa import run_full_state_diagnostics_qa


def main() -> None:
    """Run Step 4A state diagnostics pipeline."""
    parser = argparse.ArgumentParser(description="Step 4A PoP State-Dependence Diagnostic Pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory to save processed diagnostic outputs",
    )
    args = parser.parse_args()

    print("=== Step 4A: PoP State-Dependence Diagnostic Pipeline ===")
    report = run_full_state_diagnostics_qa(processed_dir=args.output_dir)
    print("\nStep 4A pipeline completed successfully!")


if __name__ == "__main__":
    main()
