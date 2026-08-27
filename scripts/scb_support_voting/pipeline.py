"""Pipeline entrypoint for SCB PSU support-voting data acquisition and processing.
"""
import argparse
import sys
from scripts.scb_support_voting.fetch import fetch_all_scb_tables
from scripts.scb_support_voting.process import process_all
from scripts.scb_support_voting.qa import run_all_qa


def main() -> int:
    parser = argparse.ArgumentParser(description="SCB PSU Support Voting Data Pipeline")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch latest raw data from SCB API and update raw archives/manifest.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Process datasets offline from archived raw files (default).",
    )
    args = parser.parse_args()

    if args.fetch:
        print("=== Step 1: Fetching Raw SCB Data ===")
        fetch_all_scb_tables()

    print("=== Step 2: Processing Datasets Offline ===")
    process_all()

    print("=== Step 3: Running QA and Generating Validation Report ===")
    report = run_all_qa()

    if not report["assertions"]["all_assertions_passed"]:
        print("ERROR: One or more validation assertions failed!", file=sys.stderr)
        return 1

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
