"""End-to-end execution pipeline for official Swedish parliamentary election results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .fetch import fetch_all_elections
from .normalize import normalize_all_elections
from .validate import validate_processed_files


def run_pipeline(
    fetch: bool = True,
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> int:
    """Run full election results data pipeline: fetch -> parse/normalize -> validate."""
    r_dir = Path(raw_dir) if raw_dir else None
    p_dir = Path(processed_dir) if processed_dir else None

    if fetch:
        print(">>> Step 1: Fetching official election documents from Valmyndigheten ...")
        fetch_all_elections(raw_dir=r_dir)
    else:
        print(">>> Step 1: Offline mode (using existing raw election files) ...")

    print("\n>>> Step 2: Normalizing and building processed datasets ...")
    res = normalize_all_elections(raw_dir=r_dir, processed_dir=p_dir)

    print("\n>>> Step 3: Validating data integrity ...")
    report = validate_processed_files(processed_dir=p_dir)

    print(f"\nPipeline successfully completed! Status: {report['status']}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official Swedish Parliamentary Election Results Data Pipeline."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network fetching and build from existing raw files.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Custom raw directory path.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Custom processed directory path.",
    )

    args = parser.parse_args(args_list)
    return run_pipeline(
        fetch=not args.offline,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
