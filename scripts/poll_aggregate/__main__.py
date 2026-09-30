"""Build the SwedishPolls aggregate: ``python -m scripts.poll_aggregate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_2026_RESULT_MANIFEST,
    DEFAULT_ELECTION_RESULTS_FILE,
    DEFAULT_POLLS_FILE,
    SPECS,
    HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS,
    HISTORY_START_ELECTION,
    METADATA_FILENAME,
    TIMESERIES_FILENAME,
)
from .outputs import build_outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polls", type=Path, default=DEFAULT_POLLS_FILE)
    parser.add_argument("--election-results", type=Path, default=DEFAULT_ELECTION_RESULTS_FILE)
    parser.add_argument("--certified-result", type=Path, action="append", default=None,
                        help="Certified result manifest (repeatable); defaults to the 2026 manifest")
    parser.add_argument("--version", choices=sorted(SPECS), default="v0.1", dest="aggregate_version",
                        help="frozen aggregate version to build (default v0.1)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="defaults to the version's own directory")
    parser.add_argument("--reuse-stored-fits", action="store_true",
                        help="reuse segment fits recorded in the output directory's metadata")
    args = parser.parse_args(argv)
    spec = SPECS[args.aggregate_version]
    args.output_dir = args.output_dir or spec.output_dir

    manifests = args.certified_result if args.certified_result is not None else [DEFAULT_2026_RESULT_MANIFEST]
    metadata = build_outputs(args.polls, args.election_results, manifests, args.output_dir, spec,
                             reuse_stored_fits=args.reuse_stored_fits)
    print(json.dumps({k: metadata[k] for k in ("rows", "first_date", "last_date", "polls_used", "stats")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
