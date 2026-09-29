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
from .build import build_aggregate, write_timeseries
from .data import load_elections, load_polls, sha256_file


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
    args = parser.parse_args(argv)
    spec = SPECS[args.aggregate_version]
    args.output_dir = args.output_dir or spec.output_dir

    manifests = args.certified_result if args.certified_result is not None else [DEFAULT_2026_RESULT_MANIFEST]
    elections = load_elections(args.election_results, manifests)
    dataset = load_polls(args.polls, [e.obs_date for e in elections])
    result = build_aggregate(dataset.polls, elections, spec=spec)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_timeseries(result, args.output_dir / TIMESERIES_FILENAME, version=spec.version)
    metadata = {
        "aggregate_version": spec.version,
        "reported_level": "consensus_reading" if spec.consensus_window_days else "latent_support",
        "consensus_window_days": spec.consensus_window_days,
        "inputs": {
            "polls": {"path": str(args.polls.name), "sha256": sha256_file(args.polls)},
            "election_results": {"path": str(args.election_results.name),
                                 "sha256": sha256_file(args.election_results)},
            "certified_results": [{"path": str(m.name), "sha256": sha256_file(m)} for m in manifests],
        },
        "history_start_election": HISTORY_START_ELECTION.isoformat(),
        "historical_result_availability_lag_days": HISTORICAL_RESULT_AVAILABILITY_LAG_DAYS,
        "elections": [{"date": e.obs_date.isoformat(), "available": e.available.isoformat()} for e in elections],
        "houses": result.houses,
        "polls_used": len(dataset.polls),
        "exclusions": {reason: {"count": len(ids), "poll_ids": sorted(ids)}
                       for reason, ids in sorted(dataset.exclusions.items())},
        "imputed_sample_size": sorted(dataset.imputed_sample_size),
        "approximate_fieldwork": sorted(dataset.approximate_fieldwork),
        "overlap_discounted_count": len(dataset.overlap_discounted),
        "segments": [
            {
                "hyperparameters_fit_date": s.fit_date.isoformat(),
                "reference_election": s.reference_election,
                "observations": s.n_obs,
                "log_likelihood": round(s.loglik, 4),
                "optimizer_iterations": s.optimizer_iterations,
                "optimizer_converged": s.optimizer_converged,
                "hyperparameters": {k: float(f"{v:.6g}") for k, v in s.params.to_dict().items()},
                "consensus_weights": (
                    {h: round(w, 6) for h, w in s.consensus_weights.items()} if s.consensus_weights else None
                ),
            }
            for s in result.segments
        ],
        "rows": len(result.rows),
        "first_date": result.rows[0].day.isoformat() if result.rows else None,
        "last_date": result.rows[-1].day.isoformat() if result.rows else None,
        "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.stats.items()},
    }
    (args.output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: metadata[k] for k in ("rows", "first_date", "last_date", "polls_used", "stats")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
