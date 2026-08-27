"""Export a frozen Candidate A draw bundle for paired external comparison."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from scripts.simulator.config import DEFAULT_ELECTION_DATE, DEFAULT_SIMULATION_SEED, DEFAULT_SIMULATION_SAMPLES, MODEL_VERSION
from scripts.simulator.engine import simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict

from .adapters import bundle_from_simulation_result, write_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export unmodified ElectionSimulator draws for Botten Ada comparison")
    parser.add_argument("--as-of", required=True, help="Forecast cutoff date (YYYY-MM-DD)")
    parser.add_argument("--election-date", default=DEFAULT_ELECTION_DATE)
    parser.add_argument("--baseline-year", type=int, default=2022)
    parser.add_argument("--samples", type=int, default=DEFAULT_SIMULATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = simulate_election(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
        baseline_year=args.baseline_year,
    )
    source = {
        "source_git_commit": result.manifest.get("source_git_commit"),
        "source_worktree_clean": result.manifest.get("source_worktree_clean"),
        "deterministic_payload_sha256": build_canonical_summary_dict(result)["deterministic_payload_sha256"],
        "note": "Candidate A is exported without calibration or model changes; raw draws are for benchmark exchange only, not the prospective compact archive.",
    }
    bundle = bundle_from_simulation_result(result, source=source)
    write_bundle(bundle, args.output)
    print(json.dumps({"output": str(args.output), "samples": args.samples, "model_version": MODEL_VERSION}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
