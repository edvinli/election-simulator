"""CLI for exporting the frozen Candidate-A static publication contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from scripts.simulator.config import DEFAULT_ELECTION_DATE, DEFAULT_SIMULATION_SAMPLES, DEFAULT_SIMULATION_SEED
from scripts.simulator.engine import simulate_election

from .exporter import export_static_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export compact static JSON for the frozen ElectionSimulator")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--election-date", default=DEFAULT_ELECTION_DATE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SIMULATION_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SIMULATION_SEED)
    parser.add_argument("--generated-at-utc", default=None)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = simulate_election(
        as_of=args.as_of,
        election_date=args.election_date,
        samples=args.samples,
        seed=args.seed,
    )
    manifest = export_static_data(
        result,
        output_dir=args.output_dir,
        generated_at_utc=args.generated_at_utc,
        calibration_dir=args.calibration_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

