"""Pipeline runner for GeographicProjection v1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .config import DEFAULT_PROCESSED_GEOGRAPHY_DIR, DEFAULT_RAW_GEOGRAPHY_DIR
from .evaluate import run_all_historical_evaluations
from .fetch import fetch_raw_geography_data
from .process import process_all_geography_data


def run_geography_pipeline(fetch: bool = False) -> int:
    """Execute complete geography acquisition, processing, and historical evaluations."""
    if fetch:
        print(">>> Step 1: Fetching raw geographical datasets from Valmyndigheten ...")
        fetch_raw_geography_data()

    print("\n>>> Step 2: Processing and normalizing geographical matrices ...")
    processed_paths = process_all_geography_data()
    print(f"Processed datasets: {processed_paths}")

    print("\n==========================================================================================")
    print("HISTORICAL GEOGRAPHIC PROJECTION EVALUATIONS (2014 -> 2018 & 2018 -> 2022)")
    print("==========================================================================================")
    eval_results = run_all_historical_evaluations()

    for k, m in eval_results.items():
        print(f"\n--- Evaluation: {k} ---")
        print(f"  Baseline: {m.baseline_year} -> Target: {m.target_year} (Mode: {m.mode}, IPF iterations: {m.ipf_iterations})")
        print(f"  Constituency Party-Share MAE: {m.constituency_share_mae * 100:.3f}%")
        print(f"  Constituency Valid Votes MAPE: {m.constituency_valid_votes_mape:.2f}% (Max Abs Err: {m.constituency_valid_votes_max_err:,.0f})")
        print(f"  National Share Max Error: {m.national_share_max_err:.2e}")
        print(f"  Total Absolute Seat Error: {m.total_seat_error} seats")
        print(f"  Projected Seats: {m.projected_seats}")
        print(f"  Certified Seats: {m.certified_seats}")
        print(f"  Seat Differences (Proj - Cert): {m.seat_differences}")
        print("  Party MAEs (%):", {p: round(v * 100, 3) for p, v in m.party_share_maes.items()})

    print("\nGeographic Projection pipeline completed successfully!")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Swedish Riksdag Geographic Projection Pipeline Runner.")
    parser.add_argument("--fetch", action="store_true", help="Download raw geographic datasets from Valmyndigheten.")
    args = parser.parse_args(args_list)
    return run_geography_pipeline(fetch=args.fetch)


if __name__ == "__main__":
    sys.exit(main())
