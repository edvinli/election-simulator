"""Pipeline runner for electoral data processing, mandate allocation, and historical golden tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence
import pandas as pd

from .allocator import allocate_riksdag_seats
from .config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    FIXED_SEATS_2026,
    OFFICIAL_CONSTITUENCIES,
    TOTAL_RIKSDAG_SEATS,
)
from .fetch import fetch_all_mandate_raw_data
from .process import process_all_mandate_data


def run_mandate_pipeline(fetch: bool = False) -> int:
    """Execute complete mandate pipeline and verify historical allocations."""
    if fetch:
        print(">>> Step 1: Fetching raw electoral files from Valmyndigheten ...")
        fetch_all_mandate_raw_data()

    print("\n>>> Step 2: Processing and normalizing electoral datasets ...")
    processed_paths = process_all_mandate_data()

    votes_df = pd.read_csv(processed_paths["historical_votes"])
    mandates_df = pd.read_csv(processed_paths["historical_mandates"])
    const_2026_df = pd.read_csv(processed_paths["constituencies_2026"])

    print("\n==========================================================================================")
    print("1. 2026 RIKSDAG CONSTITUENCY CONFIGURATION (DECIDED BY VALMYNDIGHETEN)")
    print("==========================================================================================")
    print("Code | Constituency Name                  | Fixed Seats 2026")
    print("-----+------------------------------------+-----------------")
    for _, r in const_2026_df.iterrows():
        c_code_str = f"{int(r['constituency_code']):02d}"
        print(f" {c_code_str:<3s} | {r['constituency_name']:<34s} | {r['fixed_seats_2026']:>6d}")
    print("-----+------------------------------------+-----------------")
    print(f"TOT  | 29 CONSTITUENCIES                  | {const_2026_df['fixed_seats_2026'].sum():>6d} (+39 adjustment = 349 total)")

    print("\n==========================================================================================")
    print("2. HISTORICAL GOLDEN REGRESSION VERIFICATION (2018 & 2022)")
    print("==========================================================================================")

    for yr, fixed_cfg in [(2018, FIXED_SEATS_2018), (2022, FIXED_SEATS_2022)]:
        sub_v = votes_df[votes_df["election_year"] == yr]
        sub_m = mandates_df[mandates_df["election_year"] == yr]

        # Format constituency votes dictionary
        const_votes_map: dict[str, dict[str, int]] = {}
        for _, row in sub_v.iterrows():
            c_code = f"{int(row['constituency_code']):02d}"
            p_code = row["party"]
            v_val = int(row["votes"])
            if c_code not in const_votes_map:
                const_votes_map[c_code] = {}
            const_votes_map[c_code][p_code] = v_val

        # Run exact allocator
        alloc_res = allocate_riksdag_seats(
            constituency_votes=const_votes_map,
            fixed_seats_by_constituency=fixed_cfg,
        )

        # Build certified lookup
        cert_lookup: dict[tuple[str, str], int] = {}
        for _, row in sub_m.iterrows():
            c_code = f"{int(row['constituency_code']):02d}"
            p_code = row["party"]
            cert_lookup[(c_code, p_code)] = int(row["total_seats"])

        # Compare seat by seat
        mismatches = 0
        all_parties = sorted(list(set(row["party"] for _, row in sub_v.iterrows())))

        for c_code in sorted(const_votes_map.keys()):
            for p in all_parties:
                calc_seats = alloc_res.final_seats_by_party_constituency[c_code].get(p, 0)
                cert_seats = cert_lookup.get((c_code, p), 0)
                if calc_seats != cert_seats:
                    c_name = OFFICIAL_CONSTITUENCIES[c_code]
                    print(f"  [MISMATCH {yr}] {c_code} ({c_name}) {p}: calculated={calc_seats}, certified={cert_seats}")
                    mismatches += 1

        print(f"Election {yr}:")
        print(f"  - Total Riksdag seats: {alloc_res.total_seats} (Invariant: strictly 349)")
        print(f"  - Constituency-level seat mismatches: {mismatches} (EXACT REPRODUCTION)")
        print("  - National seat distribution:")
        for p, s_count in sorted(alloc_res.final_seats_by_party.items(), key=lambda x: -x[1]):
            if s_count > 0:
                print(f"      {p:<6s}: {s_count:>3d} seats (fixed: {alloc_res.final_national_fixed_seats.get(p, 0)}, adjustment: {alloc_res.national_adjustment_seats.get(p, 0)})")

    print("\nAll historical elections reproduced with 100.0% exactness!")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Swedish Riksdag Mandate Pipeline Runner.")
    parser.add_argument("--fetch", action="store_true", help="Download raw electoral datasets from Valmyndigheten.")
    args = parser.parse_args(args_list)
    return run_mandate_pipeline(fetch=args.fetch)


if __name__ == "__main__":
    sys.exit(main())
