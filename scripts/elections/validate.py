"""Strict data integrity validation suite for official Riksdag election results."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any
import pandas as pd

from .config import CANONICAL_PARTIES, ELECTIONS


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections"


class ElectionValidationError(Exception):
    """Raised when an election data integrity check fails."""


def validate_election_results(
    canonical_df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Perform comprehensive data integrity checks on normalized election datasets."""
    report: dict[str, Any] = {
        "status": "PASSED",
        "elections_checked": [],
        "errors": [],
        "warnings": [],
    }

    # 1. Check duplicate rows on (election_year, party)
    duplicates = canonical_df[canonical_df.duplicated(subset=["election_year", "party"], keep=False)]
    if not duplicates.empty:
        err_msg = f"Found duplicate (election_year, party) rows in canonical dataset: {duplicates[['election_year', 'party']].to_dict(orient='records')}"
        report["errors"].append(err_msg)
        report["status"] = "FAILED"

    # 2. Check each election
    for year, meta in sorted(ELECTIONS.items()):
        sub_c = canonical_df[canonical_df["election_year"] == year]
        if sub_c.empty:
            report["errors"].append(f"Missing canonical election results for year {year}")
            report["status"] = "FAILED"
            continue

        # Check exact row count (must be exactly 10 canonical categories)
        if len(sub_c) != len(CANONICAL_PARTIES):
            report["errors"].append(
                f"Year {year}: Expected exactly {len(CANONICAL_PARTIES)} canonical parties, found {len(sub_c)}"
            )
            report["status"] = "FAILED"

        # Check election date
        election_date_val = sub_c["election_date"].iloc[0]
        if election_date_val != meta.election_date.isoformat():
            report["errors"].append(
                f"Year {year}: Election date mismatch: got {election_date_val}, expected {meta.election_date.isoformat()}"
            )
            report["status"] = "FAILED"

        # Check valid_votes_total consistency
        valid_totals = sub_c["valid_votes_total"].unique()
        if len(valid_totals) != 1:
            report["errors"].append(f"Year {year}: Inconsistent valid_votes_total across rows: {valid_totals}")
            report["status"] = "FAILED"
        valid_total = int(valid_totals[0])

        # Check vote totals sum
        sum_votes = int(sub_c["votes"].sum())
        if sum_votes != valid_total:
            report["errors"].append(
                f"Year {year}: Canonical votes sum ({sum_votes}) does not match valid_votes_total ({valid_total}). Diff: {valid_total - sum_votes}"
            )
            report["status"] = "FAILED"

        # Check vote share sum (should be ~100%)
        sum_shares = float(sub_c["vote_share"].sum())
        if abs(sum_shares - 100.0) > 0.05:
            report["errors"].append(
                f"Year {year}: Canonical vote shares sum ({sum_shares:.4f}%) deviates from 100% by > 0.05%"
            )
            report["status"] = "FAILED"

        # Check individual row invariants
        for _, row in sub_c.iterrows():
            party = row["party"]
            votes = int(row["votes"])
            vote_share = float(row["vote_share"])
            src_share = float(row["source_vote_share"]) if pd.notnull(row["source_vote_share"]) else None

            # Non-negative votes
            if votes < 0:
                report["errors"].append(f"Year {year}, Party {party}: Negative votes ({votes})")
                report["status"] = "FAILED"

            # Share bounds [0, 100]
            if not (0.0 <= vote_share <= 100.0):
                report["errors"].append(f"Year {year}, Party {party}: Vote share out of bounds ({vote_share})")
                report["status"] = "FAILED"

            # Calculated share consistency
            expected_share = (votes / valid_total) * 100.0
            if abs(vote_share - expected_share) > 0.001:
                report["errors"].append(
                    f"Year {year}, Party {party}: Calculated vote_share ({vote_share}) deviates from votes/total ({expected_share:.4f})"
                )
                report["status"] = "FAILED"

            # Compare with source published share if single party
            if src_share is not None and party != "OTHER":
                if abs(vote_share - src_share) > 0.02:
                    report["warnings"].append(
                        f"Year {year}, Party {party}: Calculated share ({vote_share:.2f}%) differs from source ({src_share:.2f}%) by > 0.02%"
                    )

        # Source parties check if source_df provided
        if source_df is not None:
            sub_s = source_df[source_df["election_year"] == year]
            sum_src_votes = int(sub_s["votes"].sum())
            if sum_src_votes != valid_total:
                report["errors"].append(
                    f"Year {year}: Source parties votes sum ({sum_src_votes}) does not match valid_votes_total ({valid_total})"
                )
                report["status"] = "FAILED"

        report["elections_checked"].append({
            "year": year,
            "election_date": meta.election_date.isoformat(),
            "valid_votes_total": valid_total,
            "sum_canonical_votes": sum_votes,
            "sum_canonical_shares": round(sum_shares, 4),
            "canonical_parties_count": len(sub_c),
        })

    if report["errors"]:
        raise ElectionValidationError(f"Election validation failed with errors:\n" + "\n".join(report["errors"]))

    return report


def validate_processed_files(processed_dir: Path | str | None = None) -> dict[str, Any]:
    """Load processed CSV files and execute integrity validation."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    canonical_csv = p_dir / "riksdag_election_results.csv"
    source_csv = p_dir / "riksdag_election_results_source_parties.csv"

    if not canonical_csv.exists():
        raise FileNotFoundError(f"Missing canonical CSV at {canonical_csv}")

    df_canonical = pd.read_csv(canonical_csv)
    df_source = pd.read_csv(source_csv) if source_csv.exists() else None

    report = validate_election_results(df_canonical, df_source)
    report_path = p_dir / "election_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Validation successful! Saved report to {report_path}")
    return report


if __name__ == "__main__":
    validate_processed_files()
