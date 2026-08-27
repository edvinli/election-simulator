"""Normalize source party results into canonical election result datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence
import pandas as pd

from .config import CANONICAL_PARTIES, ELECTIONS
from .parse import ElectionParsedData, SourcePartyResult, parse_election_by_year


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "elections"
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "elections"


def build_source_parties_table(
    all_parsed_elections: Sequence[ElectionParsedData],
) -> pd.DataFrame:
    """Build detailed source-party level dataframe preserving every reported party line."""
    rows: list[dict[str, Any]] = []

    for election in all_parsed_elections:
        for p in election.source_parties:
            calc_share = round((p.votes / election.valid_votes_total) * 100.0, 4)
            rows.append({
                "election_date": election.election_date.isoformat(),
                "election_year": election.election_year,
                "canonical_party": p.canonical_party,
                "party_source_name": p.party_source_name,
                "party_source_code": p.party_source_code,
                "votes": p.votes,
                "source_vote_share": p.source_vote_share,
                "vote_share": calc_share,
                "valid_votes_total": election.valid_votes_total,
                "source_url": p.source_url,
                "retrieved_at": p.retrieved_at,
            })

    return pd.DataFrame(rows)


def build_canonical_results_table(
    all_parsed_elections: Sequence[ElectionParsedData],
) -> pd.DataFrame:
    """Build normalized canonical election results table with fixed 10-party grid per election."""
    rows: list[dict[str, Any]] = []

    for election in all_parsed_elections:
        meta = ELECTIONS[election.election_year]

        # Group source parties by canonical code
        by_canon: dict[str, list[SourcePartyResult]] = {cat: [] for cat in CANONICAL_PARTIES}
        for p in election.source_parties:
            by_canon[p.canonical_party].append(p)

        for cat in CANONICAL_PARTIES:
            p_list = by_canon[cat]
            if p_list:
                tot_votes = sum(p.votes for p in p_list)
                # Primary source label
                if cat == "OTHER":
                    source_name = "Övriga partier"
                else:
                    source_name = p_list[0].party_source_name

                # Published source vote share
                if len(p_list) == 1 and p_list[0].source_vote_share is not None:
                    src_share = p_list[0].source_vote_share
                else:
                    src_share = round(sum(p.source_vote_share or 0.0 for p in p_list), 2)
            else:
                tot_votes = 0
                source_name = f"{cat} (ej deltagit)" if cat == "FI" else cat
                src_share = 0.0

            calc_share = round((tot_votes / election.valid_votes_total) * 100.0, 4)

            rows.append({
                "election_date": election.election_date.isoformat(),
                "election_year": election.election_year,
                "party": cat,
                "party_source_name": source_name,
                "votes": tot_votes,
                "vote_share": calc_share,
                "source_vote_share": src_share,
                "valid_votes_total": election.valid_votes_total,
                "source_url": election.source_url,
                "retrieved_at": election.retrieved_at,
            })

    return pd.DataFrame(rows)


def normalize_all_elections(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Parse raw election files and output source-level and canonical CSV datasets."""
    r_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = r_dir / "retrieval_manifest.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    all_parsed = [
        parse_election_by_year(year, raw_dir=r_dir, manifest=manifest)
        for year in sorted(ELECTIONS.keys())
    ]

    df_source = build_source_parties_table(all_parsed)
    df_canonical = build_canonical_results_table(all_parsed)

    source_csv_path = p_dir / "riksdag_election_results_source_parties.csv"
    canonical_csv_path = p_dir / "riksdag_election_results.csv"

    df_source.to_csv(source_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    df_canonical.to_csv(canonical_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved {len(df_source)} source party rows to {source_csv_path}")
    print(f"Saved {len(df_canonical)} canonical election rows to {canonical_csv_path}")

    return {
        "source_parties_df": df_source,
        "canonical_df": df_canonical,
        "source_csv": str(source_csv_path),
        "canonical_csv": str(canonical_csv_path),
    }


if __name__ == "__main__":
    normalize_all_elections()
