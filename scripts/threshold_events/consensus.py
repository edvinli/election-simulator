"""Leakage-safe final pre-election polling consensus engine for threshold events.

Applies canonical lookback window, latest-per-pollster deduplication, bounded
sample-size weighting, party-specific coverage tracking, and non-zero missing handling.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from scripts.threshold_events.config import (
    KNOWN_NAMED_MINOR_PARTIES,
    PARLIAMENTARY_PARTIES,
    POLLS_FILE,
    normalize_party_name,
)


@dataclass(frozen=True)
class ContributingPollsterRecord:
    """Audit record of a pollster's selected latest poll in the election window."""
    election_year: int
    election_date: date
    window_days: int
    poll_id: str
    pollster: str
    pollster_original: str
    interview_start: Optional[date]
    interview_end: date
    publication_date: date
    sample_size: Optional[int]
    sample_size_missing: bool
    weight: float
    party_support: Dict[str, float]


@dataclass(frozen=True)
class PartyConsensusSummary:
    """Party-level consensus result and party-specific coverage metrics."""
    party: str
    consensus_pct: float
    party_eligible_poll_count: int
    party_contributing_poll_count: int
    party_pollster_count: int
    party_sample_size_coverage: float
    contributing_weights_sum: float


@dataclass(frozen=True)
class ElectionConsensusResult:
    """Complete pre-election consensus result for an election."""
    election_year: int
    election_date: date
    window_days: int
    window_start: date
    window_end: date
    total_eligible_polls_in_window: int
    total_retained_pollsters: int
    election_sample_size_coverage: float
    party_consensus: Dict[str, PartyConsensusSummary]
    contributing_records: List[ContributingPollsterRecord]


def compute_sample_size_weight(sample_size: Optional[float]) -> Tuple[float, bool]:
    """Compute bounded sqrt sample size weight: clip(sqrt(N / 1000), 0.7, 1.5).
    
    Missing, NaN, or non-positive sample size returns default weight 1.0.
    """
    if sample_size is None or pd.isna(sample_size) or float(sample_size) <= 0:
        return 1.0, True
    raw_w = np.sqrt(float(sample_size) / 1000.0)
    clipped_w = float(np.clip(raw_w, 0.7, 1.5))
    return round(clipped_w, 4), False


def build_final_polling_consensus(
    election_date: date,
    election_year: int,
    polls_df: pd.DataFrame,
    window_days: int = 14,
) -> Optional[ElectionConsensusResult]:
    """Construct final pre-election polling consensus for an election date.
    
    Strict anti-leakage eligibility:
      1. publication_date <= election_date
      2. interview_end <= election_date
      3. interview_end >= election_date - window_days
      4. interview_end and publication_date must not be missing/indeterminate
    
    Latest-poll-per-pollster rule:
      Selects the latest eligible poll per pollster at the poll level using deterministic tie-breakers.
      Missing party values in that poll remain missing and are never treated as 0%.
    """
    window_start = election_date - timedelta(days=window_days)
    window_end = election_date
    
    w_start_str = window_start.isoformat()
    w_end_str = window_end.isoformat()
    
    # 1. Filter eligible records with valid date metadata
    valid_dates_mask = (
        polls_df["interview_end"].notna() &
        polls_df["publication_date"].notna() &
        (polls_df["interview_end"] != "") &
        (polls_df["publication_date"] != "")
    )
    
    df_valid = polls_df[valid_dates_mask].copy()
    
    # Anti-leakage date filtering
    eligible_mask = (
        (df_valid["interview_end"] >= w_start_str) &
        (df_valid["interview_end"] <= w_end_str) &
        (df_valid["publication_date"] <= w_end_str)
    )
    eligible_polls = df_valid[eligible_mask].copy()
    
    total_eligible_polls_count = int(eligible_polls["poll_id"].nunique())
    if total_eligible_polls_count == 0:
        return None
        
    # 2. Normalize party names in polling data
    eligible_polls["party_canon"] = eligible_polls["party"].apply(normalize_party_name)

    # A poll is represented by one row per party.  Duplicate canonical party
    # rows are ambiguous (for example, both ``L`` and ``Liberalerna`` for the
    # same poll) and must not be averaged or silently discarded by the pivot.
    duplicate_party_rows = eligible_polls[
        eligible_polls.duplicated(subset=["poll_id", "party_canon"], keep=False)
    ]
    if not duplicate_party_rows.empty:
        duplicate_keys = sorted(
            {
                (str(row.poll_id), str(row.party_canon))
                for row in duplicate_party_rows.itertuples(index=False)
            }
        )
        raise ValueError(
            "Duplicate party rows for eligible poll(s); refusing to build "
            f"consensus: {duplicate_keys}"
        )
    
    # Count total eligible polls per party in the window (before deduplication)
    party_eligible_counts: Dict[str, int] = {}
    for p_code, p_grp in eligible_polls.groupby("party_canon"):
        valid_support = p_grp[p_grp["support"].notna()]
        party_eligible_counts[p_code] = int(valid_support["poll_id"].nunique())
        
    # 3. Pivot support values to wide format (one row per poll).
    #
    # Do not include nullable metadata in a pivot_table index.  pandas drops
    # rows with a null index key by default, which would silently discard an
    # otherwise eligible poll before the documented missing-N fallback can
    # run (and would also discard polls with a missing interview_start).
    # Pivot only on the non-null poll_id/party key, then join the metadata
    # selected from the original rows.  The source has one metadata record per
    # poll, but the deterministic sort/drop_duplicates below also makes the
    # behavior well-defined if a future source revision repeats a poll's
    # metadata across party rows.
    pivot_cols = [
        "poll_id",
        "pollster",
        "pollster_original",
        "interview_start",
        "interview_end",
        "publication_date",
        "sample_size",
    ]

    # All party rows for one poll must carry the same poll-level metadata.
    # Otherwise joining one selected metadata row to the party pivot would
    # combine support values from one poll with dates/N/house information from
    # another row, corrupting both latest-poll selection and audit coverage.
    inconsistent_metadata: List[Tuple[str, str]] = []
    metadata_columns = pivot_cols[1:]
    for poll_id, poll_group in eligible_polls.groupby("poll_id", sort=False, dropna=False):
        for metadata_column in metadata_columns:
            if poll_group[metadata_column].nunique(dropna=False) > 1:
                inconsistent_metadata.append((str(poll_id), metadata_column))
    if inconsistent_metadata:
        raise ValueError(
            "Inconsistent poll metadata across party rows; refusing to build "
            f"consensus: {sorted(inconsistent_metadata)}"
        )

    support_wide = eligible_polls.pivot(
        index="poll_id",
        columns="party_canon",
        values="support",
    ).reset_index()

    metadata = (
        eligible_polls[pivot_cols]
        .sort_values(
            by=["poll_id", "interview_end", "publication_date", "interview_start", "sample_size"],
            ascending=[True, False, False, False, False],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(subset=["poll_id"], keep="first")
    )
    piv = metadata.merge(
        support_wide,
        on="poll_id",
        how="left",
        validate="one_to_one",
    )
    
    # 4. Deterministic sort and latest-per-pollster selection
    piv_sorted = piv.sort_values(
        by=["interview_end", "publication_date", "interview_start", "sample_size", "poll_id"],
        ascending=[False, False, False, False, False],
        na_position="last",
        kind="mergesort",
    )
    latest_per_pollster = piv_sorted.drop_duplicates(subset=["pollster"], keep="first").copy()
    
    # 5. Build ContributingPollsterRecord objects
    contributing_records: List[ContributingPollsterRecord] = []
    for _, row in latest_per_pollster.iterrows():
        n_raw = row["sample_size"]
        n_int = int(float(n_raw)) if pd.notnull(n_raw) and float(n_raw) > 0 else None
        weight, is_missing_n = compute_sample_size_weight(n_int)
        
        party_support: Dict[str, float] = {}
        for col in latest_per_pollster.columns:
            if col not in pivot_cols and pd.notnull(row[col]):
                party_support[col] = float(row[col])
                
        contributing_records.append(
            ContributingPollsterRecord(
                election_year=election_year,
                election_date=election_date,
                window_days=window_days,
                poll_id=str(row["poll_id"]),
                pollster=str(row["pollster"]),
                pollster_original=str(row["pollster_original"]),
                interview_start=date.fromisoformat(row["interview_start"]) if pd.notnull(row["interview_start"]) and str(row["interview_start"]) != "" else None,
                interview_end=date.fromisoformat(row["interview_end"]),
                publication_date=date.fromisoformat(row["publication_date"]),
                sample_size=n_int,
                sample_size_missing=is_missing_n,
                weight=weight,
                party_support=party_support,
            )
        )
        
    # 6. Compute party-specific consensus and coverage
    all_discovered_parties = set()
    for cp in contributing_records:
        all_discovered_parties.update(cp.party_support.keys())
        
    party_summaries: Dict[str, PartyConsensusSummary] = {}
    for p in sorted(all_discovered_parties):
        if p in ["OTHER", "REST", "UNCERTAIN"]:
            continue
            
        # Only polls that actually reported party p
        reporting_polls = [cp for cp in contributing_records if p in cp.party_support]
        if not reporting_polls:
            continue
            
        w_sum = sum(cp.weight for cp in reporting_polls)
        weighted_val = sum(cp.party_support[p] * cp.weight for cp in reporting_polls)
        consensus_val = round(weighted_val / w_sum, 4)
        
        party_contrib_cnt = len(reporting_polls)
        party_house_cnt = len(set(cp.pollster for cp in reporting_polls))
        non_missing_n_cnt = sum(1 for cp in reporting_polls if not cp.sample_size_missing)
        sample_size_cov = round(non_missing_n_cnt / party_contrib_cnt, 4) if party_contrib_cnt > 0 else 0.0
        party_eligible_cnt = party_eligible_counts.get(p, party_contrib_cnt)
        
        party_summaries[p] = PartyConsensusSummary(
            party=p,
            consensus_pct=consensus_val,
            party_eligible_poll_count=party_eligible_cnt,
            party_contributing_poll_count=party_contrib_cnt,
            party_pollster_count=party_house_cnt,
            party_sample_size_coverage=sample_size_cov,
            contributing_weights_sum=round(w_sum, 4),
        )
        
    # Overall election sample size coverage
    total_retained = len(contributing_records)
    elec_n_cov = round(sum(1 for cp in contributing_records if not cp.sample_size_missing) / total_retained, 4) if total_retained > 0 else 0.0
    
    return ElectionConsensusResult(
        election_year=election_year,
        election_date=election_date,
        window_days=window_days,
        window_start=window_start,
        window_end=window_end,
        total_eligible_polls_in_window=total_eligible_polls_count,
        total_retained_pollsters=total_retained,
        election_sample_size_coverage=elec_n_cov,
        party_consensus=party_summaries,
        contributing_records=contributing_records,
    )
