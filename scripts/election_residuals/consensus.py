"""Construct standardized final pre-election polling consensus per election."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .config import (
    ALL_CATEGORIES,
    DEFAULT_POLLS_FILE,
    LOOKBACK_WINDOW_DAYS,
    PARLIAMENTARY_PARTIES,
    SAMPLE_SIZE_BENCHMARK,
    WEIGHT_MAX,
    WEIGHT_MIN,
)


@dataclass(frozen=True)
class ContributingPoll:
    """Represents a single latest retained poll from a pollster in the election window."""

    poll_id: str
    pollster: str
    pollster_original: str
    interview_start: date | None
    interview_end: date
    publication_date: date
    sample_size: int | None
    sample_size_missing: bool
    weight: float
    party_support: dict[str, float]


@dataclass(frozen=True)
class ElectionPollConsensus:
    """Pre-election polling consensus and contributing polls for one election."""

    election_date: date
    election_year: int
    window_start: date
    window_end: date
    total_eligible_polls_in_window: int
    retained_pollsters_count: int
    consensus_composition: dict[str, float]
    contributing_polls: list[ContributingPoll]


def compute_poll_weight(sample_size: int | float | None) -> tuple[float, bool]:
    """Calculate bounded sqrt sample-size weight: clip(sqrt(n / 1000), 0.7, 1.5).

    Returns:
        (weight, sample_size_missing_flag)
    """
    if sample_size is None or pd.isna(sample_size) or float(sample_size) <= 0:
        return 1.0, True

    n_val = float(sample_size)
    raw_weight = np.sqrt(n_val / SAMPLE_SIZE_BENCHMARK)
    clipped_weight = float(np.clip(raw_weight, WEIGHT_MIN, WEIGHT_MAX))
    return clipped_weight, False


def build_election_polling_consensus(
    election_date: date,
    polls_df: pd.DataFrame,
    window_days: int = LOOKBACK_WINDOW_DAYS,
) -> ElectionPollConsensus:
    """Construct final pre-election polling consensus for an election date.

    Eligibility criteria:
        1. publication_date <= election_date
        2. interview_end <= election_date
        3. interview_end >= election_date - window_days

    Deduplication:
        For each pollster, retain only the latest eligible poll.
        Tie-breaking rule:
            1. interview_end descending
            2. publication_date descending
            3. interview_start descending
            4. sample_size descending
            5. poll_id descending (deterministic tie-breaker)
    """
    window_start = election_date - timedelta(days=window_days)
    window_end = election_date

    # 1. Filter eligible rows
    w_start_str = window_start.isoformat()
    w_end_str = window_end.isoformat()

    sub = polls_df[
        (polls_df["interview_end"] >= w_start_str)
        & (polls_df["interview_end"] <= w_end_str)
        & (polls_df["publication_date"] <= w_end_str)
    ].copy()

    total_eligible_polls = sub["poll_id"].nunique()
    if total_eligible_polls == 0:
        raise ValueError(
            f"No eligible polls found for election {election_date} in window {w_start_str} to {w_end_str}"
        )

    # 2. Pivot to wide format (one row per poll)
    pivot_cols = [
        "poll_id",
        "pollster",
        "pollster_original",
        "interview_start",
        "interview_end",
        "publication_date",
        "sample_size",
    ]
    piv = sub.pivot_table(index=pivot_cols, columns="party", values="support").reset_index()

    # Ensure all parliamentary parties exist as columns
    for p in PARLIAMENTARY_PARTIES:
        if p not in piv.columns:
            piv[p] = np.nan

    # 3. Deterministic sort and deduplication
    piv_sorted = piv.sort_values(
        by=["interview_end", "publication_date", "interview_start", "sample_size", "poll_id"],
        ascending=[False, False, False, False, False],
    )
    latest_per_pollster = piv_sorted.drop_duplicates(subset=["pollster"], keep="first").copy()

    # 4. Compute weights and construct ContributingPoll objects
    contributing_polls: list[ContributingPoll] = []
    for _, row in latest_per_pollster.iterrows():
        n_raw = row["sample_size"]
        n_int = int(n_raw) if pd.notnull(n_raw) and float(n_raw) > 0 else None
        weight, is_missing_n = compute_poll_weight(n_int)

        party_vals: dict[str, float] = {}
        for p in PARLIAMENTARY_PARTIES:
            val = row[p]
            if pd.notnull(val):
                party_vals[p] = float(val)

        contributing_polls.append(
            ContributingPoll(
                poll_id=str(row["poll_id"]),
                pollster=str(row["pollster"]),
                pollster_original=str(row["pollster_original"]),
                interview_start=date.fromisoformat(row["interview_start"]) if pd.notnull(row["interview_start"]) else None,
                interview_end=date.fromisoformat(row["interview_end"]),
                publication_date=date.fromisoformat(row["publication_date"]),
                sample_size=n_int,
                sample_size_missing=is_missing_n,
                weight=round(weight, 4),
                party_support=party_vals,
            )
        )

    # 5. Calculate weighted means for the 8 parliamentary parties
    consensus_comp: dict[str, float] = {}
    for p in PARLIAMENTARY_PARTIES:
        # Sum weights of polls that reported this party
        eligible_polls = [cp for cp in contributing_polls if p in cp.party_support]
        if eligible_polls:
            w_sum = sum(cp.weight for cp in eligible_polls)
            val_weighted = sum(cp.party_support[p] * cp.weight for cp in eligible_polls)
            consensus_comp[p] = round(val_weighted / w_sum, 4)
        else:
            consensus_comp[p] = 0.0

    # 6. Derive REST = 100 - sum(8 parliamentary parties)
    sum_8 = sum(consensus_comp[p] for p in PARLIAMENTARY_PARTIES)
    rest_val = round(100.0 - sum_8, 4)

    # Sanity checks
    if rest_val < -0.01:
        raise ValueError(
            f"Election {election_date}: Materially invalid composition, sum of 8 parties ({sum_8:.4f}%) > 100%"
        )
    if rest_val < 0.0:
        rest_val = 0.0

    consensus_comp["REST"] = rest_val

    # Strict sum verification
    tot_comp = sum(consensus_comp.values())
    if abs(tot_comp - 100.0) > 0.001:
        raise ValueError(f"Election {election_date}: Consensus sum ({tot_comp:.4f}%) does not equal 100%")

    return ElectionPollConsensus(
        election_date=election_date,
        election_year=election_date.year,
        window_start=window_start,
        window_end=window_end,
        total_eligible_polls_in_window=total_eligible_polls,
        retained_pollsters_count=len(contributing_polls),
        consensus_composition=consensus_comp,
        contributing_polls=contributing_polls,
    )
