"""Election-eve final-poll consensus under election-noise residual amendments 001 and 002.

``scripts/election_residuals/consensus.py`` is left byte-for-byte unchanged so
that archived forecasts and frozen harnesses stay reproducible. This module
is the amended definition (``docs/election_noise_residual_amendment_001.md``
and ``..._002.md``):

* **001, eligibility.** A poll counts for election day ``E`` only if
  ``publication_date < E`` and ``interview_end < E``, with
  ``interview_end >= E - window_days``. This is the information a forecaster
  had on election eve.
* **002, completeness.** Every eligible poll reaches deduplication. The
  committed builder pivots with ``sample_size`` and ``interview_start`` among
  the pandas index keys, so a poll missing either value was silently dropped.
  Here a missing sample size gets weight 1.0, as ``compute_poll_weight``
  specifies. A missing value sorts last in the tie-break.

Everything else is the committed builder's rule: the latest eligible poll
per pollster, ordered by ``interview_end``, ``publication_date``,
``interview_start``, ``sample_size`` and ``poll_id`` (all descending); bounded
sqrt sample-size weights; and REST as the remainder. On data with no
election-day polls and no missing keys, both builders return identical
results (``tests/test_eve_consensus.py``).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import LOOKBACK_WINDOW_DAYS, PARLIAMENTARY_PARTIES
from .consensus import ContributingPoll, ElectionPollConsensus, compute_poll_weight

RULE_ID = "election-eve-consensus-amendments-001-002"
_META = ("pollster", "pollster_original", "interview_start", "interview_end", "publication_date", "sample_size")
_ORDER = ("interview_end", "publication_date", "interview_start", "sample_size", "poll_id")


def eligible_eve_rows(election_date: date, polls_df: pd.DataFrame, window_days: int = LOOKBACK_WINDOW_DAYS) -> pd.DataFrame:
    """Long-format rows of polls eligible under amendment 001 (dates are ISO strings)."""

    e = election_date.isoformat()
    lo = (election_date - timedelta(days=window_days)).isoformat()
    published = polls_df["publication_date"].notna() & (polls_df["publication_date"] < e)
    ended = polls_df["interview_end"].notna() & (polls_df["interview_end"] < e) & (polls_df["interview_end"] >= lo)
    return polls_df[published & ended]


def build_eve_polling_consensus(
    election_date: date,
    polls_df: pd.DataFrame,
    window_days: int = LOOKBACK_WINDOW_DAYS,
) -> ElectionPollConsensus:
    """Final pre-election polling consensus under amendments 001 and 002."""

    sub = eligible_eve_rows(election_date, polls_df, window_days)
    total_eligible = int(sub["poll_id"].nunique())
    if total_eligible == 0:
        raise ValueError(f"No eligible polls for election {election_date} under {RULE_ID}")

    # One metadata row per poll, keyed by poll_id alone, so no missing value can
    # drop a poll. Support is pivoted on poll_id with the committed aggregation.
    meta = sub.drop_duplicates("poll_id").set_index("poll_id")[list(_META)]
    support = sub.pivot_table(index="poll_id", columns="party", values="support")
    wide = meta.join(support).reset_index()
    for p in PARLIAMENTARY_PARTIES:
        if p not in wide.columns:
            wide[p] = np.nan

    ordered = wide.sort_values(by=list(_ORDER), ascending=[False] * len(_ORDER), na_position="last", kind="mergesort")
    latest = ordered.drop_duplicates(subset=["pollster"], keep="first")

    contributing: list[ContributingPoll] = []
    for _, row in latest.iterrows():
        n_raw = row["sample_size"]
        n_int = int(n_raw) if pd.notnull(n_raw) and float(n_raw) > 0 else None
        weight, missing = compute_poll_weight(n_int)
        contributing.append(
            ContributingPoll(
                poll_id=str(row["poll_id"]),
                pollster=str(row["pollster"]),
                pollster_original=str(row["pollster_original"]),
                interview_start=date.fromisoformat(row["interview_start"]) if pd.notnull(row["interview_start"]) else None,
                interview_end=date.fromisoformat(row["interview_end"]),
                publication_date=date.fromisoformat(row["publication_date"]),
                sample_size=n_int,
                sample_size_missing=missing,
                weight=round(weight, 4),
                party_support={p: float(row[p]) for p in PARLIAMENTARY_PARTIES if pd.notnull(row[p])},
            )
        )

    composition: dict[str, float] = {}
    for p in PARLIAMENTARY_PARTIES:
        reporting = [c for c in contributing if p in c.party_support]
        if reporting:
            w_sum = sum(c.weight for c in reporting)
            composition[p] = round(sum(c.party_support[p] * c.weight for c in reporting) / w_sum, 4)
        else:
            composition[p] = 0.0
    sum_8 = sum(composition[p] for p in PARLIAMENTARY_PARTIES)
    rest = round(100.0 - sum_8, 4)
    if rest < -0.01:
        raise ValueError(f"Election {election_date}: sum of 8 parties ({sum_8:.4f}%) > 100%")
    composition["REST"] = max(rest, 0.0)
    if abs(sum(composition.values()) - 100.0) > 0.001:
        raise ValueError(f"Election {election_date}: consensus does not sum to 100%")

    return ElectionPollConsensus(
        election_date=election_date,
        election_year=election_date.year,
        window_start=election_date - timedelta(days=window_days),
        window_end=election_date - timedelta(days=1),
        total_eligible_polls_in_window=total_eligible,
        retained_pollsters_count=len(contributing),
        consensus_composition=composition,
        contributing_polls=contributing,
    )
