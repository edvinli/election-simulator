"""Reference-invariant empirical pollster precision estimation (Experiment 2).

Estimates pollster precision multipliers q_g from de-meaned CLR residuals,
standardizing for sample size (N) and shrinking low-sample houses toward the pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.state import (
    ReconstructedPoll,
    subtract_calendar_years,
)
from scripts.pollofpolls.state_config import (
    ALL_CATEGORIES,
    COVARIANCE_LOOKBACK_YEARS,
    MAX_ESTIMATE_MATCH_LAG_DAYS,
    MAX_SAMPLE_WEIGHT,
    MIN_POLLS_FOR_HOUSE_EFFECT,
    MIN_SAMPLE_WEIGHT,
    MIN_SHARE_PCT,
    PARTIES,
    REFERENCE_CATEGORY,
    SAMPLE_SIZE_BENCHMARK,
)

from .config import (
    ALL_CATEGORIES_9,
    M_MIN_HISTORY,
    M0_PRIMARY,
    M0_SENSITIVITY,
    Q_MAX,
    Q_MIN,
)


@dataclass(frozen=True)
class HousePrecisionProfile:
    """Historical precision metrics and multiplier for a specific polling house."""

    pollster: str
    poll_count: int
    median_sample_size: float
    missing_sample_size_count: int
    raw_dispersion_clr: float       # Mean unadjusted CLR squared dispersion
    adj_dispersion_clr: float       # Sample-size adjusted dispersion D_adj = D * w_N^2
    shrunk_dispersion_clr: float    # Shrunk dispersion using empirical-Bayes M0
    raw_precision_ratio: float      # sqrt(s_pool^2 / s_shrunk^2)
    normalized_multiplier_q: float  # Final bounded standard-deviation precision multiplier q_g
    is_eligible: bool               # poll_count >= M_MIN_HISTORY


@dataclass
class PollsterPrecisionState:
    """As-of precision estimation container for all polling houses."""

    as_of: date
    lookback_start: date
    total_eligible_polls: int
    pooled_dispersion_clr: float
    profiles_by_house: Dict[str, HousePrecisionProfile]
    precision_multipliers_q: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def compute_sample_size_weight(sample_size: Optional[int]) -> float:
    """Compute standard RC1 sample-size weight w_N in [0.7, 1.5]."""
    if sample_size is not None and sample_size > 0:
        raw_w = math.sqrt(sample_size / SAMPLE_SIZE_BENCHMARK)
        return min(max(raw_w, MIN_SAMPLE_WEIGHT), MAX_SAMPLE_WEIGHT)
    return 1.0


def extract_historical_clr_residuals(
    target_as_of: date,
    individual_polls: Sequence[ReconstructedPoll],
    pop_by_date: Dict[date, Dict[str, float]],
    lookback_years: int = COVARIANCE_LOOKBACK_YEARS,
    categories: Sequence[str] = ALL_CATEGORIES_9,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Extract leakage-safe CLR residuals and house-effect locations.

    The canonical Poll of Polls series is not a valid precision reference here:
    it contains the poll being scored, so residuals would be mechanically
    shrunk for influential houses.  Instead, each poll is compared with an
    equal-weight CLR mean of *other* polling houses with the same reference
    date, or the nearest prior date within the existing three-day matching
    window.  A poll is omitted when no leave-one-house-out reference exists.

    Eligibility is deliberately aligned with the frozen OpinionState
    residual contract: publication must be strictly before ``target_as_of``,
    interview_end must be present and no later than the cutoff, and the
    reference date must lie in the trailing window.
    """
    lookback_start = subtract_calendar_years(target_as_of, lookback_years)
    eligible_polls: List[ReconstructedPoll] = []

    for poll in individual_polls:
        if poll.publication_date is None or poll.publication_date >= target_as_of:
            continue
        if poll.interview_end is None or poll.interview_end > target_as_of:
            continue
        if poll.reference_date is None:
            continue
        if not (lookback_start <= poll.reference_date <= target_as_of):
            continue
        eligible_polls.append(poll)

    # Index eligible polls by reference date once.  This keeps the leave-one-
    # house-out construction linear in the number of polls rather than doing a
    # full scan of all polls for every residual.
    polls_by_reference_date: Dict[date, List[ReconstructedPoll]] = {}
    for poll in eligible_polls:
        assert poll.reference_date is not None  # narrowed above
        polls_by_reference_date.setdefault(poll.reference_date, []).append(poll)

    raw_residuals_by_house: Dict[str, List[np.ndarray]] = {}
    active_poll_records: List[Dict[str, Any]] = []

    for poll in eligible_polls:
        assert poll.reference_date is not None  # narrowed above

        reference_date_used: date | None = None
        reference_polls: List[ReconstructedPoll] = []
        for lag in range(MAX_ESTIMATE_MATCH_LAG_DAYS + 1):
            candidate_date = poll.reference_date - timedelta(days=lag)
            candidates = [
                candidate
                for candidate in polls_by_reference_date.get(candidate_date, [])
                if candidate.poll_id != poll.poll_id and candidate.pollster != poll.pollster
            ]
            if candidates:
                reference_date_used = candidate_date
                reference_polls = candidates
                break

        if not reference_polls:
            continue

        # Compute a contemporaneous, leave-one-house-out CLR reference.  The
        # geometric (CLR) mean is the natural equal-weight center on the
        # simplex and is independent of the choice of ALR reference category.
        poll_clr, _ = composition_to_clr(poll.composition, categories=categories)
        reference_clrs = np.array(
            [composition_to_clr(candidate.composition, categories=categories)[0] for candidate in reference_polls],
            dtype=float,
        )
        reference_clr = np.mean(reference_clrs, axis=0)
        r_clr = poll_clr - reference_clr  # Shape (9,)

        if poll.pollster not in raw_residuals_by_house:
            raw_residuals_by_house[poll.pollster] = []
        raw_residuals_by_house[poll.pollster].append(r_clr)

        w_n = compute_sample_size_weight(poll.sample_size)
        active_poll_records.append({
            "pollster": poll.pollster,
            "reference_date": poll.reference_date,
            "sample_size": poll.sample_size,
            "w_n": w_n,
            "r_clr": r_clr,
            "reference_method": "leave_one_pollster_out_clr_mean",
            "reference_date_used": reference_date_used,
            "reference_poll_ids": tuple(candidate.poll_id for candidate in reference_polls),
            "reference_pollsters": tuple(sorted({candidate.pollster for candidate in reference_polls})),
        })

    # Estimate CLR house effect locations h_g^CLR
    house_effects_clr: Dict[str, np.ndarray] = {}
    for pollster, r_list in raw_residuals_by_house.items():
        if len(r_list) >= MIN_POLLS_FOR_HOUSE_EFFECT:
            house_effects_clr[pollster] = np.mean(r_list, axis=0)
        else:
            house_effects_clr[pollster] = np.zeros(len(categories), dtype=float)

    return active_poll_records, house_effects_clr


def estimate_pollster_precision(
    target_as_of: date,
    individual_polls: Sequence[ReconstructedPoll],
    pop_by_date: Dict[date, Dict[str, float]],
    m0_prior: float = M0_PRIMARY,
    m_min: int = M_MIN_HISTORY,
    lookback_years: int = COVARIANCE_LOOKBACK_YEARS,
    categories: Sequence[str] = ALL_CATEGORIES_9,
) -> PollsterPrecisionState:
    """Estimate leakage-safe reference-invariant pollster precision multipliers q_g.

    For a poll with sampling variance approximately proportional to ``1/N``,
    multiplying its squared residual by ``w_N**2`` expresses dispersion at the
    benchmark sample size.  The configured clipping bounds make that
    standardization robust to implausible or missing sample sizes; they are not
    claimed to estimate a literal survey-design variance.
    """
    lookback_start = subtract_calendar_years(target_as_of, lookback_years)
    records, house_effects = extract_historical_clr_residuals(
        target_as_of=target_as_of,
        individual_polls=individual_polls,
        pop_by_date=pop_by_date,
        lookback_years=lookback_years,
        categories=categories,
    )

    k_dim = float(len(categories))  # 9 categories
    records_by_house: Dict[str, List[Dict[str, Any]]] = {}
    all_adj_dispersions: List[float] = []

    for rec in records:
        pollster = rec["pollster"]
        he = house_effects.get(pollster, np.zeros(int(k_dim), dtype=float))
        u_clr = rec["r_clr"] - he
        d_raw = float(np.sum(u_clr ** 2) / k_dim)  # Reference-invariant Aitchison squared norm
        d_adj = d_raw * (rec["w_n"] ** 2)         # N-standardized dispersion

        rec["d_raw"] = d_raw
        rec["d_adj"] = d_adj

        if pollster not in records_by_house:
            records_by_house[pollster] = []
        records_by_house[pollster].append(rec)
        all_adj_dispersions.append(d_adj)

    # Pooled historical dispersion
    if all_adj_dispersions:
        s_pool_sq = float(np.mean(all_adj_dispersions))
    else:
        s_pool_sq = 1.0

    profiles: Dict[str, HousePrecisionProfile] = {}
    raw_ratios: Dict[str, float] = {}

    for pollster, p_records in records_by_house.items():
        m_g = len(p_records)
        raw_disps = [r["d_raw"] for r in p_records]
        adj_disps = [r["d_adj"] for r in p_records]
        sample_sizes = [r["sample_size"] for r in p_records if r["sample_size"] is not None]
        missing_n = m_g - len(sample_sizes)
        med_n = float(np.median(sample_sizes)) if sample_sizes else np.nan

        s_raw_sq = float(np.mean(raw_disps))
        s_adj_sq = float(np.mean(adj_disps))

        is_eligible = (m_g >= m_min)

        if is_eligible:
            # Empirical-Bayes shrinkage toward s_pool_sq
            s_shrunk_sq = (m_g / (m_g + m0_prior)) * s_adj_sq + (m0_prior / (m_g + m0_prior)) * s_pool_sq
            raw_ratio = math.sqrt(s_pool_sq / max(s_shrunk_sq, 1e-8))
        else:
            s_shrunk_sq = s_pool_sq
            raw_ratio = 1.0

        raw_ratios[pollster] = raw_ratio
        profiles[pollster] = HousePrecisionProfile(
            pollster=pollster,
            poll_count=m_g,
            median_sample_size=med_n,
            missing_sample_size_count=missing_n,
            raw_dispersion_clr=s_raw_sq,
            adj_dispersion_clr=s_adj_sq,
            shrunk_dispersion_clr=s_shrunk_sq,
            raw_precision_ratio=raw_ratio,
            normalized_multiplier_q=1.0,  # Updated after normalization
            is_eligible=is_eligible,
        )

    # Normalize precision multipliers across eligible houses so Mean(q_g) approx 1.0
    eligible_ratios = [r for p, r in raw_ratios.items() if profiles[p].is_eligible]
    mean_ratio = float(np.mean(eligible_ratios)) if eligible_ratios else 1.0

    multipliers_q: Dict[str, float] = {}
    for pollster, prof in profiles.items():
        if prof.is_eligible:
            norm_q = raw_ratios[pollster] / max(mean_ratio, 1e-6)
            final_q = min(max(norm_q, Q_MIN), Q_MAX)
        else:
            final_q = 1.0
        multipliers_q[pollster] = round(final_q, 4)

        # Update profile with final q
        profiles[pollster] = HousePrecisionProfile(
            pollster=prof.pollster,
            poll_count=prof.poll_count,
            median_sample_size=prof.median_sample_size,
            missing_sample_size_count=prof.missing_sample_size_count,
            raw_dispersion_clr=prof.raw_dispersion_clr,
            adj_dispersion_clr=prof.adj_dispersion_clr,
            shrunk_dispersion_clr=prof.shrunk_dispersion_clr,
            raw_precision_ratio=prof.raw_precision_ratio,
            normalized_multiplier_q=final_q,
            is_eligible=prof.is_eligible,
        )

    # Diagnostics & correlation checks
    eligible_profiles = [p for p in profiles.values() if p.is_eligible]
    corr_diagnostics: Dict[str, Any] = {}
    if len(eligible_profiles) >= 3:
        qs = [p.normalized_multiplier_q for p in eligible_profiles]
        ns = [p.median_sample_size if not np.isnan(p.median_sample_size) else 1000.0 for p in eligible_profiles]
        counts = [float(p.poll_count) for p in eligible_profiles]

        r_n, _ = pearsonr(qs, ns)
        rho_n, _ = spearmanr(qs, ns)
        r_count, _ = pearsonr(qs, counts)
        rho_count, _ = spearmanr(qs, counts)

        corr_diagnostics = {
            "eligible_house_count": len(eligible_profiles),
            "corr_q_vs_median_n": {"pearson": round(float(r_n), 3), "spearman": round(float(rho_n), 3)},
            "corr_q_vs_poll_count": {"pearson": round(float(r_count), 3), "spearman": round(float(rho_count), 3)},
        }

    return PollsterPrecisionState(
        as_of=target_as_of,
        lookback_start=lookback_start,
        total_eligible_polls=len(records),
        pooled_dispersion_clr=round(s_pool_sq, 6),
        profiles_by_house=profiles,
        precision_multipliers_q=multipliers_q,
        diagnostics={
            **corr_diagnostics,
            "reference_method": "leave_one_pollster_out_clr_mean",
            "eligible_poll_count_before_reference_match": sum(
                1
                for poll in individual_polls
                if poll.publication_date is not None
                and poll.publication_date < target_as_of
                and poll.interview_end is not None
                and poll.interview_end <= target_as_of
                and poll.reference_date is not None
                and lookback_start <= poll.reference_date <= target_as_of
            ),
            "matched_poll_count": len(records),
            "unmatched_poll_count": max(
                0,
                sum(
                    1
                    for poll in individual_polls
                    if poll.publication_date is not None
                    and poll.publication_date < target_as_of
                    and poll.interview_end is not None
                    and poll.interview_end <= target_as_of
                    and poll.reference_date is not None
                    and lookback_start <= poll.reference_date <= target_as_of
                )
                - len(records),
            ),
            "dispersion_standardization": "D_adj = D_raw * clip(sqrt(N/1000), 0.7, 1.5)^2; missing N -> 1.0",
            "effective_sample_size_formula": "(sum(w_base*q))^2 / sum((w_base*q)^2)",
        },
    )
