"""Precision-weighted OpinionState estimator (Experiment 2).

Implements three weighting arms:
    Arm A: RC1 Baseline (w_age * w_N)
    Arm B: Equal Weighting Diagnostic Control (w_age * 1.0)
    Arm C: Precision Challenger (w_age * w_N * q_g) with derived n_eff^precision
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from scripts.pollofpolls.clr import composition_to_clr
from scripts.pollofpolls.normalize import parse_date
from scripts.pollofpolls.state import (
    OpinionState,
    ReconstructedPoll,
    calculate_poll_reference_date,
    estimate_opinion,
    load_timeseries_dataset,
    subtract_calendar_years,
)
from scripts.pollofpolls.state_config import (
    ALL_CATEGORIES,
    COVARIANCE_DIAGONAL_SHRINKAGE,
    COVARIANCE_LOOKBACK_YEARS,
    MAX_EFFECTIVE_POLLS,
    MAX_ESTIMATE_MATCH_LAG_DAYS,
    MAX_SAMPLE_WEIGHT,
    MIN_POLLS_FOR_HOUSE_EFFECT,
    MIN_RESIDUAL_POLLS,
    MIN_SAMPLE_WEIGHT,
    MIN_SHARE_PCT,
    PARTIES,
    RECENCY_HALF_LIFE_DAYS,
    RECENT_POLL_LOOKBACK_DAYS,
    REFERENCE_CATEGORY,
    SAMPLE_SIZE_BENCHMARK,
)
from scripts.pollofpolls.state_math import (
    alr_to_composition,
    apply_covariance_shrinkage,
    calculate_sample_covariance,
    calculate_sample_mean,
    cholesky_decomposition_with_jitter,
    composition_to_alr,
    sample_multivariate_normal,
    summarize_samples,
)

from .config import (
    ALL_CATEGORIES_9,
    M_MIN_HISTORY,
    M0_PRIMARY,
    M0_SENSITIVITY,
)
from .precision import (
    PollsterPrecisionState,
    compute_sample_size_weight,
    estimate_pollster_precision,
)


def calculate_kish_effective_count(weights: Sequence[float]) -> float:
    """Return Kish's effective count for the supplied final observation weights."""
    values = np.asarray(list(weights), dtype=float)
    if values.ndim != 1 or values.size == 0:
        return 1.0
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Kish weights must be finite and non-negative")
    denominator = float(np.sum(values * values))
    if denominator <= 0.0:
        return 1.0
    return float(np.sum(values) ** 2 / denominator)


def estimate_opinion_with_precision_arm(
    target_as_of: date,
    individual_polls: Sequence[ReconstructedPoll],
    pop_timeseries: Sequence[Dict[str, Any]],
    weighting_arm: str = "precision_challenger",  # "rc1_baseline", "equal_weighting", "precision_challenger"
    precision_state: Optional[PollsterPrecisionState] = None,
    reference_category: str = REFERENCE_CATEGORY,
    m0_prior: float = M0_PRIMARY,
    data_dir: Path | str | None = None,
) -> OpinionState:
    """Estimate OpinionState under specified weighting arm.

    Arm ``rc1_baseline`` is deliberately delegated to the production estimator.
    Keeping a second copy of RC1's eligibility, residual-window, covariance,
    and Kish calculations in an experiment would make a comparison invalid as
    soon as the production implementation changed.  The other arms remain
    experiment-only implementations and are never used by the production
    simulator.

    ``data_dir`` is forwarded only for the shared RC1 path.  It is useful for
    isolated tests and preserves the production estimator's canonical input
    contract (a directory containing ``individual_polls.csv`` and
    ``pollofpolls_timeseries.csv``).
    """
    if weighting_arm not in {"rc1_baseline", "equal_weighting", "precision_challenger"}:
        raise ValueError(f"Unknown weighting arm: {weighting_arm}")
    if weighting_arm == "rc1_baseline":
        if reference_category != REFERENCE_CATEGORY:
            raise ValueError(
                "The frozen RC1 baseline uses REST as its reference category; "
                "use an experiment arm for alternative reference bases."
            )
        return estimate_opinion(as_of=target_as_of, data_dir=data_dir)

    # Build PoP lookup by date
    pop_by_date: Dict[date, Dict[str, float]] = {}
    for row in pop_timeseries:
        pop_by_date[row["date"]] = row["composition"]

    # 1. Match current PoP estimate
    current_estimate_date = target_as_of
    current_pop_composition = pop_by_date.get(current_estimate_date)
    if current_pop_composition is None:
        for lag in range(1, MAX_ESTIMATE_MATCH_LAG_DAYS + 1):
            cand_date = target_as_of - timedelta(days=lag)
            if cand_date in pop_by_date:
                current_estimate_date = cand_date
                current_pop_composition = pop_by_date[cand_date]
                break

    if current_pop_composition is None:
        raise ValueError(f"No contemporaneous PoP estimate found for {target_as_of}")

    # 2. Derive active ALR coordinates for 8 non-reference parties
    alr_mean = composition_to_alr(current_pop_composition)

    # 3. Trailing 4-year ALR residuals & House Effects (frozen RC1 logic)
    lookback_start = subtract_calendar_years(target_as_of, COVARIANCE_LOOKBACK_YEARS)
    raw_residuals_by_house: Dict[str, List[List[float]]] = {}
    active_residuals: List[Dict[str, Any]] = []

    for poll in individual_polls:
        if poll.publication_date is None or poll.publication_date > target_as_of:
            continue
        if poll.interview_end is not None and poll.interview_end > target_as_of:
            continue
        if poll.reference_date is None:
            continue
        if not (lookback_start <= poll.reference_date <= target_as_of):
            continue

        ref_d = poll.reference_date
        pop_comp = pop_by_date.get(ref_d)
        if pop_comp is None:
            for lag in range(1, MAX_ESTIMATE_MATCH_LAG_DAYS + 1):
                cand_d = ref_d - timedelta(days=lag)
                if cand_d in pop_by_date:
                    pop_comp = pop_by_date[cand_d]
                    break
        if pop_comp is None:
            continue

        poll_alr = composition_to_alr(poll.composition)
        pop_alr = composition_to_alr(pop_comp)
        res = [poll_alr[i] - pop_alr[i] for i in range(len(PARTIES))]

        if poll.pollster not in raw_residuals_by_house:
            raw_residuals_by_house[poll.pollster] = []
        raw_residuals_by_house[poll.pollster].append(res)
        active_residuals.append({"pollster": poll.pollster, "residual": res})

    # House effects
    house_effects_alr: Dict[str, List[float]] = {}
    for pollster, r_list in raw_residuals_by_house.items():
        if len(r_list) >= MIN_POLLS_FOR_HOUSE_EFFECT:
            house_effects_alr[pollster] = calculate_sample_mean(r_list)
        else:
            house_effects_alr[pollster] = [0.0] * len(PARTIES)

    # Adjusted residuals & shrunk covariance
    adjusted_residuals: List[List[float]] = []
    for r in active_residuals:
        he = house_effects_alr.get(r["pollster"], [0.0] * len(PARTIES))
        adj = [r["residual"][i] - he[i] for i in range(len(PARTIES))]
        adjusted_residuals.append(adj)

    if len(adjusted_residuals) < MIN_RESIDUAL_POLLS:
        cov_raw = [[0.01 if i == j else 0.0 for j in range(len(PARTIES))] for i in range(len(PARTIES))]
    else:
        cov_raw = calculate_sample_covariance(adjusted_residuals)

    residual_cov = apply_covariance_shrinkage(cov_raw, COVARIANCE_DIAGONAL_SHRINKAGE)

    # 4. Estimate precision multipliers if challenger arm
    if weighting_arm == "precision_challenger":
        if precision_state is None:
            precision_state = estimate_pollster_precision(
                target_as_of=target_as_of,
                individual_polls=individual_polls,
                pop_by_date=pop_by_date,
                m0_prior=m0_prior,
            )
        precision_qs = precision_state.precision_multipliers_q
    else:
        precision_qs = {}

    # 5. Effective Polls & Information Calculation (Trailing 60 days)
    recent_lookback_start = target_as_of - timedelta(days=RECENT_POLL_LOOKBACK_DAYS)
    recent_polls_selected: List[ReconstructedPoll] = []
    base_weights: List[float] = []
    applied_qs: List[float] = []

    for poll in individual_polls:
        if poll.publication_date is None or poll.publication_date > target_as_of:
            continue
        if poll.interview_end is not None and poll.interview_end > target_as_of:
            continue
        if poll.reference_date is None:
            continue
        if not (recent_lookback_start <= poll.reference_date <= target_as_of):
            continue

        recent_polls_selected.append(poll)

        age_days = (target_as_of - poll.reference_date).days
        w_age = math.exp(-math.log(2.0) * age_days / RECENCY_HALF_LIFE_DAYS)

        if weighting_arm == "equal_weighting":
            w_n = 1.0
            q_g = 1.0
        elif weighting_arm == "rc1_baseline":
            w_n = compute_sample_size_weight(poll.sample_size)
            q_g = 1.0
        elif weighting_arm == "precision_challenger":
            w_n = compute_sample_size_weight(poll.sample_size)
            q_g = precision_qs.get(poll.pollster, 1.0)
        else:
            raise ValueError(f"Unknown weighting arm: {weighting_arm}")

        w_base = w_age * w_n
        base_weights.append(w_base)
        applied_qs.append(q_g)

    # Kish's effective sample size is defined on the final weights actually
    # used by the estimator.  Both the numerator and denominator therefore
    # include q; a denominator with only one q factor is not mathematically
    # coherent and can overstate information.
    if base_weights:
        effective_weights = [w * q for w, q in zip(base_weights, applied_qs)]
        n_eff = calculate_kish_effective_count(effective_weights)

        n_eff_used = min(max(n_eff, 1.0), MAX_EFFECTIVE_POLLS)
    else:
        n_eff = 1.0
        n_eff_used = 1.0

    # 6. State Covariance
    state_cov = [
        [residual_cov[i][j] / n_eff_used for j in range(len(PARTIES))]
        for i in range(len(PARTIES))
    ]

    cholesky_L, jitter_used = cholesky_decomposition_with_jitter(state_cov)

    diagnostics = {
        "weighting_arm": weighting_arm,
        "recent_poll_count": len(recent_polls_selected),
        "effective_poll_count_raw": round(n_eff, 4),
        "effective_poll_count_used": round(n_eff_used, 4),
        "jitter_used": jitter_used,
        "mean_q_applied": round(float(np.mean(applied_qs)), 4) if applied_qs else 1.0,
    }

    return OpinionState(
        as_of=target_as_of,
        estimate_date=current_estimate_date,
        estimate_age_days=(target_as_of - current_estimate_date).days,
        parties=PARTIES,
        mean_pct={p: current_pop_composition[p] for p in PARTIES},
        rest_pct=current_pop_composition["REST"],
        mean_alr=alr_mean,
        covariance_alr=state_cov,
        residual_covariance_alr=residual_cov,
        recent_poll_count=len(recent_polls_selected),
        effective_poll_count=n_eff_used,
        residual_poll_count=len(active_residuals),
        covariance_fallback_used=False,
        house_effects_alr=house_effects_alr,
        diagnostics=diagnostics,
        _cholesky_L=cholesky_L,
    )
