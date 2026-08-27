"""Comparative evaluation engine and calendar-block bootstrap (Experiment 2).

Evaluates Arm A (RC1), Arm B (Equal-Weighting Control), Arm C (Precision Primary),
and Sensitivity Arm C25 over canonical rolling backtest cases using common random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from scripts.pollofpolls.backtest_metrics import calculate_crps, precompute_crps_sample_term
from scripts.pollofpolls.clr import clr_to_composition, clr_to_composition_matrix, composition_to_clr
from scripts.pollofpolls.normalize import parse_date
from scripts.pollofpolls.state import (
    OpinionState,
    ReconstructedPoll,
    calculate_poll_reference_date,
    load_timeseries_dataset,
)
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES, REFERENCE_CATEGORY
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
)
from scripts.pop_baseline.metrics import continuous_crps, energy_score

from .config import (
    ALL_CATEGORIES_9,
    BASE_RANDOM_SEED,
    BOOTSTRAP_BLOCK_MONTHS,
    BOOTSTRAP_REPLICATIONS,
    DEFAULT_HORIZONS,
    EVALUATION_DRAWS_COUNT,
    M0_PRIMARY,
    M0_SENSITIVITY,
    PARLIAMENTARY_PARTIES,
    POLLS_FILE,
    POP_TIMESERIES_FILE,
)
from .manifest import RollingCaseManifestRecord, build_canonical_rolling_manifest
from .opinion_state import estimate_opinion_with_precision_arm
from .precision import PollsterPrecisionState, estimate_pollster_precision


@dataclass(frozen=True)
class CaseEvaluationResult:
    """Out-of-sample proper scores and calibration metrics for a single backtest case."""

    origin_date: date
    horizon_days: int
    target_date: date
    calendar_block_6m: str

    # Energy Scores (9-party vote share space)
    es_arm_a_rc1: float
    es_arm_b_equal: float
    es_arm_c_precision: float
    es_sens_c25: float

    # Paired Relative ES Improvements (Positive = Candidate is better)
    rel_es_imp_precision: float  # (ES_A - ES_C) / ES_A
    rel_es_imp_equal: float      # (ES_A - ES_B) / ES_A

    # Marginal CRPS (mean across 8 parliamentary parties)
    crps_arm_a_rc1: float
    crps_arm_b_equal: float
    crps_arm_c_precision: float
    crps_sens_c25: float

    # Paired Relative CRPS Improvement
    rel_crps_imp_precision: float  # (CRPS_A - CRPS_C) / CRPS_A

    # REST CRPS
    crps_rest_rc1: float
    crps_rest_precision: float

    # Effective Poll Counts
    n_eff_rc1: float
    n_eff_precision: float


def derive_case_seed(base_seed: int, origin_date: date, horizon_days: int) -> int:
    """Deterministic integer seed for Monte Carlo draws of a case."""
    token = f"{base_seed}:{origin_date.isoformat()}:{horizon_days}:opinion_precision".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % 2_147_483_647


def sample_predictive_vote_shares(
    opinion_state: OpinionState,
    candidate_transitions: Sequence[HistoricalTransition],
    standard_normals_z: np.ndarray,      # Shape (M, 8)
    transition_indices: np.ndarray,       # Shape (M,)
    sign_flips: np.ndarray,               # Shape (M,) in {-1.0, 1.0}
    categories: Sequence[str] = ALL_CATEGORIES_9,
) -> np.ndarray:
    """Generate predictive vote shares (M, 9) using common random numbers across arms."""
    m_draws = len(standard_normals_z)
    cholesky_L = np.array(opinion_state._cholesky_L, dtype=float)  # (8, 8)
    alr_mean = np.array(opinion_state.mean_alr, dtype=float)        # (8,)

    # 1. State draws in ALR space: theta_alr = alr_mean + L * z
    theta_alr = alr_mean + standard_normals_z @ cholesky_L.T           # (M, 8)

    # 2. Convert state draws to 9-party simplex composition
    exp_alr = np.exp(theta_alr)                                        # (M, 8)
    sum_exp = 1.0 + np.sum(exp_alr, axis=1, keepdims=True)            # (M, 1)
    shares_8 = 100.0 * (exp_alr / sum_exp)                            # (M, 8)
    shares_rest = 100.0 / sum_exp                                      # (M, 1)
    shares_9 = np.hstack([shares_8, shares_rest])                     # (M, 9)

    # 3. Convert to CLR coordinates
    log_shares = np.log(np.maximum(shares_9, 1e-8))
    mean_log = np.mean(log_shares, axis=1, keepdims=True)
    state_clr = log_shares - mean_log                                  # (M, 9)

    # 4. Apply sampled historical transition with sign flip
    if candidate_transitions:
        trans_matrix = np.array([candidate_transitions[idx].clr_transition for idx in transition_indices])  # (M, 9)
        final_clr = state_clr + trans_matrix * sign_flips[:, np.newaxis]
    else:
        final_clr = state_clr

    # 5. Inverse CLR back to 9-party simplex percentage vote shares
    exp_clr = np.exp(final_clr)
    final_shares = 100.0 * (exp_clr / np.sum(exp_clr, axis=1, keepdims=True))  # (M, 9)
    return final_shares


def evaluate_all_manifest_cases(
    manifest: Sequence[RollingCaseManifestRecord],
    individual_polls: Sequence[ReconstructedPoll],
    pop_timeseries: Sequence[Dict[str, Any]],
    all_transitions_by_horizon: Dict[int, List[HistoricalTransition]],
    m_draws: int = EVALUATION_DRAWS_COUNT,
    base_seed: int = BASE_RANDOM_SEED,
) -> Tuple[List[CaseEvaluationResult], pd.DataFrame]:
    """Execute comparative 3-arm rolling backtest over all manifest cases."""
    pop_by_date: Dict[date, Dict[str, float]] = {row["date"]: row["composition"] for row in pop_timeseries}

    # Group manifest by origin date to avoid re-estimating OpinionState for each horizon
    manifest_by_origin: Dict[date, List[RollingCaseManifestRecord]] = {}
    for rec in manifest:
        if rec.origin_date not in manifest_by_origin:
            manifest_by_origin[rec.origin_date] = []
        manifest_by_origin[rec.origin_date].append(rec)

    results: List[CaseEvaluationResult] = []
    df_rows: List[Dict[str, Any]] = []
    skipped_cases: List[Dict[str, Any]] = []

    for o_date, o_cases in manifest_by_origin.items():
        # Precompute precision profiles as of o_date
        try:
            prec_primary = estimate_pollster_precision(o_date, individual_polls, pop_by_date, m0_prior=M0_PRIMARY)
            prec_sens = estimate_pollster_precision(o_date, individual_polls, pop_by_date, m0_prior=M0_SENSITIVITY)
        except Exception as err:
            for case in o_cases:
                skipped_cases.append({
                    "origin_date": o_date.isoformat(),
                    "horizon_days": case.horizon_days,
                    "target_date": case.target_date.isoformat(),
                    "reason": f"precision_profile_failed: {type(err).__name__}: {err}",
                })
            continue

        # Estimate OpinionState for each arm
        try:
            # Arm A is the production estimator.  Early historical origins
            # can legitimately fail its minimum residual-data requirement;
            # such origins are recorded as skips rather than evaluated against
            # a non-RC1 fallback covariance.
            state_rc1 = estimate_opinion_with_precision_arm(
                o_date,
                individual_polls,
                pop_timeseries,
                weighting_arm="rc1_baseline",
                data_dir=POLLS_FILE.parent,
            )
        except Exception as err:
            for case in o_cases:
                skipped_cases.append({
                    "origin_date": o_date.isoformat(),
                    "horizon_days": case.horizon_days,
                    "target_date": case.target_date.isoformat(),
                    "reason": f"rc1_baseline_failed: {type(err).__name__}: {err}",
                })
            continue
        try:
            state_equal = estimate_opinion_with_precision_arm(
                o_date, individual_polls, pop_timeseries, weighting_arm="equal_weighting"
            )
            state_prec = estimate_opinion_with_precision_arm(
                o_date, individual_polls, pop_timeseries, weighting_arm="precision_challenger", precision_state=prec_primary
            )
            state_sens = estimate_opinion_with_precision_arm(
                o_date, individual_polls, pop_timeseries, weighting_arm="precision_challenger", precision_state=prec_sens
            )
        except Exception as err:
            for case in o_cases:
                skipped_cases.append({
                    "origin_date": o_date.isoformat(),
                    "horizon_days": case.horizon_days,
                    "target_date": case.target_date.isoformat(),
                    "reason": f"experiment_arm_failed: {type(err).__name__}: {err}",
                })
            continue

        for case in o_cases:
            h = case.horizon_days
            target_d = case.target_date
            actual_comp = pop_by_date[target_d]
            actual_vec_9 = np.array([actual_comp[cat] for cat in ALL_CATEGORIES_9], dtype=float)

            # Filter historical transitions strictly as of origin date
            h_transitions = all_transitions_by_horizon.get(h, [])
            cand_pool = filter_transitions_as_of(h_transitions, o_date)

            # Common random numbers for deterministic pairing across arms
            case_seed = derive_case_seed(base_seed, o_date, h)
            rng = np.random.default_rng(case_seed)

            standard_normals = rng.standard_normal((m_draws, 8))
            transition_indices = rng.integers(0, len(cand_pool), size=m_draws) if cand_pool else np.zeros(m_draws, dtype=int)
            sign_flips = rng.choice([-1.0, 1.0], size=m_draws)

            # Sample predictive vote shares for each arm
            shares_a = sample_predictive_vote_shares(state_rc1, cand_pool, standard_normals, transition_indices, sign_flips)
            shares_b = sample_predictive_vote_shares(state_equal, cand_pool, standard_normals, transition_indices, sign_flips)
            shares_c = sample_predictive_vote_shares(state_prec, cand_pool, standard_normals, transition_indices, sign_flips)
            shares_s = sample_predictive_vote_shares(state_sens, cand_pool, standard_normals, transition_indices, sign_flips)

            # 1. Energy Scores in 9-party vote-share space
            es_a = energy_score(shares_a, actual_vec_9)
            es_b = energy_score(shares_b, actual_vec_9)
            es_c = energy_score(shares_c, actual_vec_9)
            es_s = energy_score(shares_s, actual_vec_9)

            rel_imp_prec = (es_a - es_c) / max(es_a, 1e-6)
            rel_imp_eq = (es_a - es_b) / max(es_a, 1e-6)

            # 2. Marginal CRPS (mean across 8 parliamentary parties)
            crps_a_parties = [continuous_crps(shares_a[:, i], actual_vec_9[i]) for i in range(8)]
            crps_b_parties = [continuous_crps(shares_b[:, i], actual_vec_9[i]) for i in range(8)]
            crps_c_parties = [continuous_crps(shares_c[:, i], actual_vec_9[i]) for i in range(8)]
            crps_s_parties = [continuous_crps(shares_s[:, i], actual_vec_9[i]) for i in range(8)]

            crps_a = float(np.mean(crps_a_parties))
            crps_b = float(np.mean(crps_b_parties))
            crps_c = float(np.mean(crps_c_parties))
            crps_s = float(np.mean(crps_s_parties))

            rel_crps_prec = (crps_a - crps_c) / max(crps_a, 1e-6)

            # REST CRPS
            crps_rest_a = continuous_crps(shares_a[:, 8], actual_vec_9[8])
            crps_rest_c = continuous_crps(shares_c[:, 8], actual_vec_9[8])

            case_res = CaseEvaluationResult(
                origin_date=o_date,
                horizon_days=h,
                target_date=target_d,
                calendar_block_6m=case.calendar_block_6m,
                es_arm_a_rc1=round(es_a, 5),
                es_arm_b_equal=round(es_b, 5),
                es_arm_c_precision=round(es_c, 5),
                es_sens_c25=round(es_s, 5),
                rel_es_imp_precision=round(rel_imp_prec, 6),
                rel_es_imp_equal=round(rel_imp_eq, 6),
                crps_arm_a_rc1=round(crps_a, 5),
                crps_arm_b_equal=round(crps_b, 5),
                crps_arm_c_precision=round(crps_c, 5),
                crps_sens_c25=round(crps_s, 5),
                rel_crps_imp_precision=round(rel_crps_prec, 6),
                crps_rest_rc1=round(crps_rest_a, 5),
                crps_rest_precision=round(crps_rest_c, 5),
                n_eff_rc1=round(state_rc1.effective_poll_count, 2),
                n_eff_precision=round(state_prec.effective_poll_count, 2),
            )
            results.append(case_res)

            df_rows.append({
                "origin_date": o_date.isoformat(),
                "horizon_days": h,
                "target_date": target_d.isoformat(),
                "calendar_block_6m": case.calendar_block_6m,
                "es_arm_a_rc1": es_a,
                "es_arm_b_equal": es_b,
                "es_arm_c_precision": es_c,
                "es_sens_c25": es_s,
                "rel_es_improvement_precision": rel_imp_prec,
                "rel_es_improvement_equal": rel_imp_eq,
                "crps_arm_a_rc1": crps_a,
                "crps_arm_b_equal": crps_b,
                "crps_arm_c_precision": crps_c,
                "crps_sens_c25": crps_s,
                "rel_crps_improvement_precision": rel_crps_prec,
                "crps_rest_rc1": crps_rest_a,
                "crps_rest_precision": crps_rest_c,
                "n_eff_rc1": state_rc1.effective_poll_count,
                "n_eff_precision": state_prec.effective_poll_count,
            })

    df_cases = pd.DataFrame(df_rows)
    df_cases.attrs["skipped_cases"] = skipped_cases
    return results, df_cases


def run_calendar_block_bootstrap(
    df_cases: pd.DataFrame,
    n_replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BASE_RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute paired 6-month calendar-block bootstrap on relative Energy Score and CRPS improvements."""
    blocks = sorted(df_cases["calendar_block_6m"].unique())
    n_blocks = len(blocks)
    df_by_block = {b: df_cases[df_cases["calendar_block_6m"] == b] for b in blocks}

    rng = np.random.default_rng(seed)

    boot_rel_es_imp: List[float] = []
    boot_rel_crps_imp: List[float] = []

    for _ in range(n_replications):
        sampled_blocks = rng.choice(blocks, size=n_blocks, replace=True)
        sampled_dfs = [df_by_block[b] for b in sampled_blocks]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)

        mean_es_a = boot_df["es_arm_a_rc1"].mean()
        mean_es_c = boot_df["es_arm_c_precision"].mean()
        rel_es = (mean_es_a - mean_es_c) / max(mean_es_a, 1e-6)
        boot_rel_es_imp.append(rel_es)

        mean_crps_a = boot_df["crps_arm_a_rc1"].mean()
        mean_crps_c = boot_df["crps_arm_c_precision"].mean()
        rel_crps = (mean_crps_a - mean_crps_c) / max(mean_crps_a, 1e-6)
        boot_rel_crps_imp.append(rel_crps)

    es_arr = np.array(boot_rel_es_imp)
    crps_arr = np.array(boot_rel_crps_imp)

    point_rel_es = (df_cases["es_arm_a_rc1"].mean() - df_cases["es_arm_c_precision"].mean()) / df_cases["es_arm_a_rc1"].mean()
    point_rel_crps = (df_cases["crps_arm_a_rc1"].mean() - df_cases["crps_arm_c_precision"].mean()) / df_cases["crps_arm_a_rc1"].mean()

    return {
        "relative_es_improvement": {
            "point_estimate_pct": round(float(point_rel_es * 100.0), 3),
            "ci_95_pct": [round(float(np.percentile(es_arr, 2.5) * 100.0), 3), round(float(np.percentile(es_arr, 97.5) * 100.0), 3)],
            "prob_positive": round(float(np.mean(es_arr > 0)), 4),
            "n_blocks_6m": n_blocks,
        },
        "relative_crps_improvement": {
            "point_estimate_pct": round(float(point_rel_crps * 100.0), 3),
            "ci_95_pct": [round(float(np.percentile(crps_arr, 2.5) * 100.0), 3), round(float(np.percentile(crps_arr, 97.5) * 100.0), 3)],
            "prob_positive": round(float(np.mean(crps_arr > 0)), 4),
            "n_blocks_6m": n_blocks,
        },
    }
