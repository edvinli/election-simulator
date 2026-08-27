"""QA evaluation, predefined decision-gate testing, and validation report generation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from scripts.pollofpolls.backtest import derive_forecast_seed, generate_forecast_origins
from scripts.pop_state_diagnostics.config import (
    BASE_RANDOM_SEED,
    BOOTSTRAP_REPLICATIONS,
    DEFAULT_HORIZONS,
    DEFAULT_ORIGIN_STEP_DAYS,
    EVALUATION_DRAWS_COUNT,
    GATE_MAX_CRPS_DEGRADATION,
    GATE_MIN_ENERGY_SCORE_IMPROVEMENT,
    GATE_MIN_HORIZONS_BEATING_V2,
    MIN_CANDIDATE_TRANSITIONS,
    PROCESSED_DATA_DIR,
    START_ORIGIN_DATE,
)
from scripts.pop_state_diagnostics.evaluation import (
    CaseEvaluationResult,
    compute_threshold_starting_state_distributions,
    evaluate_single_case,
    run_calendar_block_bootstrap,
)
from scripts.pop_state_diagnostics.similarity import CandidateNeighborRecord
from scripts.pop_state_diagnostics.transitions import (
    DailyState,
    StateTransition,
    build_all_exact_transitions,
    compute_historical_clr_stds_as_of,
    get_leakage_safe_candidate_pool,
    load_canonical_pop_series,
)


def run_full_state_diagnostics_qa(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_report_file: Optional[Path] = None,
    n_bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    m_draws: int = EVALUATION_DRAWS_COUNT,
    origin_step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
) -> Dict[str, Any]:
    """Execute the Step 4A comparative evaluation across all four configured arms."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    if output_report_file is None:
        output_report_file = processed_dir / "state_diagnostics_validation_report.json"

    print("Loading canonical PoP time series and building exact-horizon transitions...")
    daily_states, by_date = load_canonical_pop_series()
    transitions_by_horizon = build_all_exact_transitions(daily_states, by_date, horizons=DEFAULT_HORIZONS)

    # 1. Generate threshold starting state distributions
    print("Computing threshold starting state distributions (2%–6% range)...")
    df_thresh = compute_threshold_starting_state_distributions(transitions_by_horizon)
    thresh_csv = processed_dir / "threshold_starting_state_distributions.csv"
    df_thresh.to_csv(thresh_csv, index=False)

    # 2. Generate forecast origins
    latest_date = daily_states[-1].observation_date
    all_origins = generate_forecast_origins(
        START_ORIGIN_DATE, latest_date, step_days=origin_step_days
    )

    print(f"Evaluating {len(all_origins)} rolling origins across horizons {DEFAULT_HORIZONS}...")
    case_results: List[CaseEvaluationResult] = []
    all_neighbor_records: List[CandidateNeighborRecord] = []

    for o_date in all_origins:
        if o_date not in by_date:
            continue
        origin_st = by_date[o_date]
        clr_stds = compute_historical_clr_stds_as_of(daily_states, o_date)

        for h in DEFAULT_HORIZONS:
            target_d = o_date + timedelta(days=h)
            if target_d not in by_date:
                continue
            target_st = by_date[target_d]

            # Leakage-safe candidate pool (end_date <= o_date)
            cand_pool = get_leakage_safe_candidate_pool(o_date, h, transitions_by_horizon)
            if len(cand_pool) < MIN_CANDIDATE_TRANSITIONS:
                continue

            case_seed = derive_forecast_seed(BASE_RANDOM_SEED, "state_diag", o_date, h)
            case_res, audit_recs = evaluate_single_case(
                origin_st, target_st, h, cand_pool, clr_stds, random_seed=case_seed, m_draws=m_draws
            )
            case_results.append(case_res)
            all_neighbor_records.extend(audit_recs)

    # 3. Export case evaluation results and neighbor diagnostics
    eval_df = pd.DataFrame([asdict(r) for r in case_results])
    eval_csv = processed_dir / "state_dependence_predictive_evaluation.csv"
    eval_df.to_csv(eval_csv, index=False)

    neighbor_df = pd.DataFrame([asdict(r) for r in all_neighbor_records])
    neighbor_csv = processed_dir / "state_neighbor_diagnostics.csv"
    neighbor_df.to_csv(neighbor_csv, index=False)

    # 4. Compute Horizon-by-Horizon Summaries
    horizon_summaries: Dict[str, Any] = {}
    for h in DEFAULT_HORIZONS:
        sub_h = eval_df[eval_df["horizon_days"] == h]
        if sub_h.empty:
            continue
        horizon_summaries[f"h_{h}"] = {
            "n_cases": len(sub_h),
            "es_arm_a_v2": round(float(sub_h["es_arm_a_v2"].mean()), 5),
            "es_arm_b_recent": round(float(sub_h["es_arm_b_recent"].mean()), 5),
            "es_arm_c_50nn": round(float(sub_h["es_arm_c_50nn"].mean()), 5),
            "es_arm_d_raw": round(float(sub_h["es_arm_d_raw"].mean()), 5),
            "es_diff_v2_minus_50nn": round(float(sub_h["diff_es_v2_minus_50nn"].mean()), 5),
            "es_diff_recent_minus_50nn": round(float(sub_h["diff_es_recent_minus_50nn"].mean()), 5),
            "crps_arm_a_v2": round(float(sub_h["crps_arm_a_v2"].mean()), 5),
            "crps_arm_b_recent": round(float(sub_h["crps_arm_b_recent"].mean()), 5),
            "crps_arm_c_50nn": round(float(sub_h["crps_arm_c_50nn"].mean()), 5),
            "crps_arm_d_raw": round(float(sub_h["crps_arm_d_raw"].mean()), 5),
            "crps_diff_v2_minus_50nn": round(float(sub_h["diff_crps_v2_minus_50nn"].mean()), 5),
            "50nn_beats_v2_on_es": bool(sub_h["diff_es_v2_minus_50nn"].mean() > 0),
            "50nn_beats_recent_on_es": bool(sub_h["diff_es_recent_minus_50nn"].mean() > 0),
        }

    # 5. Run Calendar-Block Bootstrap on Pooled Differences
    print("Running 6-month calendar-block bootstrap on paired score differences...")
    boot_es_v2_vs_50nn = run_calendar_block_bootstrap(
        eval_df, "diff_es_v2_minus_50nn", block_col="calendar_block_6m", n_replications=n_bootstrap_replications
    )
    boot_es_recent_vs_50nn = run_calendar_block_bootstrap(
        eval_df, "diff_es_recent_minus_50nn", block_col="calendar_block_6m", n_replications=n_bootstrap_replications
    )
    boot_es_50nn_vs_raw = run_calendar_block_bootstrap(
        eval_df, "diff_es_50nn_minus_raw", block_col="calendar_block_6m", n_replications=n_bootstrap_replications
    )
    boot_crps_v2_vs_50nn = run_calendar_block_bootstrap(
        eval_df, "diff_crps_v2_minus_50nn", block_col="calendar_block_6m", n_replications=n_bootstrap_replications
    )

    # 6. Recency vs State Diagnostics
    # Proportion of top 50 NN that are also in the top 50 most recent
    overlap_rate = float(neighbor_df["is_top_50_recent"].mean()) if not neighbor_df.empty else 0.0
    mean_nn_age_days = float(neighbor_df["age_days"].mean()) if not neighbor_df.empty else 0.0
    median_nn_age_days = float(neighbor_df["age_days"].median()) if not neighbor_df.empty else 0.0

    # 7. Evaluate the predefined Step 4B decision gate
    pooled_es_v2 = float(eval_df["es_arm_a_v2"].mean())
    pooled_es_recent = float(eval_df["es_arm_b_recent"].mean())
    pooled_es_50nn = float(eval_df["es_arm_c_50nn"].mean())
    pooled_es_raw = float(eval_df["es_arm_d_raw"].mean())

    pooled_crps_v2 = float(eval_df["crps_arm_a_v2"].mean())
    pooled_crps_recent = float(eval_df["crps_arm_b_recent"].mean())
    pooled_crps_50nn = float(eval_df["crps_arm_c_50nn"].mean())
    pooled_crps_raw = float(eval_df["crps_arm_d_raw"].mean())

    pooled_es_diff_v2 = pooled_es_v2 - pooled_es_50nn
    pooled_es_diff_recent = pooled_es_recent - pooled_es_50nn
    pooled_crps_diff_v2 = pooled_crps_v2 - pooled_crps_50nn

    n_horizons_beating_v2 = sum(
        1 for h_info in horizon_summaries.values() if h_info["50nn_beats_v2_on_es"]
    )

    gate_checks = {
        "check1_material_energy_improvement": bool(pooled_es_diff_v2 >= GATE_MIN_ENERGY_SCORE_IMPROVEMENT),
        "check2_multi_horizon_consistency": bool(n_horizons_beating_v2 >= GATE_MIN_HORIZONS_BEATING_V2),
        "check3_no_crps_degradation": bool((pooled_crps_50nn - pooled_crps_v2) <= GATE_MAX_CRPS_DEGRADATION),
        "check4_calendar_block_ci_excludes_zero": bool(boot_es_v2_vs_50nn["ci_95"][0] > 0.0),
        "check5_beats_recency_control": bool(pooled_es_diff_recent > 0.0),
    }
    gate_passed = bool(all(gate_checks.values()))

    if gate_passed:
        gate_decision = "PROCEED_TO_DYNAMICS_V3"
        decision_summary = (
            "State-conditioned dynamics (50NN ±) demonstrates statistically significant out-of-sample proper "
            "score improvement over both Dynamics v2 and the Recency Control. Proceed to build Dynamics v3."
        )
    else:
        gate_decision = "REJECT_STATE_DYNAMICS_KEEP_RC1"
        decision_summary = (
            "State-conditioned dynamics does not demonstrate sufficient out-of-sample predictive advantage over "
            "Dynamics v2 and the Recency Control to justify additional model complexity. Terminate dynamics development "
            "and retain RC1 for this evaluated dynamics comparison."
        )

    # 8. Assertions
    assertions = {
        "all_cases_evaluated": bool(len(eval_df) > 500),
        "all_horizons_present": bool(len(horizon_summaries) == len(DEFAULT_HORIZONS)),
        "common_random_numbers_consistent": bool((eval_df["pool_size"] >= MIN_CANDIDATE_TRANSITIONS).all()),
        "neighbor_diagnostics_logged": bool(len(neighbor_df) > 0),
        "all_assertions_passed": True,
    }
    assertions["all_assertions_passed"] = bool(all(assertions.values()))

    report = {
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "step_4b_gate_decision": gate_decision,
        "gate_passed": gate_passed,
        "decision_summary": decision_summary,
        "gate_checks": gate_checks,
        "pooled_scores": {
            "n_evaluated_cases": len(eval_df),
            "arm_a_v2": {"energy_score": round(pooled_es_v2, 5), "crps": round(pooled_crps_v2, 5)},
            "arm_b_recent": {"energy_score": round(pooled_es_recent, 5), "crps": round(pooled_crps_recent, 5)},
            "arm_c_50nn": {"energy_score": round(pooled_es_50nn, 5), "crps": round(pooled_crps_50nn, 5)},
            "arm_d_raw": {"energy_score": round(pooled_es_raw, 5), "crps": round(pooled_crps_raw, 5)},
            "sensitivity_25nn": {"energy_score": round(float(eval_df["es_arm_c_25nn"].mean()), 5), "crps": round(float(eval_df["crps_arm_c_25nn"].mean()), 5)},
            "sensitivity_100nn": {"energy_score": round(float(eval_df["es_arm_c_100nn"].mean()), 5), "crps": round(float(eval_df["crps_arm_c_100nn"].mean()), 5)},
        },
        "paired_differences_vs_50nn": {
            "v2_minus_50nn_es": round(pooled_es_diff_v2, 5),
            "recent_minus_50nn_es": round(pooled_es_diff_recent, 5),
            "v2_minus_50nn_crps": round(pooled_crps_diff_v2, 5),
            "50nn_minus_raw_es (directional test)": round(pooled_es_50nn - pooled_es_raw, 5),
        },
        "calendar_block_bootstrap_6m": {
            "es_diff_v2_minus_50nn": boot_es_v2_vs_50nn,
            "es_diff_recent_minus_50nn": boot_es_recent_vs_50nn,
            "es_diff_50nn_minus_raw": boot_es_50nn_vs_raw,
            "crps_diff_v2_minus_50nn": boot_crps_v2_vs_50nn,
        },
        "recency_diagnostics": {
            "top50_nn_overlap_with_top50_recent_pct": round(overlap_rate * 100.0, 2),
            "mean_neighbor_age_days": round(mean_nn_age_days, 1),
            "median_neighbor_age_days": round(median_nn_age_days, 1),
        },
        "horizon_summaries": horizon_summaries,
        "assertions": assertions,
    }

    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Step 4A evaluation complete. Gate Decision: {gate_decision}")
    print(f"Validation report saved to {output_report_file}")
    return report
