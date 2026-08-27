"""QA assertions, reference invariance gate, and validation report generator (Experiment 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from scripts.pollofpolls.state import (
    load_individual_polls_dataset,
    load_timeseries_dataset,
)
from scripts.pollofpolls.transitions import build_all_historical_transitions

from .config import (
    DEFAULT_HORIZONS,
    GATE_MIN_HORIZONS_WON,
    GATE_MIN_RELATIVE_CRPS_IMPROVEMENT,
    GATE_MIN_RELATIVE_ES_IMPROVEMENT,
    POLLS_FILE,
    POP_TIMESERIES_FILE,
    MIN_CASES_PER_HORIZON,
    PROCESSED_DIR,
)
from .evaluation import evaluate_all_manifest_cases, run_calendar_block_bootstrap
from .guardrail import evaluate_election_guardrail
from .manifest import build_canonical_rolling_manifest
from .opinion_state import estimate_opinion_with_precision_arm
from .precision import estimate_pollster_precision


def verify_reference_invariance_hard_gate(
    target_as_of: Any,
    individual_polls: Any,
    pop_timeseries: Any,
) -> Dict[str, Any]:
    """Hard test: changing ALR reference category must leave precision multipliers q_g invariant."""
    pop_by_date = {row["date"]: row["composition"] for row in pop_timeseries}

    # Estimate precision under default reference (REST)
    prec_rest = estimate_pollster_precision(target_as_of, individual_polls, pop_by_date, categories=("M", "L", "C", "KD", "S", "V", "MP", "SD", "REST"))

    # Estimate precision under alternative reference (S)
    prec_s = estimate_pollster_precision(target_as_of, individual_polls, pop_by_date, categories=("REST", "M", "L", "C", "KD", "V", "MP", "SD", "S"))

    # Compare q_g multipliers for all houses
    max_diff_q = 0.0
    for pollster, q_val in prec_rest.precision_multipliers_q.items():
        q_alt = prec_s.precision_multipliers_q.get(pollster, 1.0)
        diff = abs(q_val - q_alt)
        if diff > max_diff_q:
            max_diff_q = diff

    passed = (max_diff_q < 1e-5)
    return {
        "reference_invariance_passed": passed,
        "max_q_difference_across_reference_bases": round(float(max_diff_q), 8),
        "tested_bases": ["REST", "S"],
    }


def determine_final_decision(
    *,
    coverage_ready: bool,
    coverage_status: str,
    score_gate_passed: bool,
    guardrail_res: Dict[str, Any],
) -> Dict[str, str]:
    """Apply the adoption decision without conflating missing data and failure.

    A score comparison is interpretable only when every requested case is
    represented and each horizon meets the minimum case count.  Incomplete
    coverage therefore produces an explicit ``NOT_EVALUATED`` decision,
    regardless of the provisional scores calculated from the available rows.
    A substantive rejection is reserved for a complete evaluation whose score
    gate (or, after it passes, election guardrail) fails.
    """
    if not coverage_ready:
        status = coverage_status if coverage_status in {"PARTIAL", "NOT_RUN"} else "PARTIAL"
        return {
            "final_decision": "PRECISION_CHALLENGER_NOT_EVALUATED_KEEP_RC1",
            "decision_summary": (
                f"The rolling evaluation is {status.lower()} because the requested case coverage or minimum "
                "per-horizon coverage was not met. Treat available scores as diagnostic only; Stage 2 election "
                "guardrail was not run, and retain RC1."
            ),
        }

    if score_gate_passed and guardrail_res.get("status") == "EVALUATED" and guardrail_res.get("guardrail_passed", False):
        return {
            "final_decision": "PRECISION_CHALLENGER_ACCEPTED",
            "decision_summary": (
                "Empirical pollster precision weighting demonstrated sufficient out-of-sample proper-score performance "
                "over RC1 and passed the complete election/seat guardrail. Candidate accepted as an experiment result."
            ),
        }

    if score_gate_passed and guardrail_res.get("status") != "EVALUATED":
        return {
            "final_decision": "PRECISION_CHALLENGER_NOT_EVALUATED_KEEP_RC1",
            "decision_summary": (
                "The rolling score gate passed, but the complete election/seat guardrail was not evaluated "
                "successfully. The candidate is not eligible for adoption; retain RC1 and report the guardrail "
                "status explicitly."
            ),
        }

    return {
        "final_decision": "PRECISION_CHALLENGER_REJECTED_KEEP_RC1",
        "decision_summary": (
            "Empirical pollster precision weighting does not demonstrate sufficient out-of-sample predictive improvement "
            "over RC1 baseline to justify model changes. Retain RC1; this experiment does not establish that all future "
            "candidate model families are exhausted."
        ),
    }


def run_full_opinion_precision_qa(
    processed_dir: Path = PROCESSED_DIR,
    output_report_file: Optional[Path] = None,
    origin_step_days: int = 7,
    m_draws: int = 1000,
    n_bootstrap_replications: int = 2000,
) -> Dict[str, Any]:
    """Execute complete Step 2 OpinionState precision comparative evaluation and gate decision."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    if output_report_file is None:
        output_report_file = processed_dir / "precision_validation_report.json"

    # 1. Load canonical polling data & PoP time series
    print("Loading canonical reconstructed polls and PoP series...")
    individual_polls, _ = load_individual_polls_dataset(POLLS_FILE)
    pop_timeseries = load_timeseries_dataset(POP_TIMESERIES_FILE)
    all_transitions = build_all_historical_transitions(pop_timeseries, horizons=DEFAULT_HORIZONS)

    # 2. Reference Invariance Hard Gate Verification
    print("Verifying Reference-Invariance Hard Gate...")
    sample_origin = pop_timeseries[-10]["date"]
    ref_inv_res = verify_reference_invariance_hard_gate(sample_origin, individual_polls, pop_timeseries)
    if not ref_inv_res["reference_invariance_passed"]:
        raise RuntimeError(f"FATAL: Reference-invariance gate failed! Diff: {ref_inv_res['max_q_difference_across_reference_bases']}")

    # 3. Build Canonical Manifest
    print("Building canonical rolling backtest case manifest...")
    manifest = build_canonical_rolling_manifest(
        pop_file=POP_TIMESERIES_FILE,
        horizons=DEFAULT_HORIZONS,
        step_days=origin_step_days,
    )
    print(f"Generated {len(manifest)} canonical rolling evaluation cases across {len(set(r.origin_date for r in manifest))} origins.")

    # 4. Evaluate all cases across Arms A, B, C, C25
    print(f"Evaluating {len(manifest)} cases with M={m_draws} draws and common random numbers...")
    results, df_cases = evaluate_all_manifest_cases(
        manifest=manifest,
        individual_polls=individual_polls,
        pop_timeseries=pop_timeseries,
        all_transitions_by_horizon=all_transitions,
        m_draws=m_draws,
    )
    skipped_cases = list(df_cases.attrs.get("skipped_cases", []))
    horizon_case_counts = {
        h: int((df_cases["horizon_days"] == h).sum()) if not df_cases.empty else 0
        for h in DEFAULT_HORIZONS
    }
    complete_case_coverage = len(skipped_cases) == 0 and len(df_cases) == len(manifest)
    adequate_horizon_coverage = all(count >= MIN_CASES_PER_HORIZON for count in horizon_case_counts.values())
    coverage_ready = complete_case_coverage and adequate_horizon_coverage
    if coverage_ready:
        coverage_status = "COMPLETE"
    elif len(df_cases) == 0:
        coverage_status = "NOT_RUN"
    else:
        coverage_status = "PARTIAL"

    cases_csv = processed_dir / "precision_challenger_predictive_evaluation.csv"
    df_cases.to_csv(cases_csv, index=False)

    # 5. Calendar-Block Bootstrap Inference
    print(f"Running 6-month calendar-block bootstrap ({n_bootstrap_replications} reps)...")
    bootstrap_res = run_calendar_block_bootstrap(
        df_cases, n_replications=n_bootstrap_replications
    )

    # 6. Horizon-by-Horizon Summaries
    horizon_summaries: Dict[str, Any] = {}
    horizons_won_on_es = 0

    for h in DEFAULT_HORIZONS:
        sub_df = df_cases[df_cases["horizon_days"] == h]
        n_c = len(sub_df)
        es_rc1 = float(sub_df["es_arm_a_rc1"].mean())
        es_equal = float(sub_df["es_arm_b_equal"].mean())
        es_prec = float(sub_df["es_arm_c_precision"].mean())
        es_sens = float(sub_df["es_sens_c25"].mean())

        rel_es_imp = float((es_rc1 - es_prec) / max(es_rc1, 1e-6))
        prec_beats_rc1 = (es_prec < es_rc1)
        if prec_beats_rc1:
            horizons_won_on_es += 1

        crps_rc1 = float(sub_df["crps_arm_a_rc1"].mean())
        crps_equal = float(sub_df["crps_arm_b_equal"].mean())
        crps_prec = float(sub_df["crps_arm_c_precision"].mean())

        horizon_summaries[f"h_{h}"] = {
            "n_cases": n_c,
            "es_arm_a_rc1": round(es_rc1, 5),
            "es_arm_b_equal": round(es_equal, 5),
            "es_arm_c_precision": round(es_prec, 5),
            "es_sens_c25": round(es_sens, 5),
            "relative_es_improvement_pct": round(rel_es_imp * 100.0, 3),
            "precision_beats_rc1_on_es": prec_beats_rc1,
            "crps_arm_a_rc1": round(crps_rc1, 5),
            "crps_arm_b_equal": round(crps_equal, 5),
            "crps_arm_c_precision": round(crps_prec, 5),
        }

    # 7. Pooled Scores Across All Cases
    pooled_es_rc1 = float(df_cases["es_arm_a_rc1"].mean())
    pooled_es_equal = float(df_cases["es_arm_b_equal"].mean())
    pooled_es_prec = float(df_cases["es_arm_c_precision"].mean())
    pooled_es_sens = float(df_cases["es_sens_c25"].mean())

    pooled_rel_es_imp = (pooled_es_rc1 - pooled_es_prec) / max(pooled_es_rc1, 1e-6)

    pooled_crps_rc1 = float(df_cases["crps_arm_a_rc1"].mean())
    pooled_crps_equal = float(df_cases["crps_arm_b_equal"].mean())
    pooled_crps_prec = float(df_cases["crps_arm_c_precision"].mean())

    pooled_rel_crps_imp = (pooled_crps_rc1 - pooled_crps_prec) / max(pooled_crps_rc1, 1e-6)

    # 8. Test Stage 1 Rolling Decision Gate
    es_ci_low = bootstrap_res["relative_es_improvement"]["ci_95_pct"][0]
    check1_material_es = (pooled_rel_es_imp >= GATE_MIN_RELATIVE_ES_IMPROVEMENT)
    check2_no_crps_deg = (pooled_rel_crps_imp >= GATE_MIN_RELATIVE_CRPS_IMPROVEMENT)
    check3_multi_horizon = (horizons_won_on_es >= GATE_MIN_HORIZONS_WON)
    check4_block_ci = (es_ci_low > 0.0)
    check5_sens_consistent = (pooled_es_sens <= pooled_es_rc1)

    score_gate_passed = (
        check1_material_es
        and check2_no_crps_deg
        and check3_multi_horizon
        and check4_block_ci
        and check5_sens_consistent
    )
    rolling_gate_passed = score_gate_passed and coverage_ready

    # 9. Test Stage 2 Election Guardrail
    guardrail_res = evaluate_election_guardrail(rolling_gate_passed)
    decision = determine_final_decision(
        coverage_ready=coverage_ready,
        coverage_status=coverage_status,
        score_gate_passed=score_gate_passed,
        guardrail_res=guardrail_res,
    )
    final_decision = decision["final_decision"]
    decision_summary = decision["decision_summary"]

    report: Dict[str, Any] = {
        "metadata": {
            "experiment": "Experiment 2: Empirical Pollster Precision Challenger (OpinionState v1.2-candidate)",
            "as_of_latest_date": pop_timeseries[-1]["date"].isoformat(),
            "n_requested_cases": len(manifest),
            "n_evaluated_cases": len(df_cases),
            "n_skipped_cases": len(skipped_cases),
            "n_evaluated_origins": len(set(df_cases["origin_date"])) if not df_cases.empty else 0,
            "horizons": list(DEFAULT_HORIZONS),
            "coverage_status": coverage_status,
        },
        "experiment_0_and_1_status": {
            "status": "ALREADY_REJECTED",
            "audit_document": "docs/industry_bias_audit.md",
        },
        "reference_invariance_hard_gate": ref_inv_res,
        "rolling_decision_gate": {
            "rolling_gate_passed": rolling_gate_passed,
            "score_gate_passed": score_gate_passed,
            "coverage_ready": coverage_ready,
            "coverage_status": coverage_status,
            "gate_checks": {
                "check1_material_relative_es_improvement": check1_material_es,
                "check2_no_crps_degradation": check2_no_crps_deg,
                "check3_multi_horizon_consistency": check3_multi_horizon,
                "check4_calendar_block_ci_positive": check4_block_ci,
                "check5_sensitivity_consistent": check5_sens_consistent,
                "check6_complete_case_coverage": complete_case_coverage,
                "check7_minimum_horizon_coverage": adequate_horizon_coverage,
            },
            "horizons_won_on_es": f"{horizons_won_on_es} of {len(DEFAULT_HORIZONS)}",
            "requested_case_count": len(manifest),
            "evaluated_case_count": len(df_cases),
            "skipped_case_count": len(skipped_cases),
            "skipped_cases": skipped_cases,
            "horizon_case_counts": horizon_case_counts,
        },
        "stage_2_election_guardrail": guardrail_res,
        "final_decision": final_decision,
        "decision_summary": decision_summary,
        "model_search_closed": False,
        "search_scope": "Only the configured pollster-precision candidate and its stated controls were evaluated.",
        "pooled_scores": {
            "arm_a_rc1": {"energy_score": round(pooled_es_rc1, 5), "crps": round(pooled_crps_rc1, 5)},
            "arm_b_equal": {"energy_score": round(pooled_es_equal, 5), "crps": round(pooled_crps_equal, 5)},
            "arm_c_precision": {"energy_score": round(pooled_es_prec, 5), "crps": round(pooled_crps_prec, 5)},
            "sensitivity_c25": {"energy_score": round(pooled_es_sens, 5)},
            "relative_es_improvement_pct": round(pooled_rel_es_imp * 100.0, 3),
            "relative_crps_improvement_pct": round(pooled_rel_crps_imp * 100.0, 3),
        },
        "calendar_block_bootstrap_6m": bootstrap_res,
        "horizon_summaries": horizon_summaries,
    }

    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
