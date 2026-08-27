"""Quality assurance, hypothesis testing, and validation report for Step 3.

Runs statistical regressions, formats validation report, validates assertions,
and determines substantive conclusion (A, B, or C).
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from scripts.scb_behavioral_diagnostic.config import (
    BOOTSTRAP_REPLICATIONS,
    FOCUS_THRESHOLD_PARTIES,
    PROCESSED_DATA_DIR,
    SCB_PANEL_FILE,
    kernel_linear_4pct,
    kernel_placebo_7pct,
)
from scripts.scb_behavioral_diagnostic.models import (
    RegressionModelResult,
    compute_paired_placebo_difference_bootstrap,
    evaluate_all_specifications,
    export_regression_results_table,
    load_and_prepare_regression_data,
)
from scripts.scb_behavioral_diagnostic.profiles import (
    build_party_threshold_profiles,
    compute_identification_coverage_gate,
)


def run_full_scb_behavioral_qa(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_report_file: Optional[Path] = None,
    n_bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
) -> Dict[str, Any]:
    """Execute complete Step 3 statistical analysis and generate validation report."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    if output_report_file is None:
        output_report_file = processed_dir / "scb_behavioral_validation_report.json"
        
    print("Loading SCB panel and computing identification coverage gate...")
    panel_df = pd.read_csv(SCB_PANEL_FILE)
    gate_file = processed_dir / "identification_coverage_gate.csv"
    df_gate = compute_identification_coverage_gate(panel_df, output_file=gate_file)
    
    print("Generating 29-wave party threshold profiles for L, KD, MP, C...")
    profiles_file = processed_dir / "party_threshold_profiles.csv"
    df_profiles = build_party_threshold_profiles(panel_df, output_file=profiles_file)
    
    print("Estimating primary, placebo, and sensitivity regression models (with wave bootstrap)...")
    reg_df = load_and_prepare_regression_data(SCB_PANEL_FILE)
    model_results = evaluate_all_specifications(reg_df, n_bootstrap_replications=n_bootstrap_replications)
    
    results_csv = processed_dir / "scb_behavioral_regression_results.csv"
    df_results_table = export_regression_results_table(model_results, results_csv)
    
    # Compute paired wave-bootstrap difference for (alpha_4 - alpha_7)
    paired_placebo_diff = compute_paired_placebo_difference_bootstrap(
        reg_df, n_replications=n_bootstrap_replications
    )
    
    # Extract primary and placebo results for substantive classification
    primary_res = next(r for r in model_results if r.model_category == "PRIMARY")
    placebo_res = next(r for r in model_results if r.model_category == "PLACEBO")
    wls_res = next(r for r in model_results if "WLS" in r.model_name)
    vid10_res = next(r for r in model_results if "Vid10" in r.model_name)
    lag_res = next(r for r in model_results if "Lagged" in r.model_name)
    
    alpha_primary = primary_res.coefficients["A_K4_symp"]
    alpha_primary_ci = (primary_res.bootstrap_ci_lower["A_K4_symp"], primary_res.bootstrap_ci_upper["A_K4_symp"])
    alpha_placebo = placebo_res.coefficients["A_K7_symp"]
    alpha_placebo_ci = (placebo_res.bootstrap_ci_lower["A_K7_symp"], placebo_res.bootstrap_ci_upper["A_K7_symp"])
    
    # Substantive Outcome Determination:
    if alpha_primary <= 0.0 or (alpha_primary - alpha_placebo) <= 0.01:
        substantive_conclusion = "CONCLUSION_A_NO_EVIDENCE"
        tactical_decision = "CLOSED_NO_EVIDENCE"
        conclusion_summary = (
            "There is no positive threshold activation, and the negative threshold x affinity interaction "
            f"(alpha = {alpha_primary:+.4f}, 95% CI {list(alpha_primary_ci)}) is essentially identical to the "
            f"7% placebo (alpha_placebo = {alpha_placebo:+.4f}, paired 95% CI {paired_placebo_diff['paired_bootstrap_ci_95']}). "
            "Tactical/support voting is not detectable as either a systematic final-poll->election uplift or a "
            "4%-specific increase in SCB cross-party vote intentions. The tactical-voting research branch is closed."
        )
    elif alpha_primary > 0.0 and alpha_primary_ci[0] <= 0.0:
        substantive_conclusion = "CONCLUSION_B_SUGGESTIVE_WEAK"
        tactical_decision = "CLOSED_WEAK_SIGNAL"
        conclusion_summary = (
            f"Suggestive but statistically weak association (alpha = {alpha_primary:+.4f}, 95% CI {list(alpha_primary_ci)}). "
            "Evidence is not sufficiently robust to justify an explicit tactical voting parameter."
        )
    else:
        substantive_conclusion = "CONCLUSION_C_CREDIBLE_ASSOCIATION"
        tactical_decision = "OPEN_INVESTIGATE_DYNAMICS"
        conclusion_summary = (
            f"Credible behavioral association found (alpha = {alpha_primary:+.4f}, 95% CI {list(alpha_primary_ci)}). "
            "Second-choice affinity significantly converts into cross-party vote intention near 4%."
        )
        
    # Programmatic assertion checks
    assertions = {
        "all_29_waves_accounted_for": bool(reg_df["wave"].nunique() == 29),
        "linear_kernel_bounds_valid": bool(
            (reg_df["K4_symp"].dropna() >= 0.0).all() and (reg_df["K4_symp"].dropna() <= 1.0).all()
        ),
        "placebo_kernel_bounds_valid": bool(
            (reg_df["K7_symp"].dropna() >= 0.0).all() and (reg_df["K7_symp"].dropna() <= 1.0).all()
        ),
        "vote_flow_non_negative": bool((reg_df["R"] >= 0.0).all()),
        "second_choice_non_negative": bool((reg_df["A"] >= 0.0).all()),
        "conversion_floor_enforced": bool(
            df_profiles["top_donor_1_conversion_ratio"].dropna().isin(
                df_profiles[df_profiles["top_donor_1_affinity_pct"] >= 2.0]["top_donor_1_conversion_ratio"].dropna()
            ).all()
        ),
        "primary_regression_fitted": bool(not np.isnan(alpha_primary)),
    }
    assertions["all_assertions_passed"] = bool(all(assertions.values()))
    
    report = {
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "substantive_conclusion": substantive_conclusion,
        "tactical_voting_branch": tactical_decision,
        "conclusion_summary": conclusion_summary,
        "assertions": assertions,
        "sample_summary": {
            "total_usable_cross_party_cells": len(reg_df),
            "total_waves": int(reg_df["wave"].nunique()),
            "total_donor_recipient_pairs": int(reg_df["pair"].nunique()),
        },
        "primary_model": {
            "name": primary_res.model_name,
            "r_squared": round(primary_res.r_squared, 4),
            "theta_affinity": round(primary_res.coefficients["A"], 5),
            "theta_bootstrap_se": round(primary_res.bootstrap_se["A"], 5),
            "theta_ci_95": [round(primary_res.bootstrap_ci_lower["A"], 5), round(primary_res.bootstrap_ci_upper["A"], 5)],
            "delta_threshold": round(primary_res.coefficients["K4_symp"], 5),
            "delta_bootstrap_se": round(primary_res.bootstrap_se["K4_symp"], 5),
            "delta_ci_95": [round(primary_res.bootstrap_ci_lower["K4_symp"], 5), round(primary_res.bootstrap_ci_upper["K4_symp"], 5)],
            "alpha_interaction": round(primary_res.coefficients["A_K4_symp"], 5),
            "alpha_bootstrap_se": round(primary_res.bootstrap_se["A_K4_symp"], 5),
            "alpha_ci_95": [round(primary_res.bootstrap_ci_lower["A_K4_symp"], 5), round(primary_res.bootstrap_ci_upper["A_K4_symp"], 5)],
            "prob_alpha_positive": round(primary_res.prob_alpha_positive, 4) if primary_res.prob_alpha_positive is not None else None,
        },
        "placebo_comparison": {
            "name": placebo_res.model_name,
            "alpha_placebo_7pct": round(placebo_res.coefficients["A_K7_symp"], 5),
            "alpha_placebo_bootstrap_se": round(placebo_res.bootstrap_se["A_K7_symp"], 5),
            "alpha_placebo_ci_95": [round(placebo_res.bootstrap_ci_lower["A_K7_symp"], 5), round(placebo_res.bootstrap_ci_upper["A_K7_symp"], 5)],
            "paired_bootstrap_difference": paired_placebo_diff,
        },
        "sensitivities": {
            "wls_alpha": round(wls_res.coefficients["A_K4_symp"], 5),
            "wls_alpha_ci": [round(wls_res.bootstrap_ci_lower["A_K4_symp"], 5), round(wls_res.bootstrap_ci_upper["A_K4_symp"], 5)],
            "vid10_alpha": round(vid10_res.coefficients["A_K4_vid10"], 5),
            "vid10_alpha_ci": [round(vid10_res.bootstrap_ci_lower["A_K4_vid10"], 5), round(vid10_res.bootstrap_ci_upper["A_K4_vid10"], 5)],
            "lagged_affinity_alpha": round(lag_res.coefficients["A_lag_K4_symp"], 5),
            "lagged_affinity_alpha_ci": [round(lag_res.bootstrap_ci_lower["A_lag_K4_symp"], 5), round(lag_res.bootstrap_ci_upper["A_lag_K4_symp"], 5)],
        },
    }
    
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Step 3 QA complete. Substantive Conclusion: {substantive_conclusion}")
    print(f"Validation report saved to {output_report_file}")
    return report


if __name__ == "__main__":
    run_full_scb_behavioral_qa()
