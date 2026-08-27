"""Reconciliation and QA diagnostics module for SCB PSU support-voting dataset.

Performs row-sum validation, overall reconciliation comparisons, coverage checks,
and integrity assertions, generating a machine-readable validation report.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from scripts.scb_support_voting.config import (
    METHODOLOGY_METADATA,
    PARLIAMENTARY_PARTIES,
    PROCESSED_DATA_DIR,
    WAVES_2010_2026,
)


def run_row_sum_diagnostics(df: pd.DataFrame, dest_col: str) -> Dict[str, Any]:
    """Calculate row sum diagnostics for conditional tables across (wave x donor)."""
    rows_summary = []
    max_dev = 0.0
    total_suppressed = 0
    total_observed = 0
    
    for (wave, donor_code), grp in df.groupby(["wave", "best_party_code_raw"]):
        donor_label = grp["best_party_raw"].iloc[0]
        obs_cells = grp[grp["value_status"] == "observed"]
        supp_cells = grp[grp["value_status"] == "suppressed"]
        miss_cells = grp[grp["value_status"] == "missing"]
        
        obs_sum = float(obs_cells["estimate_pct"].sum())
        supp_cnt = len(supp_cells)
        miss_cnt = len(miss_cells)
        obs_cnt = len(obs_cells)
        tot_cnt = len(grp)
        dev = round(100.0 - obs_sum, 4)
        
        total_observed += obs_cnt
        total_suppressed += supp_cnt
        if abs(dev) > max_dev:
            max_dev = abs(dev)
            
        rows_summary.append({
            "wave": wave,
            "best_party_code_raw": donor_code,
            "best_party_raw": donor_label,
            "total_cells": tot_cnt,
            "observed_cell_count": obs_cnt,
            "suppressed_cell_count": supp_cnt,
            "missing_cell_count": miss_cnt,
            "observed_sum": round(obs_sum, 2),
            "deviation_from_100": dev,
        })
        
    return {
        "total_rows_evaluated": len(rows_summary),
        "total_observed_cells": total_observed,
        "total_suppressed_cells": total_suppressed,
        "max_absolute_deviation_from_100": round(max_dev, 2),
        "row_diagnostics": rows_summary,
    }


def run_reconciliation_diagnostics(
    df_a: pd.DataFrame,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare conditional cross-tabulation and overall sympathy against published Vid10.
    
    Reconciliation is strictly diagnostic; differences are documented, never forced.
    """
    reconciliation_rows = []
    
    for wave in WAVES_2010_2026:
        # Get Table C Vid10 estimates for parliamentary parties
        vid10_wave = df_c[(df_c["wave"] == wave) & (df_c["party"].isin(PARLIAMENTARY_PARTIES))]
        vid10_map = dict(zip(vid10_wave["party"], vid10_wave["estimate_pct"]))
        
        # Get Table D Partisympati051 estimates
        psymp_wave = df_d[(df_d["wave"] == wave) & (df_d["party"].isin(PARLIAMENTARY_PARTIES))]
        psymp_map = dict(zip(psymp_wave["party"], psymp_wave["estimate_pct"]))
        
        # Get Table A 'hela väljarkåren'
        hela_wave = df_a[(df_a["wave"] == wave) & (df_a["best_party_code_raw"] == "hela väljarkåren")]
        blank_row = hela_wave[hela_wave["vote_party_code_raw"] == "blankt"]
        vetej_row = hela_wave[hela_wave["vote_party_code_raw"] == "vet ej"]
        
        blank_pct = float(blank_row["estimate_pct"].iloc[0]) if not blank_row.empty and not np.isnan(blank_row["estimate_pct"].iloc[0]) else 0.0
        vetej_pct = float(vetej_row["estimate_pct"].iloc[0]) if not vetej_row.empty and not np.isnan(vetej_row["estimate_pct"].iloc[0]) else 0.0
        decided_share = 100.0 - (blank_pct + vetej_pct)
        
        hela_party_map = {}
        for _, r in hela_wave[hela_wave["vote_party"].isin(PARLIAMENTARY_PARTIES)].iterrows():
            p = r["vote_party"]
            val = r["estimate_pct"]
            hela_party_map[p] = val
            
        # Get conditional matrix for this wave
        cond_wave = df_a[df_a["wave"] == wave]
        
        for party in PARLIAMENTARY_PARTIES:
            vid10_val = vid10_map.get(party, np.nan)
            hela_raw = hela_party_map.get(party, np.nan)
            hela_rebased = (hela_raw / decided_share * 100.0) if decided_share > 0 and not np.isnan(hela_raw) else np.nan
            
            # Weighted conditional sum across parliamentary donor parties:
            # sum_j P(vote=p | sympathy=j) * P(sympathy=j)/100
            weighted_cond_sum = 0.0
            has_suppressed = False
            for donor in PARLIAMENTARY_PARTIES:
                donor_p_share = psymp_map.get(donor, np.nan)
                transfer_row = cond_wave[
                    (cond_wave["best_party"] == donor) & (cond_wave["vote_party"] == party)
                ]
                if not transfer_row.empty:
                    t_val = transfer_row["estimate_pct"].iloc[0]
                    if np.isnan(t_val):
                        has_suppressed = True
                    elif not np.isnan(donor_p_share):
                        weighted_cond_sum += (t_val * donor_p_share / 100.0)
                        
            reconciliation_rows.append({
                "wave": wave,
                "party": party,
                "vid10_headline_pct": round(float(vid10_val), 2) if not np.isnan(vid10_val) else None,
                "hela_valjarkaren_raw_pct": round(float(hela_raw), 2) if not np.isnan(hela_raw) else None,
                "hela_valjarkaren_rebased_pct": round(float(hela_rebased), 2) if not np.isnan(hela_rebased) else None,
                "weighted_conditional_sum_pct": round(float(weighted_cond_sum), 2) if not has_suppressed else None,
                "has_suppressed_conditional_cells": has_suppressed,
                "gap_rebased_vs_vid10_pp": round(float(hela_rebased - vid10_val), 2) if not np.isnan(hela_rebased) and not np.isnan(vid10_val) else None,
            })
            
    # Summary of discrepancies
    valid_gaps = [r["gap_rebased_vs_vid10_pp"] for r in reconciliation_rows if r["gap_rebased_vs_vid10_pp"] is not None]
    
    return {
        "explanation": (
            "Compares published Vid10 (decided voters reweighted) against Table A 'hela väljarkåren' "
            "and sympathy-weighted conditional flows. Differences arise from turnout calibration weights, "
            "rounding, suppressed small cells, and response universes."
        ),
        "total_party_wave_comparisons": len(reconciliation_rows),
        "mean_absolute_gap_rebased_vs_vid10_pp": round(float(np.mean(np.abs(valid_gaps))), 3) if valid_gaps else None,
        "max_absolute_gap_rebased_vs_vid10_pp": round(float(np.max(np.abs(valid_gaps))), 3) if valid_gaps else None,
        "details": reconciliation_rows,
    }


def run_coverage_diagnostics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
) -> Dict[str, Any]:
    """Report coverage, cell statuses, and MOE availability for each wave and table."""
    wave_coverage = {}
    
    for wave in WAVES_2010_2026:
        sub_a = df_a[df_a["wave"] == wave]
        sub_b = df_b[df_b["wave"] == wave]
        sub_c = df_c[df_c["wave"] == wave]
        sub_d = df_d[df_d["wave"] == wave]
        
        wave_coverage[wave] = {
            "table_a_vote_by_sympathy": {
                "donor_categories_count": int(sub_a["best_party_code_raw"].nunique()),
                "vote_categories_count": int(sub_a["vote_party_code_raw"].nunique()),
                "total_cells": len(sub_a),
                "observed_cells": int((sub_a["value_status"] == "observed").sum()),
                "suppressed_cells": int((sub_a["value_status"] == "suppressed").sum()),
                "missing_cells": int((sub_a["value_status"] == "missing").sum()),
                "moe_available_count": int(sub_a["margin_error_pp"].notna().sum()),
            },
            "table_b_second_choice_by_sympathy": {
                "donor_categories_count": int(sub_b["best_party_code_raw"].nunique()),
                "second_choice_categories_count": int(sub_b["second_choice_code_raw"].nunique()),
                "total_cells": len(sub_b),
                "observed_cells": int((sub_b["value_status"] == "observed").sum()),
                "suppressed_cells": int((sub_b["value_status"] == "suppressed").sum()),
                "missing_cells": int((sub_b["value_status"] == "missing").sum()),
                "moe_available_count": int(sub_b["margin_error_pp"].notna().sum()),
            },
            "table_c_overall_vote_intention": {
                "party_categories_count": int(sub_c["party_code_raw"].nunique()),
                "total_cells": len(sub_c),
                "observed_cells": int((sub_c["value_status"] == "observed").sum()),
                "suppressed_cells": int((sub_c["value_status"] == "suppressed").sum()),
                "missing_cells": int((sub_c["value_status"] == "missing").sum()),
                "moe_available_count": int(sub_c["margin_error_pp"].notna().sum()),
            },
            "table_d_overall_party_sympathy": {
                "party_categories_count": int(sub_d["party_code_raw"].nunique()),
                "total_cells": len(sub_d),
                "observed_cells": int((sub_d["value_status"] == "observed").sum()),
                "suppressed_cells": int((sub_d["value_status"] == "suppressed").sum()),
                "missing_cells": int((sub_d["value_status"] == "missing").sum()),
                "moe_available_count": int(sub_d["margin_error_pp"].notna().sum()),
            },
        }
        
    return {
        "total_waves_analyzed": len(wave_coverage),
        "waves": wave_coverage,
    }


def validate_all_assertions(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
    df_panel: pd.DataFrame,
) -> Dict[str, Any]:
    """Run programmatic assertions on uniqueness, value bounds, and ordering."""
    assertions = {}
    
    # 1. No duplicates
    dup_a = df_a.duplicated(subset=["wave", "best_party_code_raw", "vote_party_code_raw"]).sum()
    dup_b = df_b.duplicated(subset=["wave", "best_party_code_raw", "second_choice_code_raw"]).sum()
    dup_c = df_c.duplicated(subset=["wave", "party_code_raw"]).sum()
    dup_d = df_d.duplicated(subset=["wave", "party_code_raw"]).sum()
    dup_panel = df_panel.duplicated(subset=["wave", "donor_party", "recipient_party"]).sum()
    
    assertions["no_duplicates"] = {
        "table_a_duplicates": int(dup_a),
        "table_b_duplicates": int(dup_b),
        "table_c_duplicates": int(dup_c),
        "table_d_duplicates": int(dup_d),
        "panel_duplicates": int(dup_panel),
        "passed": bool(dup_a == 0 and dup_b == 0 and dup_c == 0 and dup_d == 0 and dup_panel == 0),
    }
    
    # 2. Percentage bounds [0, 100] for observed values
    bounds_ok = True
    for name, df in [("A", df_a), ("B", df_b), ("C", df_c), ("D", df_d)]:
        obs = df[df["value_status"] == "observed"]["estimate_pct"]
        if obs.empty or obs.min() < 0.0 or obs.max() > 100.0 or not np.isfinite(obs).all():
            bounds_ok = False
            
    assertions["percentage_bounds_0_to_100"] = {
        "passed": bounds_ok,
    }
    
    # 3. Margins of error >= 0
    moe_ok = True
    for name, df in [("A", df_a), ("B", df_b), ("C", df_c), ("D", df_d)]:
        moe_obs = df[df["margin_error_pp"].notna()]["margin_error_pp"]
        if not moe_obs.empty and (moe_obs.min() < 0.0 or not np.isfinite(moe_obs).all()):
            moe_ok = False
            
    assertions["margin_of_error_nonnegative"] = {
        "passed": moe_ok,
    }
    
    # 4. Wave sequence and completeness
    waves_a = sorted(df_a["wave"].unique().tolist())
    waves_b = sorted(df_b["wave"].unique().tolist())
    waves_c = sorted(df_c["wave"].unique().tolist())
    waves_d = sorted(df_d["wave"].unique().tolist())
    
    waves_match = (
        waves_a == WAVES_2010_2026 and
        waves_b == WAVES_2010_2026 and
        waves_c == WAVES_2010_2026 and
        waves_d == WAVES_2010_2026
    )
    assertions["exact_29_waves_complete"] = {
        "expected_waves_count": len(WAVES_2010_2026),
        "table_a_waves_count": len(waves_a),
        "table_b_waves_count": len(waves_b),
        "table_c_waves_count": len(waves_c),
        "table_d_waves_count": len(waves_d),
        "passed": bool(waves_match),
    }
    
    # 5. Table D exact selectors used
    kon_used = set(df_d["kon_code_raw"].unique().tolist())
    alder_used = set(df_d["alder_code_raw"].unique().tolist())
    table_d_selectors_ok = (kon_used == {"TOT"} and alder_used == {"tot18+"})
    assertions["table_d_exact_selectors"] = {
        "kon_selectors_observed": list(kon_used),
        "alder_selectors_observed": list(alder_used),
        "passed": bool(table_d_selectors_ok),
    }
    
    assertions["all_assertions_passed"] = bool(all(a["passed"] for a in assertions.values()))
    return assertions


def run_all_qa(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete QA validation suite and generate validation_report.json."""
    if output_file is None:
        output_file = processed_dir / "validation_report.json"
        
    print("Loading processed datasets for QA...")
    df_a = pd.read_csv(processed_dir / "vote_by_sympathy.csv")
    df_b = pd.read_csv(processed_dir / "second_choice_by_sympathy.csv")
    df_c = pd.read_csv(processed_dir / "overall_vote_intention.csv")
    df_d = pd.read_csv(processed_dir / "overall_party_sympathy.csv")
    df_panel = pd.read_csv(processed_dir / "scb_donor_recipient_panel.csv")
    
    print("Running row-sum diagnostics on Table A...")
    row_diag_a = run_row_sum_diagnostics(df_a, "vote_party_code_raw")
    print("Running row-sum diagnostics on Table B...")
    row_diag_b = run_row_sum_diagnostics(df_b, "second_choice_code_raw")
    
    print("Running reconciliation diagnostics (Vid10 vs Conditional / Symp)...")
    reconciliation = run_reconciliation_diagnostics(df_a, df_c, df_d)
    
    print("Running coverage diagnostics across 29 waves...")
    coverage = run_coverage_diagnostics(df_a, df_b, df_c, df_d)
    
    print("Validating all assertions...")
    assertions = validate_all_assertions(df_a, df_b, df_c, df_d, df_panel)
    
    report = {
        "report_generated_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": METHODOLOGY_METADATA,
        "dataset_summary": {
            "vote_by_sympathy_rows": len(df_a),
            "second_choice_by_sympathy_rows": len(df_b),
            "overall_vote_intention_rows": len(df_c),
            "overall_party_sympathy_rows": len(df_d),
            "donor_recipient_panel_rows": len(df_panel),
            "total_waves": len(WAVES_2010_2026),
            "start_wave": WAVES_2010_2026[0],
            "end_wave": WAVES_2010_2026[-1],
        },
        "assertions": assertions,
        "row_sum_diagnostics": {
            "table_a_vote_by_sympathy": row_diag_a,
            "table_b_second_choice_by_sympathy": row_diag_b,
        },
        "reconciliation_diagnostics": reconciliation,
        "coverage_diagnostics": coverage,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"QA Validation complete. All assertions passed: {assertions['all_assertions_passed']}")
    print(f"Report saved to {output_file}")
    return report


if __name__ == "__main__":
    run_all_qa()
