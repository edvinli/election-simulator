"""Identification coverage gate and data-driven party profiles for L, KD, MP, C.

Computes pre-regression identification metrics and generates 29-wave historical
profiles with empirical top donor pools and conversion ratios.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from scripts.scb_behavioral_diagnostic.config import (
    CONVERSION_RATIO_FLOOR_PCT,
    FOCUS_THRESHOLD_PARTIES,
    PARLIAMENTARY_PARTIES,
    PROCESSED_DATA_DIR,
    SCB_PANEL_FILE,
    kernel_linear_4pct,
)


def compute_identification_coverage_gate(
    panel_df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute pre-regression identification and wave coverage statistics."""
    if panel_df is None:
        panel_df = pd.read_csv(SCB_PANEL_FILE)
        
    df_cross = panel_df[panel_df["donor_party"] != panel_df["recipient_party"]].copy()
    df_obs = df_cross[df_cross["vote_value_status"] == "observed"].copy()
    df_obs = df_obs.dropna(subset=[
        "vote_estimate_pct",
        "second_choice_estimate_pct",
        "recipient_overall_sympathy_pct",
    ]).copy()
    
    rows: List[Dict[str, Any]] = []
    
    for p in PARLIAMENTARY_PARTIES:
        p_df = df_obs[df_obs["recipient_party"] == p].copy()
        if p_df.empty:
            continue
            
        p_df["K4"] = p_df["recipient_overall_sympathy_pct"].apply(kernel_linear_4pct)
        total_waves = int(p_df["wave"].nunique())
        waves_in_k4 = int(p_df[p_df["K4"] > 0]["wave"].nunique())
        waves_out_k4 = int(p_df[p_df["K4"] == 0]["wave"].nunique())
        
        s_symp = p_df["recipient_overall_sympathy_pct"].astype(float)
        s_vid = pd.to_numeric(p_df["recipient_overall_vote_pct"], errors="coerce").dropna()
        moe_r = pd.to_numeric(p_df["vote_margin_error_pp"], errors="coerce").dropna()
        
        rows.append({
            "recipient_party": p,
            "is_focus_threshold_party": bool(p in FOCUS_THRESHOLD_PARTIES),
            "total_waves": total_waves,
            "observed_flow_cells": len(p_df),
            "waves_in_k4_danger": waves_in_k4,
            "waves_outside_k4": waves_out_k4,
            "sympathy_min_pct": round(float(s_symp.min()), 2),
            "sympathy_max_pct": round(float(s_symp.max()), 2),
            "sympathy_median_pct": round(float(s_symp.median()), 2),
            "vid10_min_pct": round(float(s_vid.min()), 2) if not s_vid.empty else np.nan,
            "vid10_max_pct": round(float(s_vid.max()), 2) if not s_vid.empty else np.nan,
            "vid10_median_pct": round(float(s_vid.median()), 2) if not s_vid.empty else np.nan,
            "median_vote_moe_pp": round(float(moe_r.median()), 2) if not moe_r.empty else np.nan,
            "usable_donor_parties_count": int(p_df["donor_party"].nunique()),
        })
        
    df_gate = pd.DataFrame(rows)
    df_gate.sort_values(by=["is_focus_threshold_party", "recipient_party"], ascending=[False, True], inplace=True)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_gate.to_csv(output_file, index=False, encoding="utf-8")
        
    return df_gate


def build_party_threshold_profiles(
    panel_df: Optional[pd.DataFrame] = None,
    output_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate 29-wave historical profiles for L, KD, MP, and C with data-driven donor pools."""
    if panel_df is None:
        panel_df = pd.read_csv(SCB_PANEL_FILE)
        
    df_cross = panel_df[panel_df["donor_party"] != panel_df["recipient_party"]].copy()
    
    profile_rows: List[Dict[str, Any]] = []
    
    for p in FOCUS_THRESHOLD_PARTIES:
        p_flows = df_cross[df_cross["recipient_party"] == p].copy()
        
        # Unique waves in chronological order
        waves_df = p_flows[["wave", "survey_date", "period"]].drop_duplicates().sort_values(by="survey_date")
        
        for _, w_row in waves_df.iterrows():
            w_code = w_row["wave"]
            s_date = w_row["survey_date"]
            period = w_row["period"]
            
            w_flows = p_flows[p_flows["wave"] == w_code].copy()
            if w_flows.empty:
                continue
                
            s_symp_raw = w_flows["recipient_overall_sympathy_pct"].iloc[0]
            s_vid_raw = w_flows["recipient_overall_vote_pct"].iloc[0]
            s_symp = float(s_symp_raw) if pd.notnull(s_symp_raw) else np.nan
            s_vid = float(s_vid_raw) if pd.notnull(s_vid_raw) else np.nan
            
            k4_val = kernel_linear_4pct(s_symp)
            dist_4 = round(s_symp - 4.0, 2) if pd.notnull(s_symp) else np.nan
            
            # Rank donor parties empirically by second-choice affinity A_jpt
            valid_donors = w_flows.dropna(subset=["second_choice_estimate_pct"]).sort_values(
                by="second_choice_estimate_pct", ascending=False
            )
            
            donor_cols: Dict[str, Any] = {}
            conv_ratios: List[float] = []
            
            for rank_idx in range(1, 4):
                if len(valid_donors) >= rank_idx:
                    d_row = valid_donors.iloc[rank_idx - 1]
                    d_party = str(d_row["donor_party"])
                    a_val = float(d_row["second_choice_estimate_pct"])
                    r_val = float(d_row["vote_estimate_pct"]) if pd.notnull(d_row["vote_estimate_pct"]) else np.nan
                    
                    # Compute conversion ratio R / A only if A >= floor (2.0%)
                    if a_val >= CONVERSION_RATIO_FLOOR_PCT and pd.notnull(r_val):
                        conv = round(r_val / a_val, 4)
                        conv_ratios.append(conv)
                    else:
                        conv = np.nan
                        
                    donor_cols[f"top_donor_{rank_idx}_party"] = d_party
                    donor_cols[f"top_donor_{rank_idx}_affinity_pct"] = round(a_val, 2)
                    donor_cols[f"top_donor_{rank_idx}_vote_pct"] = round(r_val, 2) if pd.notnull(r_val) else np.nan
                    donor_cols[f"top_donor_{rank_idx}_conversion_ratio"] = conv
                else:
                    donor_cols[f"top_donor_{rank_idx}_party"] = None
                    donor_cols[f"top_donor_{rank_idx}_affinity_pct"] = np.nan
                    donor_cols[f"top_donor_{rank_idx}_vote_pct"] = np.nan
                    donor_cols[f"top_donor_{rank_idx}_conversion_ratio"] = np.nan
                    
            mean_conv = round(float(np.mean(conv_ratios)), 4) if conv_ratios else np.nan
            
            profile_rows.append({
                "recipient_party": p,
                "wave": w_code,
                "survey_date": s_date,
                "period": period,
                "recipient_sympathy_pct": s_symp,
                "recipient_vote_pct": s_vid,
                "distance_from_4_pp": dist_4,
                "k4_proximity": round(k4_val, 4) if pd.notnull(k4_val) else np.nan,
                **donor_cols,
                "mean_conversion_ratio_top_donors": mean_conv,
            })
            
    df_profiles = pd.DataFrame(profile_rows)
    df_profiles.sort_values(by=["recipient_party", "survey_date"], inplace=True)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_profiles.to_csv(output_file, index=False, encoding="utf-8")
        
    return df_profiles
