"""Offline processor for SCB PSU support-voting data.

Transforms raw SCB JSON archives into clean, normalized long-form CSV datasets
with canonical party mappings, category classifications, separate uncertainty measures,
and joined donor-recipient panel without imputation.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from scripts.scb_support_voting.config import (
    PARLIAMENTARY_PARTIES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SCB_TABLES,
    classify_category,
    parse_wave_period,
)


def load_raw_json(filename: str, raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Any]:
    """Load raw JSON file from raw directory."""
    path = raw_dir / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_code_label_map(meta: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Build a mapping of {var_code: {value_code: value_text}} from SCB metadata."""
    code_map: Dict[str, Dict[str, str]] = {}
    for var in meta.get("variables", []):
        v_code = var["code"]
        values = var.get("values", [])
        value_texts = var.get("valueTexts", [])
        code_map[v_code] = dict(zip(values, value_texts))
    return code_map


def parse_scb_cell_value(val_str: Optional[str]) -> Tuple[Optional[float], str]:
    """Parse SCB cell string into (float_value, value_status).
    
    Status can be: 'observed', 'suppressed', or 'missing'.
    SCB cell value '..' indicates suppressed observation due to sample size.
    """
    if val_str is None:
        return np.nan, "missing"
    val_clean = str(val_str).strip()
    if val_clean in ["..", ".", "-"]:
        return np.nan, "suppressed"
    if val_clean == "":
        return np.nan, "missing"
    try:
        val_float = float(val_clean)
        return val_float, "observed"
    except ValueError:
        return np.nan, "missing"


def process_table_a(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Process Table A: Vote intention conditional on party sympathy (Rostningssympati170)."""
    meta = load_raw_json("table_a_vote_by_sympathy_metadata.json", raw_dir)
    data = load_raw_json("table_a_vote_by_sympathy_data.json", raw_dir)
    code_map = build_code_label_map(meta)
    
    rows = []
    for r in data.get("data", []):
        donor_code, vote_code, wave = r["key"]
        est_raw, moe_raw = r["values"][0], r["values"][1]
        
        est_val, est_status = parse_scb_cell_value(est_raw)
        moe_val, _ = parse_scb_cell_value(moe_raw)
        
        donor_label = code_map.get("Psymp", {}).get(donor_code, donor_code)
        vote_label = code_map.get("Pvalnu", {}).get(vote_code, vote_code)
        
        best_party_canon, best_party_type = classify_category(donor_code, donor_label)
        vote_party_canon, vote_party_type = classify_category(vote_code, vote_label)
        
        period, date_str = parse_wave_period(wave)
        
        rows.append({
            "wave": wave,
            "survey_date": date_str,
            "period": period,
            "best_party_code_raw": donor_code,
            "best_party_raw": donor_label,
            "best_party": best_party_canon,
            "best_party_type": best_party_type,
            "vote_party_code_raw": vote_code,
            "vote_party_raw": vote_label,
            "vote_party": vote_party_canon,
            "vote_party_type": vote_party_type,
            "estimate_pct": est_val,
            "margin_error_pp": moe_val,
            "value_status": est_status,
            "source_table": SCB_TABLES["table_a_vote_by_sympathy"]["path"],
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["wave", "best_party_code_raw", "vote_party_code_raw"], inplace=True)
    return df


def process_table_b(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Process Table B: Second-best party conditional on party sympathy (Nastbastaparti190)."""
    meta = load_raw_json("table_b_second_choice_by_sympathy_metadata.json", raw_dir)
    data = load_raw_json("table_b_second_choice_by_sympathy_data.json", raw_dir)
    code_map = build_code_label_map(meta)
    
    rows = []
    for r in data.get("data", []):
        donor_code, second_code, wave = r["key"]
        est_raw, moe_raw = r["values"][0], r["values"][1]
        
        est_val, est_status = parse_scb_cell_value(est_raw)
        moe_val, _ = parse_scb_cell_value(moe_raw)
        
        donor_label = code_map.get("Psymp", {}).get(donor_code, donor_code)
        second_label = code_map.get("Nastbastaparti", {}).get(second_code, second_code)
        
        best_party_canon, best_party_type = classify_category(donor_code, donor_label)
        second_party_canon, second_party_type = classify_category(second_code, second_label)
        
        period, date_str = parse_wave_period(wave)
        
        rows.append({
            "wave": wave,
            "survey_date": date_str,
            "period": period,
            "best_party_code_raw": donor_code,
            "best_party_raw": donor_label,
            "best_party": best_party_canon,
            "best_party_type": best_party_type,
            "second_choice_code_raw": second_code,
            "second_choice_raw": second_label,
            "second_choice_party": second_party_canon,
            "second_choice_type": second_party_type,
            "estimate_pct": est_val,
            "margin_error_pp": moe_val,
            "value_status": est_status,
            "source_table": SCB_TABLES["table_b_second_choice_by_sympathy"]["path"],
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["wave", "best_party_code_raw", "second_choice_code_raw"], inplace=True)
    return df


def process_table_c(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Process Table C: Overall vote intention / Val idag (Vid10)."""
    meta = load_raw_json("table_c_overall_vote_intention_metadata.json", raw_dir)
    data = load_raw_json("table_c_overall_vote_intention_data.json", raw_dir)
    code_map = build_code_label_map(meta)
    
    rows = []
    for r in data.get("data", []):
        party_code, wave = r["key"]
        est_raw, moe_raw = r["values"][0], r["values"][1]
        
        est_val, est_status = parse_scb_cell_value(est_raw)
        moe_val, _ = parse_scb_cell_value(moe_raw)
        
        party_label = code_map.get("Parti", {}).get(party_code, party_code)
        party_canon, party_type = classify_category(party_code, party_label)
        
        period, date_str = parse_wave_period(wave)
        
        rows.append({
            "wave": wave,
            "survey_date": date_str,
            "period": period,
            "party_code_raw": party_code,
            "party_raw": party_label,
            "party": party_canon,
            "party_type": party_type,
            "estimate_pct": est_val,
            "margin_error_pp": moe_val,
            "value_status": est_status,
            "source_table": SCB_TABLES["table_c_overall_vote_intention"]["path"],
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["wave", "party_code_raw"], inplace=True)
    return df


def process_table_d(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Process Table D: Overall party sympathy (Partisympati051)."""
    meta = load_raw_json("table_d_overall_party_sympathy_metadata.json", raw_dir)
    data = load_raw_json("table_d_overall_party_sympathy_data.json", raw_dir)
    code_map = build_code_label_map(meta)
    
    rows = []
    for r in data.get("data", []):
        kon_code, alder_code, party_code, wave = r["key"]
        est_raw, moe_raw = r["values"][0], r["values"][1]
        
        est_val, est_status = parse_scb_cell_value(est_raw)
        moe_val, _ = parse_scb_cell_value(moe_raw)
        
        party_label = code_map.get("Parti", {}).get(party_code, party_code)
        party_canon, party_type = classify_category(party_code, party_label)
        
        period, date_str = parse_wave_period(wave)
        
        rows.append({
            "wave": wave,
            "survey_date": date_str,
            "period": period,
            "kon_code_raw": kon_code,
            "alder_code_raw": alder_code,
            "party_code_raw": party_code,
            "party_raw": party_label,
            "party": party_canon,
            "party_type": party_type,
            "estimate_pct": est_val,
            "margin_error_pp": moe_val,
            "value_status": est_status,
            "source_table": SCB_TABLES["table_d_overall_party_sympathy"]["path"],
        })
        
    df = pd.DataFrame(rows)
    df.sort_values(by=["wave", "party_code_raw"], inplace=True)
    return df


def build_donor_recipient_panel(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_c: pd.DataFrame,
    df_d: pd.DataFrame,
) -> pd.DataFrame:
    """Build joined donor-recipient panel for parliamentary parties.
    
    Contains cross-party flow estimates, second-choice estimates,
    overall party sympathies, and overall vote intentions with separate uncertainty measures.
    """
    # Filter conditional tables to parliamentary parties
    vote_flows = df_a[
        df_a["best_party"].isin(PARLIAMENTARY_PARTIES) &
        df_a["vote_party"].isin(PARLIAMENTARY_PARTIES)
    ].copy()
    
    second_flows = df_b[
        df_b["best_party"].isin(PARLIAMENTARY_PARTIES) &
        df_b["second_choice_party"].isin(PARLIAMENTARY_PARTIES)
    ].copy()
    
    # Overall sympathy (Table D)
    symp_overall = df_d[df_d["party"].isin(PARLIAMENTARY_PARTIES)].copy()
    
    # Overall vote intention (Table C)
    vote_overall = df_c[df_c["party"].isin(PARLIAMENTARY_PARTIES)].copy()
    
    # Prepare merge keys
    vote_flows.rename(
        columns={
            "best_party": "donor_party",
            "best_party_code_raw": "donor_party_code_raw",
            "vote_party": "recipient_party",
            "vote_party_code_raw": "recipient_party_code_raw",
            "estimate_pct": "vote_estimate_pct",
            "margin_error_pp": "vote_margin_error_pp",
            "value_status": "vote_value_status",
        },
        inplace=True,
    )
    vote_sub = vote_flows[[
        "wave", "survey_date", "period",
        "donor_party", "donor_party_code_raw",
        "recipient_party", "recipient_party_code_raw",
        "vote_estimate_pct", "vote_margin_error_pp", "vote_value_status"
    ]]
    
    second_flows.rename(
        columns={
            "best_party": "donor_party",
            "second_choice_party": "recipient_party",
            "estimate_pct": "second_choice_estimate_pct",
            "margin_error_pp": "second_choice_margin_error_pp",
            "value_status": "second_choice_value_status",
        },
        inplace=True,
    )
    second_sub = second_flows[[
        "wave", "donor_party", "recipient_party",
        "second_choice_estimate_pct", "second_choice_margin_error_pp", "second_choice_value_status"
    ]]
    
    panel = pd.merge(
        vote_sub,
        second_sub,
        on=["wave", "donor_party", "recipient_party"],
        how="outer",
    )
    
    # Merge donor overall sympathy
    donor_symp = symp_overall[[
        "wave", "party", "estimate_pct", "margin_error_pp", "value_status"
    ]].rename(
        columns={
            "party": "donor_party",
            "estimate_pct": "donor_overall_sympathy_pct",
            "margin_error_pp": "donor_overall_sympathy_margin_error_pp",
            "value_status": "donor_overall_sympathy_value_status",
        }
    )
    panel = pd.merge(panel, donor_symp, on=["wave", "donor_party"], how="left")
    
    # Merge recipient overall sympathy
    recip_symp = symp_overall[[
        "wave", "party", "estimate_pct", "margin_error_pp", "value_status"
    ]].rename(
        columns={
            "party": "recipient_party",
            "estimate_pct": "recipient_overall_sympathy_pct",
            "margin_error_pp": "recipient_overall_sympathy_margin_error_pp",
            "value_status": "recipient_overall_sympathy_value_status",
        }
    )
    panel = pd.merge(panel, recip_symp, on=["wave", "recipient_party"], how="left")
    
    # Merge recipient overall vote intention (Vid10)
    recip_vote = vote_overall[[
        "wave", "party", "estimate_pct", "margin_error_pp", "value_status"
    ]].rename(
        columns={
            "party": "recipient_party",
            "estimate_pct": "recipient_overall_vote_pct",
            "margin_error_pp": "recipient_overall_vote_margin_error_pp",
            "value_status": "recipient_overall_vote_value_status",
        }
    )
    panel = pd.merge(panel, recip_vote, on=["wave", "recipient_party"], how="left")
    
    # Sort deterministically
    panel.sort_values(by=["wave", "donor_party", "recipient_party"], inplace=True)
    return panel


def process_all(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = PROCESSED_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Run full offline processing pipeline and save CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing Table A (Vote by Sympathy)...")
    df_a = process_table_a(raw_dir)
    file_a = output_dir / "vote_by_sympathy.csv"
    df_a.to_csv(file_a, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_a)} rows to {file_a}")
    
    print("Processing Table B (Second Choice by Sympathy)...")
    df_b = process_table_b(raw_dir)
    file_b = output_dir / "second_choice_by_sympathy.csv"
    df_b.to_csv(file_b, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_b)} rows to {file_b}")
    
    print("Processing Table C (Overall Vote Intention)...")
    df_c = process_table_c(raw_dir)
    file_c = output_dir / "overall_vote_intention.csv"
    df_c.to_csv(file_c, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_c)} rows to {file_c}")
    
    print("Processing Table D (Overall Party Sympathy)...")
    df_d = process_table_d(raw_dir)
    file_d = output_dir / "overall_party_sympathy.csv"
    df_d.to_csv(file_d, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_d)} rows to {file_d}")
    
    print("Building Joined Donor-Recipient Panel...")
    df_panel = build_donor_recipient_panel(df_a, df_b, df_c, df_d)
    file_panel = output_dir / "scb_donor_recipient_panel.csv"
    df_panel.to_csv(file_panel, index=False, encoding="utf-8")
    print(f"  -> Saved {len(df_panel)} rows to {file_panel}")
    
    return {
        "vote_by_sympathy": df_a,
        "second_choice_by_sympathy": df_b,
        "overall_vote_intention": df_c,
        "overall_party_sympathy": df_d,
        "donor_recipient_panel": df_panel,
    }


if __name__ == "__main__":
    process_all()
