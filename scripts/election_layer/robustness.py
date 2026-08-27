"""Residual-window robustness audit comparing 7-day, 14-day, and 21-day consensus windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
    PARLIAMENTARY_PARTIES,
    THRESHOLD_MARGIN_PCT,
    THRESHOLD_PCT,
)
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.clr import composition_to_clr

from .config import (
    ALL_HISTORICAL_ELECTIONS,
    DEFAULT_RESIDUALS_DIR,
    ROBUSTNESS_WINDOWS,
)


def run_window_robustness_audit(
    elections: Sequence[date] = ALL_HISTORICAL_ELECTIONS,
    windows: Sequence[int] = ROBUSTNESS_WINDOWS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute robustness audit comparing historical poll-to-election residuals across 7d, 14d, 21d windows."""
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_RESIDUALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    row_records: list[dict[str, Any]] = []

    for w in windows:
        for el_date in elections:
            target_comp = election_targets[el_date]
            consensus = build_election_polling_consensus(el_date, polls_df, window_days=w)

            clr_target, _ = composition_to_clr(target_comp, categories=ALL_CATEGORIES)
            clr_consensus, _ = composition_to_clr(consensus.consensus_composition, categories=ALL_CATEGORIES)
            clr_delta = clr_target - clr_consensus

            for idx, party in enumerate(ALL_CATEGORIES):
                c_val = consensus.consensus_composition[party]
                t_val = target_comp[party]
                res_pp = t_val - c_val
                res_clr = float(clr_delta[idx])

                row_records.append({
                    "window_days": w,
                    "election_date": el_date.isoformat(),
                    "election_year": el_date.year,
                    "party": party,
                    "is_parliamentary": party in PARLIAMENTARY_PARTIES,
                    "poll_consensus": round(c_val, 4),
                    "election_result": round(t_val, 4),
                    "residual_pp": round(res_pp, 4),
                    "residual_clr": round(res_clr, 6),
                    "distance_from_4pct": round(abs(c_val - THRESHOLD_PCT), 4),
                    "near_threshold": abs(c_val - THRESHOLD_PCT) <= THRESHOLD_MARGIN_PCT,
                    "poll_count_total": consensus.total_eligible_polls_in_window,
                    "pollster_count": consensus.retained_pollsters_count,
                })

    df_rows = pd.DataFrame(row_records)

    # 1. Summary by Window & Party
    party_window_summary: list[dict[str, Any]] = []
    for w in windows:
        sub_w = df_rows[df_rows["window_days"] == w]
        for party in ALL_CATEGORIES:
            p_sub = sub_w[sub_w["party"] == party]
            res_vals = p_sub["residual_pp"].values
            clr_vals = p_sub["residual_clr"].values
            abs_vals = np.abs(res_vals)

            pos = int(np.sum(res_vals > 0))
            neg = int(np.sum(res_vals < 0))
            zero = int(np.sum(res_vals == 0))

            party_window_summary.append({
                "window_days": w,
                "party": party,
                "is_parliamentary": party in PARLIAMENTARY_PARTIES,
                "mean_residual_pp": round(float(np.mean(res_vals)), 4),
                "median_residual_pp": round(float(np.median(res_vals)), 4),
                "MAE_pp": round(float(np.mean(abs_vals)), 4),
                "mean_residual_clr": round(float(np.mean(clr_vals)), 6),
                "sign_consistency": f"{pos}+ / {neg}- / {zero}zero",
            })

    # 2. Summary by Window & Election
    election_window_summary: list[dict[str, Any]] = []
    for w in windows:
        sub_w = df_rows[df_rows["window_days"] == w]
        for el_year in sorted(df_rows["election_year"].unique()):
            e_sub = sub_w[sub_w["election_year"] == el_year]
            e_8p = e_sub[e_sub["is_parliamentary"]]
            abs_errs_8p = np.abs(e_8p["residual_pp"].values)
            abs_errs_all = np.abs(e_sub["residual_pp"].values)

            election_window_summary.append({
                "window_days": int(w),
                "election_year": int(el_year),
                "pollster_count": int(e_sub["pollster_count"].iloc[0]),
                "MAE_8parties": round(float(np.mean(abs_errs_8p)), 4),
                "MAE_all9": round(float(np.mean(abs_errs_all)), 4),
                "S_residual_pp": round(float(e_sub[e_sub["party"] == "S"]["residual_pp"].iloc[0]), 4),
                "V_residual_pp": round(float(e_sub[e_sub["party"] == "V"]["residual_pp"].iloc[0]), 4),
                "MP_residual_pp": round(float(e_sub[e_sub["party"] == "MP"]["residual_pp"].iloc[0]), 4),
            })

    # 3. Overall window metrics
    window_overall_summary: list[dict[str, Any]] = []
    for w in windows:
        sub_w = df_rows[df_rows["window_days"] == w]
        sub_8p = sub_w[sub_w["is_parliamentary"]]
        near_t = sub_w[sub_w["near_threshold"]]

        window_overall_summary.append({
            "window_days": int(w),
            "overall_MAE_8parties": round(float(np.abs(sub_8p["residual_pp"]).mean()), 4),
            "overall_MAE_all9": round(float(np.abs(sub_w["residual_pp"]).mean()), 4),
            "S_mean_residual_pp": round(float(sub_w[sub_w["party"] == "S"]["residual_pp"].mean()), 4),
            "V_mean_residual_pp": round(float(sub_w[sub_w["party"] == "V"]["residual_pp"].mean()), 4),
            "MP_mean_residual_pp": round(float(sub_w[sub_w["party"] == "MP"]["residual_pp"].mean()), 4),
            "near_threshold_MAE_pp": round(float(np.abs(near_t["residual_pp"]).mean()), 4),
            "near_threshold_cases_count": int(len(near_t)),
        })


    df_party_summary = pd.DataFrame(party_window_summary)
    df_election_summary = pd.DataFrame(election_window_summary)
    df_window_overall = pd.DataFrame(window_overall_summary)

    csv_path = out_dir / "residual_window_robustness.csv"
    json_path = out_dir / "residual_window_robustness.json"

    df_party_summary.to_csv(csv_path, index=False)

    report_data = {
        "windows_evaluated": list(windows),
        "elections_covered": [e.isoformat() for e in elections],
        "window_overall": window_overall_summary,
        "by_party_and_window": party_window_summary,
        "by_election_and_window": election_window_summary,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return {
        "all_cases_df": df_rows,
        "party_summary_df": df_party_summary,
        "election_summary_df": df_election_summary,
        "window_overall_df": df_window_overall,
        "report": report_data,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
