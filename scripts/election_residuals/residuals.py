"""Compute, analyze, and export historical poll-to-election residuals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.clr import composition_to_clr

from .config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    EVALUATION_ELECTIONS,
    PARLIAMENTARY_PARTIES,
    THRESHOLD_MARGIN_PCT,
    THRESHOLD_PCT,
)
from .consensus import ElectionPollConsensus, build_election_polling_consensus


def calculate_residuals_study(
    elections: Sequence[date] = EVALUATION_ELECTIONS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete Historical Poll-to-Election Residual Study across 2002-2022 elections."""
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not p_file.exists():
        raise FileNotFoundError(f"Missing polls dataset at {p_file}")
    if not e_file.exists():
        raise FileNotFoundError(f"Missing elections dataset at {e_file}")

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    row_records: list[dict[str, Any]] = []
    poll_audit_records: list[dict[str, Any]] = []
    election_summaries: list[dict[str, Any]] = []

    for election_date in elections:
        if election_date not in election_targets:
            raise KeyError(f"No certified target found for election {election_date}")

        target_comp = election_targets[election_date]
        consensus = build_election_polling_consensus(election_date, polls_df)

        # 1. Joint CLR transformations
        clr_target, _ = composition_to_clr(target_comp, categories=ALL_CATEGORIES)
        clr_consensus, _ = composition_to_clr(consensus.consensus_composition, categories=ALL_CATEGORIES)
        clr_delta = clr_target - clr_consensus

        # 2. Record row-level category residuals
        for idx, party in enumerate(ALL_CATEGORIES):
            c_val = consensus.consensus_composition[party]
            t_val = target_comp[party]
            res_pp = t_val - c_val  # Positive = outperformed polling
            res_clr = float(clr_delta[idx])

            dist_4 = abs(c_val - THRESHOLD_PCT)
            near_thresh = dist_4 <= THRESHOLD_MARGIN_PCT

            row_records.append({
                "election_date": election_date.isoformat(),
                "election_year": election_date.year,
                "party": party,
                "is_parliamentary": party in PARLIAMENTARY_PARTIES,
                "poll_consensus": round(c_val, 4),
                "election_result": round(t_val, 4),
                "residual_pp": round(res_pp, 4),
                "residual_clr": round(res_clr, 6),
                "distance_from_4pct": round(dist_4, 4),
                "near_threshold": near_thresh,
                "poll_count_total": consensus.total_eligible_polls_in_window,
                "pollster_count": consensus.retained_pollsters_count,
            })

        # 3. Record contributing polls audit
        for cp in consensus.contributing_polls:
            audit_row = {
                "election_date": election_date.isoformat(),
                "election_year": election_date.year,
                "poll_id": cp.poll_id,
                "pollster": cp.pollster,
                "pollster_original": cp.pollster_original,
                "interview_start": cp.interview_start.isoformat() if cp.interview_start else None,
                "interview_end": cp.interview_end.isoformat(),
                "publication_date": cp.publication_date.isoformat(),
                "sample_size": cp.sample_size,
                "sample_size_missing": cp.sample_size_missing,
                "weight": cp.weight,
            }
            for p in ALL_CATEGORIES:
                audit_row[f"support_{p}"] = cp.party_support.get(p)
            poll_audit_records.append(audit_row)

        # 4. Election level summary
        sub_e = [r for r in row_records if r["election_year"] == election_date.year]
        sub_e_8p = [r for r in sub_e if r["is_parliamentary"]]
        abs_errs_8p = [abs(r["residual_pp"]) for r in sub_e_8p]
        abs_errs_all = [abs(r["residual_pp"]) for r in sub_e]
        max_miss_row = max(sub_e, key=lambda r: abs(r["residual_pp"]))

        election_summaries.append({
            "election_date": election_date.isoformat(),
            "election_year": election_date.year,
            "poll_count_total": consensus.total_eligible_polls_in_window,
            "pollster_count": consensus.retained_pollsters_count,
            "pollsters": [cp.pollster for cp in consensus.contributing_polls],
            "MAE_8parties": round(float(np.mean(abs_errs_8p)), 4),
            "MAE_all9": round(float(np.mean(abs_errs_all)), 4),
            "max_miss_party": max_miss_row["party"],
            "max_miss_pp": max_miss_row["residual_pp"],
            "max_miss_abs_pp": abs(max_miss_row["residual_pp"]),
            "REST_residual_pp": next(r["residual_pp"] for r in sub_e if r["party"] == "REST"),
        })

    df_residuals = pd.DataFrame(row_records)
    df_audit = pd.DataFrame(poll_audit_records)
    df_elections = pd.DataFrame(election_summaries)

    # 5. Party level aggregations across all 6 elections
    party_summaries: list[dict[str, Any]] = []
    for party in ALL_CATEGORIES:
        p_sub = df_residuals[df_residuals["party"] == party]
        res_vals = p_sub["residual_pp"].values
        clr_vals = p_sub["residual_clr"].values
        abs_vals = np.abs(res_vals)

        pos_count = int(np.sum(res_vals > 0))
        neg_count = int(np.sum(res_vals < 0))
        zero_count = int(np.sum(res_vals == 0))

        party_summaries.append({
            "party": party,
            "is_parliamentary": party in PARLIAMENTARY_PARTIES,
            "elections_count": len(p_sub),
            "mean_residual_pp": round(float(np.mean(res_vals)), 4),
            "median_residual_pp": round(float(np.median(res_vals)), 4),
            "std_residual_pp": round(float(np.std(res_vals, ddof=1)), 4) if len(res_vals) > 1 else 0.0,
            "MAE_pp": round(float(np.mean(abs_vals)), 4),
            "max_outperformance_pp": round(float(np.max(res_vals)), 4),
            "max_underperformance_pp": round(float(np.min(res_vals)), 4),
            "mean_residual_clr": round(float(np.mean(clr_vals)), 6),
            "sign_consistency": f"{pos_count}+ / {neg_count}- / {zero_count}zero",
            "persistent_positive": pos_count >= 5,
            "persistent_negative": neg_count >= 5,
        })

    df_parties = pd.DataFrame(party_summaries)

    # 6. Threshold diagnostic comparison (near 4% vs away)
    near_thresh_rows = df_residuals[df_residuals["near_threshold"]]
    away_thresh_rows = df_residuals[(~df_residuals["near_threshold"]) & (df_residuals["is_parliamentary"])]

    threshold_comparison = {
        "near_threshold_cases_count": len(near_thresh_rows),
        "near_threshold_MAE_pp": round(float(np.abs(near_thresh_rows["residual_pp"]).mean()), 4),
        "near_threshold_mean_residual_pp": round(float(near_thresh_rows["residual_pp"].mean()), 4),
        "away_threshold_cases_count": len(away_thresh_rows),
        "away_threshold_MAE_pp": round(float(np.abs(away_thresh_rows["residual_pp"]).mean()), 4),
        "away_threshold_mean_residual_pp": round(float(away_thresh_rows["residual_pp"].mean()), 4),
    }

    # 7. Political bloc co-movement diagnostics
    bloc_records: list[dict[str, Any]] = []
    for election_date in elections:
        sub_e = df_residuals[df_residuals["election_year"] == election_date.year].set_index("party")
        
        # Left/Green Bloc: S + V + MP
        left_poll = sub_e.loc["S", "poll_consensus"] + sub_e.loc["V", "poll_consensus"] + sub_e.loc["MP", "poll_consensus"]
        left_act = sub_e.loc["S", "election_result"] + sub_e.loc["V", "election_result"] + sub_e.loc["MP", "election_result"]
        left_diff = left_act - left_poll

        # Right/Alliansen Bloc: M + L + C + KD
        allians_poll = sub_e.loc["M", "poll_consensus"] + sub_e.loc["L", "poll_consensus"] + sub_e.loc["C", "poll_consensus"] + sub_e.loc["KD", "poll_consensus"]
        allians_act = sub_e.loc["M", "election_result"] + sub_e.loc["L", "election_result"] + sub_e.loc["C", "election_result"] + sub_e.loc["KD", "election_result"]
        allians_diff = allians_act - allians_poll

        # Right Bloc including SD (for 2018-2022)
        right_plus_sd_poll = allians_poll + sub_e.loc["SD", "poll_consensus"]
        right_plus_sd_act = allians_act + sub_e.loc["SD", "election_result"]
        right_plus_sd_diff = right_plus_sd_act - right_plus_sd_poll

        bloc_records.append({
            "election_year": election_date.year,
            "left_green_poll": round(left_poll, 2),
            "left_green_actual": round(left_act, 2),
            "left_green_residual_pp": round(left_diff, 2),
            "alliansen_poll": round(allians_poll, 2),
            "alliansen_actual": round(allians_act, 2),
            "alliansen_residual_pp": round(allians_diff, 2),
            "sd_poll": round(sub_e.loc["SD", "poll_consensus"], 2),
            "sd_actual": round(sub_e.loc["SD", "election_result"], 2),
            "sd_residual_pp": round(sub_e.loc["SD", "residual_pp"], 2),
        })

    df_blocs = pd.DataFrame(bloc_records)

    # 8. Save output datasets
    summary_csv_path = out_dir / "election_residuals_summary.csv"
    audit_csv_path = out_dir / "contributing_polls_audit.csv"
    parties_csv_path = out_dir / "residuals_by_party.csv"
    elections_csv_path = out_dir / "residuals_by_election.csv"
    blocs_csv_path = out_dir / "residuals_by_bloc.csv"
    json_summary_path = out_dir / "election_residuals_summary.json"

    df_residuals.to_csv(summary_csv_path, index=False)
    df_audit.to_csv(audit_csv_path, index=False)
    df_parties.to_csv(parties_csv_path, index=False)
    df_elections.to_csv(elections_csv_path, index=False)
    df_blocs.to_csv(blocs_csv_path, index=False)

    full_summary = {
        "study_metadata": {
            "elections_covered": [e.isoformat() for e in elections],
            "total_elections": len(elections),
            "window_days": 14,
            "total_evaluated_rows": len(df_residuals),
            "total_contributing_polls": len(df_audit),
        },
        "by_party": party_summaries,
        "by_election": election_summaries,
        "threshold_comparison": threshold_comparison,
        "bloc_diagnostics": bloc_records,
    }

    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)

    return {
        "residuals_df": df_residuals,
        "audit_df": df_audit,
        "parties_df": df_parties,
        "elections_df": df_elections,
        "blocs_df": df_blocs,
        "summary": full_summary,
        "paths": {
            "summary_csv": str(summary_csv_path),
            "audit_csv": str(audit_csv_path),
            "parties_csv": str(parties_csv_path),
            "elections_csv": str(elections_csv_path),
            "blocs_csv": str(blocs_csv_path),
            "summary_json": str(json_summary_path),
        },
    }
