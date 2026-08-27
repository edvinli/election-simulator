"""Exact standalone forward evaluation of vote-share models from 14-day polling consensus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.election_layer_v2.forward_eval import compute_discrete_crps
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_simplex_transfer
from scripts.election_residuals.config import ALL_CATEGORIES, PARLIAMENTARY_PARTIES
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting

from .config import (
    CANONICAL_MODELS,
    CANONICAL_WINDOW_DAYS,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    FORWARD_EVALUATION_ELECTIONS,
)
from .energy_score import compute_discrete_energy_score


def run_exact_forward_evaluation(
    elections: Sequence[date] = FORWARD_EVALUATION_ELECTIONS,
    models: Sequence[str] = CANONICAL_MODELS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute standalone forward evaluation across 2010, 2014, 2018, and 2022 using exact finite support."""
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    case_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for el_date in elections:
        target_comp = election_targets[el_date]
        target_vec = np.array([target_comp[c] for c in ALL_CATEGORIES], dtype=float)

        consensus = build_election_polling_consensus(el_date, polls_df, window_days=CANONICAL_WINDOW_DAYS)
        base_vec = np.array([consensus.consensus_composition[c] for c in ALL_CATEGORIES], dtype=float)

        training_pool = load_chronological_pp_residuals(
            target_election_year=el_date.year,
            window_days=CANONICAL_WINDOW_DAYS,
            polls_file=p_file,
            elections_file=e_file,
        )
        k = len(training_pool.training_years)

        # 1. Base model (1 point)
        pts_base = base_vec[None, :]
        lams_base = [1.0]

        # 2. pp_centered_noise (K points)
        pts_centered: list[np.ndarray] = []
        lams_centered: list[float] = []
        for i in range(k):
            x_c, l_c = apply_simplex_transfer(base_vec, training_pool.centered_residuals_matrix[i])
            pts_centered.append(x_c)
            lams_centered.append(l_c)
        pts_centered_mat = np.array(pts_centered)

        # 3. pp_symmetric_noise (2K points: -r_e and +r_e)
        pts_symmetric: list[np.ndarray] = []
        lams_symmetric: list[float] = []
        for i in range(k):
            # Sign -1
            x_neg, l_neg = apply_simplex_transfer(base_vec, -training_pool.residuals_matrix[i])
            pts_symmetric.append(x_neg)
            lams_symmetric.append(l_neg)
            # Sign +1
            x_pos, l_pos = apply_simplex_transfer(base_vec, +training_pool.residuals_matrix[i])
            pts_symmetric.append(x_pos)
            lams_symmetric.append(l_pos)
        pts_symmetric_mat = np.array(pts_symmetric)

        model_supports = {
            "base": (pts_base, lams_base),
            "pp_centered_noise": (pts_centered_mat, lams_centered),
            "pp_symmetric_noise": (pts_symmetric_mat, lams_symmetric),
        }

        for model_id in models:
            pts_mat, l_list = model_supports[model_id]
            mean_l = float(np.mean(l_list))

            # Exact multivariate Energy Score on full 9 categories
            es_val = compute_discrete_energy_score(pts_mat, target_vec)

            party_crps_list = []
            party_mae_list = []

            for idx, party in enumerate(ALL_CATEGORIES):
                act_val = target_vec[idx]
                party_pts = pts_mat[:, idx]

                fc_mean = float(np.mean(party_pts))
                fc_p50 = float(np.median(party_pts))
                abs_err = abs(fc_mean - act_val)
                crps_val = compute_discrete_crps(party_pts, act_val)

                if party in PARLIAMENTARY_PARTIES:
                    party_crps_list.append(crps_val)
                    party_mae_list.append(abs_err)

                case_rows.append({
                    "election_date": el_date.isoformat(),
                    "election_year": int(el_date.year),
                    "training_pool_size": k,
                    "model": model_id,
                    "party": party,
                    "is_parliamentary": party in PARLIAMENTARY_PARTIES,
                    "poll_consensus": round(float(base_vec[idx]), 4),
                    "actual": round(float(act_val), 4),
                    "forecast_mean": round(fc_mean, 4),
                    "forecast_p50": round(fc_p50, 4),
                    "absolute_error": round(abs_err, 4),
                    "crps": round(crps_val, 6),
                    "energy_score": round(es_val, 6),
                    "mean_lambda": round(mean_l, 4),
                })

            case_summaries.append({
                "election_year": int(el_date.year),
                "training_pool_size": k,
                "model": model_id,
                "MAE_8parties": round(float(np.mean(party_mae_list)), 4),
                "CRPS_8parties": round(float(np.mean(party_crps_list)), 4),
                "EnergyScore_all9": round(es_val, 6),
                "mean_lambda": round(mean_l, 4),
            })

    df_cases = pd.DataFrame(case_rows)
    df_summaries = pd.DataFrame(case_summaries)

    # Overall by model
    overall_summaries: list[dict[str, Any]] = []
    for model_id in models:
        sub_m = df_summaries[df_summaries["model"] == model_id]
        overall_summaries.append({
            "model": model_id,
            "overall_MAE_8parties": round(float(sub_m["MAE_8parties"].mean()), 4),
            "overall_CRPS_8parties": round(float(sub_m["CRPS_8parties"].mean()), 4),
            "overall_EnergyScore_all9": round(float(sub_m["EnergyScore_all9"].mean()), 6),
            "overall_mean_lambda": round(float(sub_m["mean_lambda"].mean()), 4),
        })
    df_overall = pd.DataFrame(overall_summaries)

    csv_path = out_dir / "forward_eval_2010_2022.csv"
    json_path = out_dir / "forward_eval_2010_2022.json"

    df_cases.to_csv(csv_path, index=False)

    report_data = {
        "elections_evaluated": [e.isoformat() for e in elections],
        "by_model_overall": overall_summaries,
        "by_model_election": case_summaries,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return {
        "cases_df": df_cases,
        "summaries_df": df_summaries,
        "overall_df": df_overall,
        "report": report_data,
        "paths": {
            "cases_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
