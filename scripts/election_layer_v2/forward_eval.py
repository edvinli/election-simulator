"""Forward chronological standalone evaluation of election layer from 14-day polling consensus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.election_residuals.config import ALL_CATEGORIES, PARLIAMENTARY_PARTIES
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting

from .config import (
    CANONICAL_WINDOW_DAYS,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    ELECTION_LAYER_V2_VARIANTS,
    FORWARD_EVALUATION_ELECTIONS,
)
from .residuals_pool import load_chronological_pp_residuals
from .transfer import apply_simplex_transfer


def compute_discrete_crps(samples: np.ndarray, actual: float) -> float:
    """Compute exact CRPS for discrete equally-weighted empirical sample points."""
    k = len(samples)
    if k == 0:
        return 0.0
    if k == 1:
        return float(abs(samples[0] - actual))

    first_term = float(np.mean(np.abs(samples - actual)))
    diff_matrix = np.abs(samples[:, None] - samples[None, :])
    second_term = 0.5 * float(np.mean(diff_matrix))
    return float(first_term - second_term)


def run_forward_election_layer_evaluation(
    elections: Sequence[date] = FORWARD_EVALUATION_ELECTIONS,
    variants: Sequence[str] = ELECTION_LAYER_V2_VARIANTS,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute standalone forward evaluation across 2010, 2014, 2018, and 2022 from 14d polling consensus."""
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    polls_df = pd.read_csv(p_file)
    election_targets = load_election_targets_for_forecasting(e_file)

    row_records: list[dict[str, Any]] = []

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

        # 1. Base
        comp_base = base_vec[None, :]  # Shape (1, 9)
        lam_base = [1.0]

        # 2. PP Bias Only
        x_bias, lam_b = apply_simplex_transfer(base_vec, training_pool.mean_bias_pp)
        comp_bias = x_bias[None, :]
        lam_bias = [lam_b]

        # 3. PP Noise Only (exact discrete K points)
        points_noise: list[np.ndarray] = []
        lams_noise: list[float] = []
        for i in range(k):
            x_n, l_n = apply_simplex_transfer(base_vec, training_pool.centered_residuals_matrix[i])
            points_noise.append(x_n)
            lams_noise.append(l_n)
        comp_noise = np.array(points_noise)  # Shape (K, 9)

        # 4. PP Bias Plus Noise (exact discrete K points)
        points_raw: list[np.ndarray] = []
        lams_raw: list[float] = []
        for i in range(k):
            x_r, l_r = apply_simplex_transfer(base_vec, training_pool.residuals_matrix[i])
            points_raw.append(x_r)
            lams_raw.append(l_r)
        comp_bias_plus_noise = np.array(points_raw)  # Shape (K, 9)

        variant_data = {
            "base": (comp_base, lam_base),
            "pp_bias_only": (comp_bias, lam_bias),
            "pp_noise_only": (comp_noise, lams_noise),
            "pp_bias_plus_noise": (comp_bias_plus_noise, lams_raw),
        }

        for var_id in variants:
            pts_mat, l_list = variant_data[var_id]
            mean_l = float(np.mean(l_list))

            for idx, party in enumerate(ALL_CATEGORIES):
                act_val = target_vec[idx]
                party_pts = pts_mat[:, idx]

                fc_mean = float(np.mean(party_pts))
                fc_p50 = float(np.median(party_pts))
                abs_err = abs(fc_mean - act_val)
                crps_val = compute_discrete_crps(party_pts, act_val)

                row_records.append({
                    "election_date": el_date.isoformat(),
                    "election_year": int(el_date.year),
                    "training_pool_size": k,
                    "training_years": list(training_pool.training_years),
                    "variant": var_id,
                    "party": party,
                    "is_parliamentary": party in PARLIAMENTARY_PARTIES,
                    "poll_consensus": round(float(base_vec[idx]), 4),
                    "actual": round(float(act_val), 4),
                    "forecast_mean": round(fc_mean, 4),
                    "forecast_p50": round(fc_p50, 4),
                    "absolute_error": round(abs_err, 4),
                    "crps": round(crps_val, 6),
                    "mean_lambda": round(mean_l, 4),
                })

    df_forward = pd.DataFrame(row_records)

    # Summaries
    by_variant_election = []
    for var_id in variants:
        for el_year in sorted(df_forward["election_year"].unique()):
            sub_ev = df_forward[(df_forward["variant"] == var_id) & (df_forward["election_year"] == el_year)]
            sub_8p = sub_ev[sub_ev["is_parliamentary"]]
            by_variant_election.append({
                "election_year": int(el_year),
                "variant": var_id,
                "training_pool_size": int(sub_ev["training_pool_size"].iloc[0]),
                "MAE_8parties": round(float(sub_8p["absolute_error"].mean()), 4),
                "mean_CRPS_8parties": round(float(sub_8p["crps"].mean()), 4),
                "MAE_all9": round(float(sub_ev["absolute_error"].mean()), 4),
                "mean_CRPS_all9": round(float(sub_ev["crps"].mean()), 4),
                "mean_lambda": round(float(sub_ev["mean_lambda"].iloc[0]), 4),
            })

    by_variant_overall = []
    for var_id in variants:
        sub_v = df_forward[df_forward["variant"] == var_id]
        sub_8p = sub_v[sub_v["is_parliamentary"]]
        by_variant_overall.append({
            "variant": var_id,
            "overall_MAE_8parties": round(float(sub_8p["absolute_error"].mean()), 4),
            "overall_mean_CRPS_8parties": round(float(sub_8p["crps"].mean()), 4),
            "overall_MAE_all9": round(float(sub_v["absolute_error"].mean()), 4),
            "overall_mean_CRPS_all9": round(float(sub_v["crps"].mean()), 4),
            "overall_mean_lambda": round(float(sub_v["mean_lambda"].mean()), 4),
        })

    csv_path = out_dir / "forward_eval_2010_2022.csv"
    json_path = out_dir / "forward_eval_2010_2022.json"

    df_forward.to_csv(csv_path, index=False)
    df_var_el = pd.DataFrame(by_variant_election)
    df_var_ov = pd.DataFrame(by_variant_overall)

    report_data = {
        "elections_evaluated": [e.isoformat() for e in elections],
        "by_variant_overall": by_variant_overall,
        "by_variant_election": by_variant_election,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return {
        "cases_df": df_forward,
        "by_election_df": df_var_el,
        "overall_df": df_var_ov,
        "report": report_data,
        "paths": {
            "cases_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
