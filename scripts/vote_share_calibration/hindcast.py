"""Hindcast execution engine for vote-share models with paired Monte Carlo and Energy Score."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import summarize_lambda_diagnostics
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.backtest_metrics import calculate_crps, precompute_crps_sample_term
from scripts.pollofpolls.clr import clr_to_composition_matrix
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES
from scripts.pollofpolls.transitions import build_all_historical_transitions, filter_transitions_as_of

from .config import (
    CANONICAL_MODELS,
    CANONICAL_WINDOW_DAYS,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_HORIZONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    ENERGY_SCORE_SUBSET_SIZE,
    EVALUATION_ELECTIONS,
    INITIAL_SAMPLES,
    INITIAL_SEED,
)
from .energy_score import compute_energy_score
from .models import apply_vote_share_models, derive_vote_share_layer_seeds


QUANTILES_TO_TRACK = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def calculate_empirical_midrank_percentile(samples: np.ndarray, actual: float) -> float:
    """Calculate empirical mid-rank percentile: P = 100 * (#(x < y) + 0.5 * #(x == y)) / n."""
    n = len(samples)
    if n == 0:
        return 0.0
    less_count = np.sum(samples < actual)
    equal_count = np.sum(samples == actual)
    return round(float(100.0 * (less_count + 0.5 * equal_count) / n), 2)


def run_vote_share_hindcasts(
    elections: Sequence[date] = EVALUATION_ELECTIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    models: Sequence[str] = CANONICAL_MODELS,
    samples: int = INITIAL_SAMPLES,
    seed: int = INITIAL_SEED,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run full-pipeline paired hindcasts evaluating 8-party CRPS and 9-category Energy Score."""
    p_file = Path(polls_file) if polls_file else DEFAULT_POLLS_FILE
    e_file = Path(elections_file) if elections_file else DEFAULT_ELECTIONS_FILE
    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    ts_file = base_path / "pollofpolls_timeseries.csv"
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timeseries_data = load_timeseries_dataset(ts_file)
    ts_by_date = {row["date"]: row for row in timeseries_data}
    election_targets = load_election_targets_for_forecasting(e_file)
    all_transitions_by_horizon = build_all_historical_transitions(timeseries_data, horizons=horizons)

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    cases_evaluated_count = 0

    for election_date in elections:
        target_comp = election_targets[election_date]
        target_vec = np.array([target_comp[c] for c in ALL_CATEGORIES], dtype=float)

        training_pool = load_chronological_pp_residuals(
            target_election_year=election_date.year,
            window_days=CANONICAL_WINDOW_DAYS,
            polls_file=p_file,
            elections_file=e_file,
        )

        for h in sorted(horizons, reverse=True):
            origin_date = election_date - timedelta(days=h)
            if origin_date not in ts_by_date:
                raise KeyError(f"Exact origin date {origin_date} missing in timeseries")

            eligible_trans = filter_transitions_as_of(all_transitions_by_horizon[h], origin_date)
            if len(eligible_trans) < 30:
                raise ValueError(f"Insufficient historical transitions ({len(eligible_trans)} < 30) for {origin_date}")

            # Paired seed derivation
            state_seed = derive_opinion_state_seed(seed, origin_date)
            dyn_seed = derive_shared_dynamics_seed(seed, origin_date, h)
            idx_seed, sign_seed = derive_vote_share_layer_seeds(seed, origin_date, h)

            # OpinionState + Dynamics base composition matrix
            op_state = estimate_opinion(as_of=origin_date, data_dir=base_path)
            state_samples = op_state.sample(n=samples, seed=state_seed)
            state_matrix = np.array([[s[cat] for cat in ALL_CATEGORIES] for s in state_samples], dtype=float)
            log_state = np.log(state_matrix)
            state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)

            sym_deltas = sample_shared_symmetric_dynamics(eligible_trans, samples, dyn_seed)
            base_clr_matrix = state_clr + sym_deltas
            base_comp_matrix = clr_to_composition_matrix(base_clr_matrix)

            # Apply all models with strict pairing
            model_outputs = apply_vote_share_models(
                base_comp_matrix=base_comp_matrix,
                training_pool=training_pool,
                samples_count=samples,
                index_seed=idx_seed,
                sign_seed=sign_seed,
            )

            cases_evaluated_count += 1

            for model_id in models:
                samples_mat, lam_arr = model_outputs[model_id]
                lam_diag = summarize_lambda_diagnostics(lam_arr)

                # Multivariate Energy Score on full 9 categories
                if samples <= ENERGY_SCORE_SUBSET_SIZE:
                    es_samples = samples_mat
                else:
                    # Deterministic evenly-spaced subset of fixed size 5000
                    step = samples // ENERGY_SCORE_SUBSET_SIZE
                    es_samples = samples_mat[::step][:ENERGY_SCORE_SUBSET_SIZE]

                es_val = compute_energy_score(es_samples, target_vec)

                party_crps_list = []
                party_mae_list = []

                for idx, party in enumerate(ALL_CATEGORIES):
                    actual_val = target_comp[party]
                    party_samples = samples_mat[:, idx]
                    sorted_samples = np.sort(party_samples)

                    q_vals = np.quantile(sorted_samples, QUANTILES_TO_TRACK, method="linear")
                    q_dict = {q: float(v) for q, v in zip(QUANTILES_TO_TRACK, q_vals)}

                    p05, p10, p25, p50, p75, p90, p95 = (
                        q_dict[0.05],
                        q_dict[0.10],
                        q_dict[0.25],
                        q_dict[0.50],
                        q_dict[0.75],
                        q_dict[0.90],
                        q_dict[0.95],
                    )

                    fc_mean = float(np.mean(sorted_samples))
                    abs_err = abs(p50 - actual_val)
                    sample_term = precompute_crps_sample_term(sorted_samples)
                    crps_val = calculate_crps(sorted_samples, actual_val, precomputed_sample_term=sample_term)
                    act_percentile = calculate_empirical_midrank_percentile(sorted_samples, actual_val)

                    if party in PARTIES:
                        party_crps_list.append(crps_val)
                        party_mae_list.append(abs_err)

                    inside_50 = p25 <= actual_val <= p75
                    inside_80 = p10 <= actual_val <= p90
                    inside_90 = p05 <= actual_val <= p95

                    w50 = p75 - p25
                    w80 = p90 - p10
                    w90 = p95 - p05

                    row_dict = {
                        "election_date": election_date.isoformat(),
                        "election_year": int(election_date.year),
                        "origin_date": origin_date.isoformat(),
                        "horizon_days": h,
                        "model": model_id,
                        "party": party,
                        "is_parliamentary": party in PARTIES,
                        "actual": round(actual_val, 4),
                        "forecast_mean": round(fc_mean, 4),
                        "p05": round(p05, 4),
                        "p10": round(p10, 4),
                        "p25": round(p25, 4),
                        "p50": round(p50, 4),
                        "p75": round(p75, 4),
                        "p90": round(p90, 4),
                        "p95": round(p95, 4),
                        "actual_percentile": act_percentile,
                        "absolute_error": round(abs_err, 4),
                        "crps": round(crps_val, 6),
                        "energy_score": round(es_val, 6),
                        "inside_50": inside_50,
                        "inside_80": inside_80,
                        "inside_90": inside_90,
                        "width_50": round(w50, 4),
                        "width_80": round(w80, 4),
                        "width_90": round(w90, 4),
                        "mean_lambda": lam_diag["mean_lambda"],
                        "p05_lambda": lam_diag["p05_lambda"],
                        "fraction_lambda_lt_0_99": lam_diag["fraction_lambda_lt_0_99"],
                        "fraction_lambda_lt_0_90": lam_diag["fraction_lambda_lt_0_90"],
                        "fraction_lambda_lt_0_75": lam_diag["fraction_lambda_lt_0_75"],
                        "training_elections_count": len(training_pool.training_years),
                        "training_elections": list(training_pool.training_years),
                        "seed": seed,
                        "samples": samples,
                    }
                    rows.append(row_dict)

                case_summaries.append({
                    "election_date": election_date.isoformat(),
                    "election_year": int(election_date.year),
                    "origin_date": origin_date.isoformat(),
                    "horizon_days": h,
                    "model": model_id,
                    "MAE_8parties": round(float(np.mean(party_mae_list)), 4),
                    "CRPS_8parties": round(float(np.mean(party_crps_list)), 4),
                    "EnergyScore_all9": round(es_val, 6),
                    "mean_lambda": lam_diag["mean_lambda"],
                    "fraction_lambda_lt_0_90": lam_diag["fraction_lambda_lt_0_90"],
                })

    df_cases = pd.DataFrame(rows)
    df_case_sums = pd.DataFrame(case_summaries)

    def summarize_group(df_sub: pd.DataFrame) -> dict[str, Any]:
        sub_8p = df_sub[df_sub["is_parliamentary"]]
        sub_rest = df_sub[df_sub["party"] == "REST"]
        return {
            "forecast_cases_count": len(df_sub[["election_date", "horizon_days"]].drop_duplicates()),
            "party_rows_count": len(df_sub),
            "MAE_8parties": float(sub_8p["absolute_error"].mean()) if not sub_8p.empty else 0.0,
            "mean_CRPS_8parties": float(sub_8p["crps"].mean()) if not sub_8p.empty else 0.0,
            "MAE_all9": float(df_sub["absolute_error"].mean()),
            "mean_CRPS_all9": float(df_sub["crps"].mean()),
            "REST_CRPS": float(sub_rest["crps"].mean()) if not sub_rest.empty else 0.0,
            "EnergyScore_all9": float(df_sub["energy_score"].mean()),
            "coverage_50": float(df_sub["inside_50"].mean()),
            "coverage_80": float(df_sub["inside_80"].mean()),
            "coverage_90": float(df_sub["inside_90"].mean()),
            "mean_width_50": float(df_sub["width_50"].mean()),
            "mean_width_80": float(df_sub["width_80"].mean()),
            "mean_width_90": float(df_sub["width_90"].mean()),
            "mean_lambda": float(df_sub["mean_lambda"].mean()),
            "fraction_lambda_lt_0_90": float(df_sub["fraction_lambda_lt_0_90"].mean()),
        }

    by_model_overall = []
    by_election_model = []
    by_horizon_model = []
    by_party_model = []

    for m in models:
        sub_m = df_cases[df_cases["model"] == m]
        if not sub_m.empty:
            m_dict = summarize_group(sub_m)
            m_dict["model"] = m
            by_model_overall.append(m_dict)

        for e_date in sorted(df_cases["election_date"].unique()):
            sub_em = df_cases[(df_cases["model"] == m) & (df_cases["election_date"] == e_date)]
            if not sub_em.empty:
                em_dict = summarize_group(sub_em)
                em_dict["model"] = m
                em_dict["election_date"] = e_date
                by_election_model.append(em_dict)

        for h in horizons:
            sub_hm = df_cases[(df_cases["model"] == m) & (df_cases["horizon_days"] == h)]
            if not sub_hm.empty:
                hm_dict = summarize_group(sub_hm)
                hm_dict["model"] = m
                hm_dict["horizon_days"] = h
                by_horizon_model.append(hm_dict)

        for p in ALL_CATEGORIES:
            sub_pm = df_cases[(df_cases["model"] == m) & (df_cases["party"] == p)]
            if not sub_pm.empty:
                by_party_model.append({
                    "model": m,
                    "party": p,
                    "is_parliamentary": p in PARTIES,
                    "MAE": float(sub_pm["absolute_error"].mean()),
                    "CRPS": float(sub_pm["crps"].mean()),
                    "coverage_50": float(sub_pm["inside_50"].mean()),
                    "coverage_80": float(sub_pm["inside_80"].mean()),
                    "coverage_90": float(sub_pm["inside_90"].mean()),
                    "mean_width_50": float(sub_pm["width_50"].mean()),
                    "mean_width_80": float(sub_pm["width_80"].mean()),
                    "mean_width_90": float(sub_pm["width_90"].mean()),
                    "avg_percentile": float(sub_pm["actual_percentile"].mean()),
                })

    df_by_model_overall = pd.DataFrame(by_model_overall)
    df_by_election_model = pd.DataFrame(by_election_model)
    df_by_horizon_model = pd.DataFrame(by_horizon_model)
    df_by_party_model = pd.DataFrame(by_party_model)

    cases_csv_path = out_dir / "vote_share_cases_2018_2022.csv"
    json_summary_path = out_dir / "vote_share_summary_2018_2022.json"
    election_csv_path = out_dir / "vote_share_by_election_model.csv"
    horizon_csv_path = out_dir / "vote_share_by_horizon_model.csv"
    party_csv_path = out_dir / "vote_share_by_party_model.csv"

    df_cases.to_csv(cases_csv_path, index=False)
    df_by_election_model.to_csv(election_csv_path, index=False)
    df_by_horizon_model.to_csv(horizon_csv_path, index=False)
    df_by_party_model.to_csv(party_csv_path, index=False)

    summary_data = {
        "elections": [e.isoformat() for e in elections],
        "horizons": list(horizons),
        "models": list(models),
        "samples": samples,
        "base_seed": seed,
        "forecast_cases_per_model": cases_evaluated_count,
        "party_rows_per_model": cases_evaluated_count * len(ALL_CATEGORIES),
        "total_party_rows": len(df_cases),
        "by_model_overall": df_by_model_overall.to_dict(orient="records"),
        "by_election_model": df_by_election_model.to_dict(orient="records"),
        "by_horizon_model": df_by_horizon_model.to_dict(orient="records"),
        "by_party_model": df_by_party_model.to_dict(orient="records"),
    }

    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "cases_df": df_cases,
        "case_summaries_df": df_case_sums,
        "summary": summary_data,
        "paths": {
            "cases_csv": str(cases_csv_path),
            "summary_json": str(json_summary_path),
            "election_csv": str(election_csv_path),
            "horizon_csv": str(horizon_csv_path),
            "party_csv": str(party_csv_path),
        },
    }
