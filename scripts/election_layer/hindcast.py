"""Hindcast execution engine for Election Result Layer v1."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.elections.load import load_election_targets_for_forecasting
from scripts.hindcasts.models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    sample_shared_symmetric_dynamics,
)
from scripts.pollofpolls.backtest_metrics import calculate_crps, precompute_crps_sample_term
from scripts.pollofpolls.state import estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES
from scripts.pollofpolls.transitions import build_all_historical_transitions, filter_transitions_as_of

from .config import (
    CANONICAL_WINDOW_DAYS,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_HORIZONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLLS_FILE,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    ELECTION_LAYER_VARIANTS,
    EVALUATION_ELECTIONS,
)
from .models import (
    ChronologicalTrainingResiduals,
    apply_election_layer_variants,
    derive_election_layer_seed,
    load_chronological_training_residuals,
)


QUANTILES_TO_TRACK = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def calculate_empirical_midrank_percentile(samples: np.ndarray, actual: float) -> float:
    """Calculate empirical mid-rank percentile: P = 100 * (#(x < y) + 0.5 * #(x == y)) / n."""
    n = len(samples)
    if n == 0:
        return 0.0
    less_count = np.sum(samples < actual)
    equal_count = np.sum(samples == actual)
    return round(float(100.0 * (less_count + 0.5 * equal_count) / n), 2)


def run_election_layer_hindcasts(
    elections: Sequence[date] = EVALUATION_ELECTIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    variants: Sequence[str] = ELECTION_LAYER_VARIANTS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    polls_file: Path | str | None = None,
    elections_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run paired election-layer hindcasts across 2018 and 2022 for all 4 variants."""
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
    cases_evaluated_count = 0

    for election_date in elections:
        target_comp = election_targets[election_date]

        # 1. Load strict chronological training pool for this election year
        training_pool = load_chronological_training_residuals(
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

            # 2. Derive paired deterministic seeds
            state_seed = derive_opinion_state_seed(seed, origin_date)
            dyn_seed = derive_shared_dynamics_seed(seed, origin_date, h)
            layer_seed = derive_election_layer_seed(seed, origin_date, h)

            # 3. Estimate OpinionState and draw paired samples
            op_state = estimate_opinion(as_of=origin_date, data_dir=base_path)
            state_samples = op_state.sample(n=samples, seed=state_seed)
            state_matrix = np.array([[s[cat] for cat in ALL_CATEGORIES] for s in state_samples], dtype=float)
            log_state = np.log(state_matrix)
            state_clr = log_state - np.mean(log_state, axis=1, keepdims=True)

            # 4. Draw paired symmetric dynamics
            sym_deltas = sample_shared_symmetric_dynamics(eligible_trans, samples, dyn_seed)

            # 5. Base CLR sample matrix (state_plus_dynamics)
            base_clr_matrix = state_clr + sym_deltas

            # 6. Apply all 4 election layer variants to the exact same base matrix
            variant_sample_matrices = apply_election_layer_variants(
                base_clr_matrix=base_clr_matrix,
                training_pool=training_pool,
                samples_count=samples,
                seed=layer_seed,
            )

            cases_evaluated_count += 1

            for var_id in variants:
                samples_mat = variant_sample_matrices[var_id]

                for idx, party in enumerate(ALL_CATEGORIES):
                    actual_val = target_comp[party]
                    party_samples = samples_mat[:, idx]
                    sorted_samples = np.sort(party_samples)

                    # Quantiles and intervals
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

                    inside_50 = p25 <= actual_val <= p75
                    inside_80 = p10 <= actual_val <= p90
                    inside_90 = p05 <= actual_val <= p95

                    w50 = p75 - p25
                    w80 = p90 - p10
                    w90 = p95 - p05

                    row_dict = {
                        "election_date": election_date.isoformat(),
                        "election_year": election_date.year,
                        "origin_date": origin_date.isoformat(),
                        "horizon_days": h,
                        "variant": var_id,
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
                        "inside_50": inside_50,
                        "inside_80": inside_80,
                        "inside_90": inside_90,
                        "width_50": round(w50, 4),
                        "width_80": round(w80, 4),
                        "width_90": round(w90, 4),
                        "training_elections_count": len(training_pool.training_years),
                        "training_elections": list(training_pool.training_years),
                    }
                    rows.append(row_dict)

    df_cases = pd.DataFrame(rows)

    # Aggregations
    def summarize_variant_group(df_sub: pd.DataFrame) -> dict[str, Any]:
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
            "coverage_50": float(df_sub["inside_50"].mean()),
            "coverage_80": float(df_sub["inside_80"].mean()),
            "coverage_90": float(df_sub["inside_90"].mean()),
            "mean_width_50": float(df_sub["width_50"].mean()),
            "mean_width_80": float(df_sub["width_80"].mean()),
            "mean_width_90": float(df_sub["width_90"].mean()),
        }

    by_variant_overall = []
    by_election_variant = []
    by_horizon_variant = []
    by_party_variant = []

    for v in variants:
        sub_v = df_cases[df_cases["variant"] == v]
        if not sub_v.empty:
            v_dict = summarize_variant_group(sub_v)
            v_dict["variant"] = v
            by_variant_overall.append(v_dict)

        for e_date in sorted(df_cases["election_date"].unique()):
            sub_ev = df_cases[(df_cases["variant"] == v) & (df_cases["election_date"] == e_date)]
            if not sub_ev.empty:
                ev_dict = summarize_variant_group(sub_ev)
                ev_dict["variant"] = v
                ev_dict["election_date"] = e_date
                by_election_variant.append(ev_dict)

        for h in horizons:
            sub_hv = df_cases[(df_cases["variant"] == v) & (df_cases["horizon_days"] == h)]
            if not sub_hv.empty:
                hv_dict = summarize_variant_group(sub_hv)
                hv_dict["variant"] = v
                hv_dict["horizon_days"] = h
                by_horizon_variant.append(hv_dict)

        for p in ALL_CATEGORIES:
            sub_pv = df_cases[(df_cases["variant"] == v) & (df_cases["party"] == p)]
            if not sub_pv.empty:
                by_party_variant.append({
                    "variant": v,
                    "party": p,
                    "is_parliamentary": p in PARTIES,
                    "MAE": float(sub_pv["absolute_error"].mean()),
                    "CRPS": float(sub_pv["crps"].mean()),
                    "coverage_50": float(sub_pv["inside_50"].mean()),
                    "coverage_80": float(sub_pv["inside_80"].mean()),
                    "coverage_90": float(sub_pv["inside_90"].mean()),
                    "mean_width_50": float(sub_pv["width_50"].mean()),
                    "mean_width_80": float(sub_pv["width_80"].mean()),
                    "mean_width_90": float(sub_pv["width_90"].mean()),
                    "avg_percentile": float(sub_pv["actual_percentile"].mean()),
                })

    df_by_variant_overall = pd.DataFrame(by_variant_overall)
    df_by_election_variant = pd.DataFrame(by_election_variant)
    df_by_horizon_variant = pd.DataFrame(by_horizon_variant)
    df_by_party_variant = pd.DataFrame(by_party_variant)

    cases_csv_path = out_dir / "election_layer_cases_2018_2022.csv"
    json_summary_path = out_dir / "election_layer_summary_2018_2022.json"
    election_csv_path = out_dir / "election_layer_by_election_model.csv"
    horizon_csv_path = out_dir / "election_layer_by_horizon_model.csv"
    party_csv_path = out_dir / "election_layer_by_party_model.csv"

    df_cases.to_csv(cases_csv_path, index=False)
    df_by_election_variant.to_csv(election_csv_path, index=False)
    df_by_horizon_variant.to_csv(horizon_csv_path, index=False)
    df_by_party_variant.to_csv(party_csv_path, index=False)

    summary_data = {
        "elections": [e.isoformat() for e in elections],
        "horizons": list(horizons),
        "variants": list(variants),
        "samples": samples,
        "base_seed": seed,
        "forecast_cases_per_variant": cases_evaluated_count,
        "party_rows_per_variant": cases_evaluated_count * len(ALL_CATEGORIES),
        "total_party_rows": len(df_cases),
        "by_variant_overall": df_by_variant_overall.to_dict(orient="records"),
        "by_election_variant": df_by_election_variant.to_dict(orient="records"),
        "by_horizon_variant": df_by_horizon_variant.to_dict(orient="records"),
        "by_party_variant": df_by_party_variant.to_dict(orient="records"),
    }

    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "cases_df": df_cases,
        "summary": summary_data,
        "paths": {
            "cases_csv": str(cases_csv_path),
            "summary_json": str(json_summary_path),
            "election_csv": str(election_csv_path),
            "horizon_csv": str(horizon_csv_path),
            "party_csv": str(party_csv_path),
        },
    }


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Election Result Layer v1 evaluation on Swedish general elections."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Monte Carlo sample count (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed (default: 12345).",
    )

    args = parser.parse_args(args_list)

    print(">>> Running Election Result Layer v1 hindcasts across 2018 and 2022 ...")
    res = run_election_layer_hindcasts(
        samples=args.samples,
        seed=args.seed,
    )

    summ = res["summary"]
    print("\n==========================================================================================")
    print("OVERALL ELECTION LAYER COMPARISON (2018 + 2022)")
    print("==========================================================================================")
    print("Variant          |  MAE(8p)  | CRPS(8p) | CRPS(all9) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
    print("-----------------+-----------+----------+------------+---------------+---------------+---------------")
    for r in summ["by_variant_overall"]:
        print(
            f"{r['variant']:<16s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} |   {r['mean_CRPS_all9']:>8.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    print("\n==========================================================================================")
    print("ELECTION LAYER COMPARISON BY ELECTION")
    print("==========================================================================================")
    print("Election   | Variant          |  MAE(8p)  | CRPS(8p) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
    print("-----------+------------------+-----------+----------+---------------+---------------+---------------")
    for r in summ["by_election_variant"]:
        print(
            f"{r['election_date']:<10s} | {r['variant']:<16s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
