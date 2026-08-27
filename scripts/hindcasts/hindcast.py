"""Election Hindcast v1 execution engine and diagnostic reporter."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np
import pandas as pd

from scripts.elections.load import load_election_targets_for_forecasting
from scripts.pollofpolls.backtest_metrics import calculate_crps, precompute_crps_sample_term
from scripts.pollofpolls.state import OpinionState, estimate_opinion, load_timeseries_dataset
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES
from scripts.pollofpolls.transitions import (
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
)

from .models import (
    derive_opinion_state_seed,
    derive_shared_dynamics_seed,
    hindcast_dynamics_only,
    hindcast_point_persistence,
    hindcast_state_plus_dynamics,
    sample_shared_symmetric_dynamics,
)


EVALUATION_ELECTIONS: tuple[date, ...] = (
    date(2018, 9, 9),
    date(2022, 9, 11),
)

DEFAULT_HORIZONS: tuple[int, ...] = (112, 84, 56, 28, 14, 7)
DEFAULT_MODELS: tuple[str, ...] = ("point_persistence", "dynamics_only", "state_plus_dynamics")
DEFAULT_SAMPLES: int = 5_000
DEFAULT_SEED: int = 12345

QUANTILES_TO_TRACK = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def calculate_empirical_midrank_percentile(samples: np.ndarray, actual: float) -> float:
    """Calculate empirical mid-rank percentile: P = 100 * (#(x < y) + 0.5 * #(x == y)) / n."""
    n = len(samples)
    if n == 0:
        return 0.0
    less_count = np.sum(samples < actual)
    equal_count = np.sum(samples == actual)
    return round(float(100.0 * (less_count + 0.5 * equal_count) / n), 2)


def run_election_hindcasts(
    elections: Sequence[date] = EVALUATION_ELECTIONS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    models: Sequence[str] = DEFAULT_MODELS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute election hindcasts across requested elections, horizons, and models."""
    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    ts_file = base_path / "pollofpolls_timeseries.csv"
    elections_file = base_path.parent / "elections" / "riksdag_election_results.csv"

    timeseries_data = load_timeseries_dataset(ts_file)
    ts_by_date = {row["date"]: row for row in timeseries_data}

    # Load official election targets in 9-category forecast space
    election_targets = load_election_targets_for_forecasting(elections_file)

    # Pre-build transitions for all requested horizons
    all_transitions_by_horizon = build_all_historical_transitions(timeseries_data, horizons=horizons)

    rows: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    evaluated_cases_count = 0

    for election_date in elections:
        if election_date not in election_targets:
            raise KeyError(f"No official target results for election {election_date}")

        target_comp = election_targets[election_date]

        for h in sorted(horizons, reverse=True):
            origin_date = election_date - timedelta(days=h)

            if origin_date not in ts_by_date:
                for m in models:
                    skipped_cases.append({
                        "election_date": election_date.isoformat(),
                        "origin_date": origin_date.isoformat(),
                        "horizon_days": h,
                        "model": m,
                        "reason": "missing_exact_origin_observation_in_timeseries",
                    })
                continue

            origin_row = ts_by_date[origin_date]
            origin_pop = origin_row["composition"]

            # Filter eligible transitions (transition_end <= origin_date)
            eligible_trans = filter_transitions_as_of(all_transitions_by_horizon[h], origin_date)

            if len(eligible_trans) < 30:
                for m in models:
                    skipped_cases.append({
                        "election_date": election_date.isoformat(),
                        "origin_date": origin_date.isoformat(),
                        "horizon_days": h,
                        "model": m,
                        "reason": f"insufficient_historical_transitions ({len(eligible_trans)} < 30)",
                    })
                continue

            # Deterministic seeds
            dyn_seed = derive_shared_dynamics_seed(seed, origin_date, h)
            state_seed = derive_opinion_state_seed(seed, origin_date)

            # Sample shared symmetric CLR dynamics (identical draws for dynamics_only and state_plus_dynamics)
            sym_deltas = sample_shared_symmetric_dynamics(eligible_trans, samples, dyn_seed)

            # Estimate OpinionState v1.1 if state_plus_dynamics is included
            op_state: OpinionState | None = None
            if "state_plus_dynamics" in models:
                try:
                    op_state = estimate_opinion(as_of=origin_date, data_dir=base_path)
                except Exception as err:
                    skipped_cases.append({
                        "election_date": election_date.isoformat(),
                        "origin_date": origin_date.isoformat(),
                        "horizon_days": h,
                        "model": "state_plus_dynamics",
                        "reason": f"opinion_state_failed: {err}",
                    })

            # Generate sample matrices for requested models
            model_sample_matrices: dict[str, np.ndarray] = {}

            if "point_persistence" in models:
                model_sample_matrices["point_persistence"] = hindcast_point_persistence(
                    origin_pop=origin_pop, samples_count=samples, categories=ALL_CATEGORIES
                )

            if "dynamics_only" in models:
                model_sample_matrices["dynamics_only"] = hindcast_dynamics_only(
                    origin_pop=origin_pop, symmetric_deltas=sym_deltas, categories=ALL_CATEGORIES
                )

            if "state_plus_dynamics" in models and op_state is not None:
                model_sample_matrices["state_plus_dynamics"] = hindcast_state_plus_dynamics(
                    opinion_state=op_state,
                    symmetric_deltas=sym_deltas,
                    state_seed=state_seed,
                    samples_count=samples,
                    categories=ALL_CATEGORIES,
                )

            evaluated_cases_count += len(model_sample_matrices)

            for model_id, samples_mat in model_sample_matrices.items():
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
                        "inside_50": inside_50,
                        "inside_80": inside_80,
                        "inside_90": inside_90,
                        "width_50": round(w50, 4),
                        "width_80": round(w80, 4),
                        "width_90": round(w90, 4),
                        "samples_count": samples,
                        "eligible_transition_count": len(eligible_trans),
                    }
                    rows.append(row_dict)

    df_cases = pd.DataFrame(rows)

    # Aggregations helper
    def summarize_group(df_sub: pd.DataFrame) -> dict[str, Any]:
        sub_8p = df_sub[df_sub["is_parliamentary"]]
        sub_rest = df_sub[df_sub["party"] == "REST"]
        return {
            "cases_count": len(df_sub[["election_date", "horizon_days"]].drop_duplicates()),
            "rows_count": len(df_sub),
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

    # Groupings
    by_election_model = []
    by_horizon_model = []
    by_party_model = []
    by_model_overall = []

    if not df_cases.empty:
        for m in models:
            sub_m = df_cases[df_cases["model"] == m]
            if not sub_m.empty:
                s_dict = summarize_group(sub_m)
                s_dict["model"] = m
                by_model_overall.append(s_dict)

            # By Election
            for e_date in sorted(df_cases["election_date"].unique()):
                sub_em = df_cases[(df_cases["model"] == m) & (df_cases["election_date"] == e_date)]
                if not sub_em.empty:
                    em_dict = summarize_group(sub_em)
                    em_dict["model"] = m
                    em_dict["election_date"] = e_date
                    by_election_model.append(em_dict)

            # By Horizon
            for h in horizons:
                sub_hm = df_cases[(df_cases["model"] == m) & (df_cases["horizon_days"] == h)]
                if not sub_hm.empty:
                    hm_dict = summarize_group(sub_hm)
                    hm_dict["model"] = m
                    hm_dict["horizon_days"] = h
                    by_horizon_model.append(hm_dict)

            # By Party
            for p in ALL_CATEGORIES:
                sub_pm = df_cases[(df_cases["model"] == m) & (df_cases["party"] == p)]
                if not sub_pm.empty:
                    pm_dict = {
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
                    }
                    by_party_model.append(pm_dict)

    df_by_election_model = pd.DataFrame(by_election_model)
    df_by_horizon_model = pd.DataFrame(by_horizon_model)
    df_by_party_model = pd.DataFrame(by_party_model)
    df_by_model_overall = pd.DataFrame(by_model_overall)

    # Identify Diagnostic Outliers (outside 90% interval, extreme percentiles)
    outliers_outside_90 = df_cases[~df_cases["inside_90"]].to_dict(orient="records")
    extreme_percentiles = df_cases[(df_cases["actual_percentile"] < 5.0) | (df_cases["actual_percentile"] > 95.0)].to_dict(orient="records")

    # Output paths
    out_dir = Path(output_dir) if output_dir else base_path.parent / "hindcasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "hindcast_cases_2018_2022.csv"
    json_summary_path = out_dir / "hindcast_summary_2018_2022.json"
    election_csv_path = out_dir / "hindcast_by_election_model.csv"
    horizon_csv_path = out_dir / "hindcast_by_horizon_model.csv"
    party_csv_path = out_dir / "hindcast_by_party_model.csv"

    df_cases.to_csv(csv_path, index=False)
    if not df_by_election_model.empty:
        df_by_election_model.to_csv(election_csv_path, index=False)
    if not df_by_horizon_model.empty:
        df_by_horizon_model.to_csv(horizon_csv_path, index=False)
    if not df_by_party_model.empty:
        df_by_party_model.to_csv(party_csv_path, index=False)

    summary_data = {
        "elections": [e.isoformat() for e in elections],
        "horizons": list(horizons),
        "models": list(models),
        "samples": samples,
        "base_seed": seed,
        "evaluated_cases_count": evaluated_cases_count,
        "skipped_cases": skipped_cases,
        "by_model_overall": df_by_model_overall.to_dict(orient="records") if not df_by_model_overall.empty else [],
        "by_election_model": df_by_election_model.to_dict(orient="records") if not df_by_election_model.empty else [],
        "by_horizon_model": df_by_horizon_model.to_dict(orient="records") if not df_by_horizon_model.empty else [],
        "by_party_model": df_by_party_model.to_dict(orient="records") if not df_by_party_model.empty else [],
        "outliers_outside_90_count": len(outliers_outside_90),
        "extreme_percentiles_count": len(extreme_percentiles),
        "outliers_sample": outliers_outside_90[:15],
    }

    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "cases_df": df_cases,
        "summary": summary_data,
        "files": {
            "cases_csv": str(csv_path),
            "summary_json": str(json_summary_path),
            "election_csv": str(election_csv_path),
            "horizon_csv": str(horizon_csv_path),
            "party_csv": str(party_csv_path),
        },
    }


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Election Hindcast v1 runner for Swedish Riksdag elections (2018 & 2022)."
    )
    parser.add_argument(
        "--models",
        default="point_persistence,dynamics_only,state_plus_dynamics",
        help="Comma-separated model identifiers.",
    )
    parser.add_argument(
        "--horizons",
        default="112,84,56,28,14,7",
        help="Comma-separated horizons in days.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5_000,
        help="Monte Carlo sample count (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )

    args = parser.parse_args(args_list)
    models_list = [m.strip() for m in args.models.split(",") if m.strip()]
    horizons_list = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]

    print(">>> Running Election Hindcast v1 for 2018 and 2022 ...")
    res = run_election_hindcasts(
        horizons=horizons_list,
        models=models_list,
        samples=args.samples,
        seed=args.seed,
    )

    summ = res["summary"]
    print("\n==========================================================================================")
    print("OVERALL HINDCAST SUMMARY (2018 + 2022)")
    print("==========================================================================================")
    print("Model                |  MAE(8p)  | CRPS(8p) | CRPS(all9) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
    print("---------------------+-----------+----------+------------+---------------+---------------+---------------")
    for r in summ["by_model_overall"]:
        print(
            f"{r['model']:<20s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} |   {r['mean_CRPS_all9']:>8.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    print("\n==========================================================================================")
    print("HINDCAST SUMMARY BY ELECTION")
    print("==========================================================================================")
    print("Election   | Model                |  MAE(8p)  | CRPS(8p) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
    print("-----------+----------------------+-----------+----------+---------------+---------------+---------------")
    for r in summ["by_election_model"]:
        print(
            f"{r['election_date']:<10s} | {r['model']:<20s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
