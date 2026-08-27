"""Probabilistic, leakage-safe historical backtesting framework for election forecasting."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .backtest_context import ForecastContext
from .backtest_metrics import calculate_crps, calculate_interval_metrics, calculate_point_error
from .backtest_models import MODELS, ForecastDistribution, ForecastModel
from .clr import composition_to_clr
from .normalize import parse_date
from .state import OpinionState, estimate_opinion, load_timeseries_dataset
from .state_config import ALL_CATEGORIES, PARTIES, REFERENCE_CATEGORY
from .transitions import (
    MIN_TRANSITIONS,
    HistoricalTransition,
    build_all_historical_transitions,
    filter_transitions_as_of,
    summarize_transition_pool,
)


DEFAULT_HORIZONS_DAYS: tuple[int, ...] = (7, 14, 28, 56, 84, 112)
DEFAULT_ORIGIN_STEP_DAYS: int = 7
DEFAULT_SAMPLES_COUNT: int = 5_000
DEFAULT_BASE_SEED: int = 12345


def get_calendar_year_block(origin_date: date) -> str:
    """Classify forecast origin into calendar year blocks."""
    if origin_date.year == 2026:
        return "2026 YTD"
    return str(origin_date.year)


def get_temporal_split(origin_date: date) -> str:
    """Classify forecast origin into fixed temporal evaluation splits (for backwards compatibility)."""
    if origin_date <= date(2022, 12, 31):
        return "Development"
    elif origin_date <= date(2023, 12, 31):
        return "Validation"
    else:
        return "Holdout"



def derive_forecast_seed(base_seed: int, model_id: str, origin_date: date, horizon_days: int) -> int:
    """Derive a stable, deterministic integer seed per forecast case using SHA-256."""
    token = f"{base_seed}:{model_id}:{origin_date.isoformat()}:{horizon_days}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def derive_origin_seed(base_seed: int, model_id: str, origin_date: date) -> int:
    """Derive a stable, deterministic integer seed per origin using SHA-256."""
    return derive_forecast_seed(base_seed, model_id, origin_date, 0)


def generate_forecast_origins(
    start_date: date,
    end_date: date,
    step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
) -> list[date]:
    """Generate sequential forecast origin dates from start_date to end_date."""
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date})")
    origins: list[date] = []
    current = start_date
    while current <= end_date:
        origins.append(current)
        current += timedelta(days=step_days)
    return origins


def run_backtest(
    model: str | ForecastModel = "symmetric_all_history",
    start_date: str | date = "2019-01-01",
    end_date: str | date = "2026-08-23",
    horizons: Sequence[int] = DEFAULT_HORIZONS_DAYS,
    origin_step_days: int = DEFAULT_ORIGIN_STEP_DAYS,
    samples: int = DEFAULT_SAMPLES_COUNT,
    seed: int = DEFAULT_BASE_SEED,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute historical backtesting run across specified horizons and origins."""
    model_obj: ForecastModel = MODELS[model] if isinstance(model, str) else model
    d_start = parse_date(start_date) if isinstance(start_date, str) else start_date
    d_end = parse_date(end_date) if isinstance(end_date, str) else end_date

    base_path = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data" / "processed" / "pollofpolls"
    ts_file = base_path / "pollofpolls_timeseries.csv"

    timeseries_data = load_timeseries_dataset(ts_file)
    ts_by_date = {row["date"]: row for row in timeseries_data}

    # Pre-build all historical transitions across horizons
    all_transitions_by_horizon = build_all_historical_transitions(timeseries_data, horizons=horizons)

    origins = generate_forecast_origins(d_start, d_end, step_days=origin_step_days)

    rows: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    skipped_origins: list[dict[str, Any]] = []
    evaluated_cases_count = 0

    for origin_date in origins:
        if origin_date not in ts_by_date:
            skipped_origins.append({
                "origin_date": origin_date.isoformat(),
                "reason": "origin_date_not_in_timeseries",
            })
            continue

        origin_row = ts_by_date[origin_date]
        origin_pop = origin_row["composition"]
        origin_clr, _ = composition_to_clr(origin_pop)

        op_state: OpinionState | None = None
        if model_obj.model_id == "no_change":
            try:
                op_state = estimate_opinion(as_of=origin_date, data_dir=base_path)
            except Exception as err:
                skipped_origins.append({
                    "origin_date": origin_date.isoformat(),
                    "reason": f"state_estimation_failed: {err}",
                })
                continue

        # Structural leakage boundary: filter transitions ending on or before origin_date
        eligible_transitions = {
            h: filter_transitions_as_of(all_transitions_by_horizon[h], origin_date)
            for h in horizons
        }

        context = ForecastContext(
            origin_date=origin_date,
            opinion_state=op_state,
            origin_pop=origin_pop,
            origin_clr=origin_clr,
            eligible_transitions_by_horizon=eligible_transitions,
            data_dir=base_path,
        )

        year_block = get_calendar_year_block(origin_date)

        # For static origin-only models (no_change, point_persistence), generate distribution once per origin
        static_dist: ForecastDistribution | None = None
        if model_obj.model_id in ("no_change", "point_persistence"):
            origin_seed = derive_origin_seed(seed, model_obj.model_id, origin_date)
            try:
                static_dist = model_obj.forecast(
                    context=context,
                    horizon_days=horizons[0],
                    samples_count=samples,
                    seed=origin_seed,
                )
            except Exception as err:
                skipped_origins.append({
                    "origin_date": origin_date.isoformat(),
                    "reason": f"origin_forecast_failed: {err}",
                })
                continue

        for h in horizons:
            target_date = origin_date + timedelta(days=h)
            if target_date not in ts_by_date:
                skipped_cases.append({
                    "model": model_obj.model_id,
                    "origin_date": origin_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "horizon_days": h,
                    "reason": "missing_exact_target_observation",
                })
                continue

            if static_dist is not None:
                dist = static_dist
                case_seed = static_dist.seed
            else:
                case_seed = derive_forecast_seed(seed, model_obj.model_id, origin_date, h)
                try:
                    dist = model_obj.forecast(
                        context=context,
                        horizon_days=h,
                        samples_count=samples,
                        seed=case_seed,
                    )
                except Exception as err:
                    skipped_cases.append({
                        "model": model_obj.model_id,
                        "origin_date": origin_date.isoformat(),
                        "target_date": target_date.isoformat(),
                        "horizon_days": h,
                        "reason": f"forecast_failed: {err}",
                    })
                    continue

            target_row = ts_by_date[target_date]
            actual_comp = target_row["composition"]
            evaluated_cases_count += 1

            diag = dist.diagnostics

            for party in ALL_CATEGORIES:
                point_fc = dist.point_forecast[party]  # Predictive P50 (median)
                actual_val = actual_comp[party]
                party_samples = dist.samples_by_party[party]
                q_dict = dist.quantiles_by_party[party]

                pt_err = calculate_point_error(point_fc, actual_val)
                iv_metrics = calculate_interval_metrics(q_dict, actual_val)
                crps_val = calculate_crps(
                    party_samples, actual_val, precomputed_sample_term=dist.crps_sample_terms[party]
                )

                row_dict = {
                    "model": model_obj.model_id,
                    "origin_date": origin_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "horizon_days": h,
                    "year_block": year_block,
                    "party": party,
                    "is_parliamentary": party in PARTIES,
                    "point_forecast": round(point_fc, 4),
                    "predictive_mean": round(dist.predictive_mean[party], 4),
                    "actual": round(actual_val, 4),
                    "error": round(pt_err["error"], 4),
                    "absolute_error": round(pt_err["absolute_error"], 4),
                    "squared_error": round(pt_err["squared_error"], 6),
                    "p05": round(iv_metrics["p05"], 4),
                    "p10": round(iv_metrics["p10"], 4),
                    "p25": round(iv_metrics["p25"], 4),
                    "p50": round(iv_metrics["p50"], 4),
                    "p75": round(iv_metrics["p75"], 4),
                    "p90": round(iv_metrics["p90"], 4),
                    "p95": round(iv_metrics["p95"], 4),
                    "interval50_contains_actual": iv_metrics["interval50_contains_actual"],
                    "interval80_contains_actual": iv_metrics["interval80_contains_actual"],
                    "interval90_contains_actual": iv_metrics["interval90_contains_actual"],
                    "width_50": round(iv_metrics["width_50"], 4),
                    "width_80": round(iv_metrics["width_80"], 4),
                    "width_90": round(iv_metrics["width_90"], 4),
                    "crps": round(crps_val, 6),
                    "samples_count": samples,
                    "seed": case_seed,
                    "eligible_transition_count": diag.get("eligible_transition_count", 0),
                    "earliest_transition_end": diag.get("earliest_transition_end", ""),
                    "latest_transition_end": diag.get("latest_transition_end", ""),
                    "weighted_mean_age_days": diag.get("weighted_mean_age_days", 0.0),
                    "kish_effective_transition_count": diag.get("kish_effective_transition_count", 0.0),
                    "origin_estimate_date": op_state.estimate_date.isoformat() if op_state else origin_date.isoformat(),
                    "origin_estimate_age_days": op_state.estimate_age_days if op_state else 0,
                }
                rows.append(row_dict)

    df_results = pd.DataFrame(rows)

    # Aggregations helper
    def compute_group_metrics(group_df: pd.DataFrame) -> dict[str, Any]:
        unique_cases = len(group_df[["origin_date", "horizon_days"]].drop_duplicates())
        sub_parl = group_df[group_df["is_parliamentary"]]
        sub_rest = group_df[group_df["party"] == REFERENCE_CATEGORY]

        return {
            "forecast_cases": unique_cases,
            "N": len(group_df),
            "MAE": float(group_df["absolute_error"].mean()),
            "RMSE": float(np.sqrt(group_df["squared_error"].mean())),
            "mean_CRPS_8parties": float(sub_parl["crps"].mean()) if not sub_parl.empty else 0.0,
            "mean_CRPS_all9": float(group_df["crps"].mean()),
            "REST_CRPS": float(sub_rest["crps"].mean()) if not sub_rest.empty else 0.0,
            "coverage_50": float(group_df["interval50_contains_actual"].mean()),
            "coverage_80": float(group_df["interval80_contains_actual"].mean()),
            "coverage_90": float(group_df["interval90_contains_actual"].mean()),
            "mean_width_50": float(group_df["width_50"].mean()),
            "mean_width_80": float(group_df["width_80"].mean()),
            "mean_width_90": float(group_df["width_90"].mean()),
        }

    horizon_summaries = []
    year_horizon_summaries = []
    year_summaries = []

    if not df_results.empty:
        # By Horizon overall
        for h in horizons:
            sub = df_results[df_results["horizon_days"] == h]
            if not sub.empty:
                m_dict = compute_group_metrics(sub)
                m_dict["model"] = model_obj.model_id
                m_dict["horizon_days"] = h
                horizon_summaries.append(m_dict)

        # By Year Block x Horizon
        all_years = sorted(df_results["year_block"].unique())
        for y_block in all_years:
            # Year overall
            sub_y = df_results[df_results["year_block"] == y_block]
            if not sub_y.empty:
                y_dict = compute_group_metrics(sub_y)
                y_dict["model"] = model_obj.model_id
                y_dict["year_block"] = y_block
                year_summaries.append(y_dict)

            # Year x Horizon
            for h in horizons:
                sub_yh = df_results[(df_results["year_block"] == y_block) & (df_results["horizon_days"] == h)]
                if not sub_yh.empty:
                    yh_dict = compute_group_metrics(sub_yh)
                    yh_dict["model"] = model_obj.model_id
                    yh_dict["year_block"] = y_block
                    yh_dict["horizon_days"] = h
                    year_horizon_summaries.append(yh_dict)

    df_by_horizon = pd.DataFrame(horizon_summaries)
    df_by_year_horizon = pd.DataFrame(year_horizon_summaries)
    df_by_year = pd.DataFrame(year_summaries)

    # Equal-weighted annual average across calendar years
    equal_weight_annual_crps_8p = float(df_by_year["mean_CRPS_8parties"].mean()) if not df_by_year.empty else 0.0
    equal_weight_annual_crps_all9 = float(df_by_year["mean_CRPS_all9"].mean()) if not df_by_year.empty else 0.0

    # Save output
    out_path = Path(output_dir) if output_dir else base_path.parents[0] / "backtests"
    out_path.mkdir(parents=True, exist_ok=True)

    csv_name = f"backtest_cases_{model_obj.model_id}_{d_start.isoformat()}_{d_end.isoformat()}.csv"
    summary_json_name = f"backtest_summary_{model_obj.model_id}_{d_start.isoformat()}_{d_end.isoformat()}.json"
    horizon_csv_name = f"backtest_by_horizon_{model_obj.model_id}_{d_start.isoformat()}_{d_end.isoformat()}.csv"
    year_horizon_csv_name = f"backtest_by_year_horizon_{model_obj.model_id}_{d_start.isoformat()}_{d_end.isoformat()}.csv"

    df_results.to_csv(out_path / csv_name, index=False)
    if not df_by_horizon.empty:
        df_by_horizon.to_csv(out_path / horizon_csv_name, index=False)
    if not df_by_year_horizon.empty:
        df_by_year_horizon.to_csv(out_path / year_horizon_csv_name, index=False)

    summary_data = {
        "model": model_obj.model_id,
        "start_date": d_start.isoformat(),
        "end_date": d_end.isoformat(),
        "horizons": list(horizons),
        "origin_step_days": origin_step_days,
        "samples": samples,
        "base_seed": seed,
        "total_origins_generated": len(origins),
        "skipped_origins_count": len(skipped_origins),
        "evaluated_cases_count": evaluated_cases_count,
        "skipped_cases_count": len(skipped_cases),
        "total_rows_evaluated": len(df_results),
        "equal_weight_annual_crps_8parties": round(equal_weight_annual_crps_8p, 4),
        "equal_weight_annual_crps_all9": round(equal_weight_annual_crps_all9, 4),
        "summary_by_horizon": df_by_horizon.to_dict(orient="records") if not df_by_horizon.empty else [],
        "summary_by_year": df_by_year.to_dict(orient="records") if not df_by_year.empty else [],
        "summary_by_year_horizon": df_by_year_horizon.to_dict(orient="records") if not df_by_year_horizon.empty else [],
        "skipped_cases_sample": skipped_cases[:20],
        "skipped_origins": skipped_origins,
    }

    with (out_path / summary_json_name).open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "results_df": df_results,
        "by_horizon_df": df_by_horizon,
        "by_year_horizon_df": df_by_year_horizon,
        "by_year_df": df_by_year,
        "summary": summary_data,
        "output_files": {
            "cases_csv": str(out_path / csv_name),
            "summary_json": str(out_path / summary_json_name),
            "horizon_csv": str(out_path / horizon_csv_name),
            "year_horizon_csv": str(out_path / year_horizon_csv_name),
        },
    }


def compute_paired_crps_comparison(
    df_raw: pd.DataFrame,
    df_symmetric: pd.DataFrame,
) -> dict[str, Any]:
    """Compute exact paired delta-CRPS = CRPS_raw - CRPS_symmetric on inner-joined cases."""
    join_keys = ["origin_date", "horizon_days", "party"]
    merged = pd.merge(
        df_raw[join_keys + ["year_block", "is_parliamentary", "crps"]],
        df_symmetric[join_keys + ["crps"]],
        on=join_keys,
        suffixes=("_raw", "_symmetric"),
    )
    merged["delta_crps"] = merged["crps_raw"] - merged["crps_symmetric"]

    # Filter to 8 parliamentary parties for primary comparison
    merged_8p = merged[merged["is_parliamentary"]]

    by_year = (
        merged_8p.groupby("year_block")["delta_crps"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .to_dict(orient="records")
    )
    by_horizon = (
        merged_8p.groupby("horizon_days")["delta_crps"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .to_dict(orient="records")
    )
    by_party = (
        merged.groupby("party")["delta_crps"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .to_dict(orient="records")
    )
    by_year_horizon = (
        merged_8p.groupby(["year_block", "horizon_days"])["delta_crps"]
        .agg(["count", "mean"])
        .reset_index()
        .to_dict(orient="records")
    )

    overall_8p_mean = float(merged_8p["delta_crps"].mean())
    overall_all9_mean = float(merged["delta_crps"].mean())

    return {
        "matched_cases_count": len(merged) // len(ALL_CATEGORIES),
        "overall_delta_crps_8parties": round(overall_8p_mean, 6),
        "overall_delta_crps_all9": round(overall_all9_mean, 6),
        "by_year": by_year,
        "by_horizon": by_horizon,
        "by_party": by_party,
        "by_year_horizon": by_year_horizon,
    }


def main(args_list: Sequence[str] | None = None) -> int:
    """CLI entry point for running probabilistic backtests."""
    parser = argparse.ArgumentParser(
        description="Run probabilistic historical backtesting for election forecasting models."
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="point_persistence,empirical_raw,symmetric_all_history,symmetric_4y,symmetric_2y,symmetric_recency_weighted",
        help="Comma-separated forecast model identifiers.",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        default="2019-01-01",
        help="Earliest forecast origin date YYYY-MM-DD (default: 2019-01-01).",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        default="2026-08-23",
        help="Latest forecast origin date YYYY-MM-DD (default: 2026-08-23).",
    )
    parser.add_argument(
        "--horizons",
        dest="horizons",
        default="7,14,28,56,84,112",
        help="Comma-separated forecast horizons in days (default: 7,14,28,56,84,112).",
    )
    parser.add_argument(
        "--step",
        dest="step_days",
        type=int,
        default=7,
        help="Origin step in days (default: 7).",
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        type=int,
        default=5_000,
        help="Monte Carlo sample count (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=None,
        help="Custom data directory path.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output machine-readable summary JSON to stdout.",
    )

    args = parser.parse_args(args_list)

    horizons_tuple = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    models_to_run = [m.strip() for m in args.model.split(",") if m.strip()]

    all_summaries: list[dict[str, Any]] = []

    for m in models_to_run:
        if m not in MODELS:
            sys.stderr.write(f"Unknown model: {m}. Available: {list(MODELS.keys())}\n")
            return 1

        print(f"\n>>> Running backtest for model: {m} ...")
        res = run_backtest(
            model=m,
            start_date=args.start_date,
            end_date=args.end_date,
            horizons=horizons_tuple,
            origin_step_days=args.step_days,
            samples=args.samples,
            seed=args.seed,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
        all_summaries.append(res["summary"])

        if not args.json_output:
            summ = res["summary"]
            print(f"--- Summary for {m.upper()} ---")
            print(f"Equal-Weight Annual CRPS (8 parties): {summ['equal_weight_annual_crps_8parties']:.4f}")
            print("Horizon | Cases |  MAE  |  RMSE | CRPS(8p) | CRPS(9p) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
            print("--------+-------+-------+-------+----------+----------+---------------+---------------+---------------")
            for h_row in summ["summary_by_horizon"]:
                print(
                    f"{h_row['horizon_days']:>4d}d   | {h_row['forecast_cases']:>5d} | "
                    f"{h_row['MAE']:>5.2f} | {h_row['RMSE']:>5.2f} | {h_row['mean_CRPS_8parties']:>8.3f} | {h_row['mean_CRPS_all9']:>8.3f} | "
                    f"{h_row['coverage_50']:>5.1%} ({h_row['mean_width_50']:>4.2f}) | "
                    f"{h_row['coverage_80']:>5.1%} ({h_row['mean_width_80']:>4.2f}) | "
                    f"{h_row['coverage_90']:>5.1%} ({h_row['mean_width_90']:>4.2f})"
                )

    if args.json_output:
        print(json.dumps(all_summaries, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
