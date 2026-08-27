"""Complete historical probabilistic Riksdag seat hindcast pipeline (SeatHindcast v1)."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import pandas as pd

from scripts.simulator.config import PARLIAMENTARY_PARTIES_8
from .config import (
    DEFAULT_HORIZONS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    EVALUATION_ELECTIONS,
)
from .diagnostics import calculate_seat_uncertainty_diagnostics
from .metrics import (
    calculate_discrete_seat_crps,
    calculate_empirical_percentile,
    calculate_interval_coverage_and_width,
    calculate_multivariate_energy_score,
)
from .models import (
    evaluate_election_simulator_v1,
    evaluate_seat_point_baseline,
)

DEFAULT_SEEDS: tuple[int, ...] = (12345, 24680, 98765, 54321, 13579)


def run_seat_hindcast_single_seed(
    seed: int,
    samples: int = DEFAULT_SAMPLES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    geography_mode: str = "chronological",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute complete 2018 and 2022 seat hindcasts for a single fixed seed."""
    all_results = []
    party_summaries = []

    for year_str, e_info in EVALUATION_ELECTIONS.items():
        elec_date = e_info["election_date"]
        base_geo_year = e_info["geography_baseline_year"]
        actual_seats = e_info["actual_seats"]
        actual_seat_vec = np.array([actual_seats[p] for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)

        for h in horizons:
            as_of = elec_date - timedelta(days=h)

            # 1. Point Baseline Evaluation
            point_seats = evaluate_seat_point_baseline(
                as_of=as_of,
                election_date=elec_date,
                baseline_year=base_geo_year,
                geography_mode=geography_mode,
            )
            point_seat_vec = np.array([point_seats.get(p, 0) for p in PARLIAMENTARY_PARTIES_8], dtype=np.int64)
            point_mae = float(np.mean(np.abs(point_seat_vec - actual_seat_vec)))
            # For a deterministic point distribution, CRPS is identically the absolute error
            baseline_crps = point_mae

            # 2. Simulator v1 Evaluation
            sim_res = evaluate_election_simulator_v1(
                as_of=as_of,
                election_date=elec_date,
                baseline_year=base_geo_year,
                samples=samples,
                seed=seed,
                geography_mode=geography_mode,
            )

            seats_matrix = sim_res.seats_matrix  # shape (N, 8)
            vote_shares_matrix = sim_res.vote_shares_matrix  # shape (N, 9)

            # 3. 8-Party Joint Energy Score
            es = calculate_multivariate_energy_score(seats_matrix, actual_seat_vec)
            # A deterministic point forecast is a degenerate distribution, so
            # its multivariate Energy Score is exactly the Euclidean seat error.
            baseline_energy_score = float(np.linalg.norm(point_seat_vec - actual_seat_vec))

            # 4. Per-Party Seat Metrics
            party_metrics = {}
            median_seats_vec = np.zeros(8, dtype=np.int64)
            mean_seats_vec = np.zeros(8, dtype=np.float64)

            for p_idx, p in enumerate(PARLIAMENTARY_PARTIES_8):
                p_draws = seats_matrix[:, p_idx]
                act_s = actual_seats[p]

                mean_s = float(np.mean(p_draws))
                median_s = int(np.median(p_draws))
                mean_seats_vec[p_idx] = mean_s
                median_seats_vec[p_idx] = median_s

                crps_s = calculate_discrete_seat_crps(p_draws, act_s)
                perc_s = calculate_empirical_percentile(p_draws, act_s)

                cov50, w50, _, _ = calculate_interval_coverage_and_width(p_draws, act_s, level=0.50)
                cov80, w80, _, _ = calculate_interval_coverage_and_width(p_draws, act_s, level=0.80)
                cov90, w90, _, _ = calculate_interval_coverage_and_width(p_draws, act_s, level=0.90)

                party_metrics[p] = {
                    "party": p,
                    "actual_seats": act_s,
                    "point_baseline_seats": int(point_seat_vec[p_idx]),
                    "mean_seats": round(mean_s, 2),
                    "median_seats": median_s,
                    "point_error": int(abs(point_seat_vec[p_idx] - act_s)),
                    "mean_error": round(float(abs(mean_s - act_s)), 2),
                    "median_error": int(abs(median_s - act_s)),
                    "crps": round(crps_s, 4),
                    "percentile": round(perc_s, 1),
                    "cov_50": cov50,
                    "width_50": w50,
                    "cov_80": cov80,
                    "width_80": w80,
                    "cov_90": cov90,
                    "width_90": w90,
                    "p_largest": float(np.mean(np.array(sim_res.largest_seat_parties) == p)),
                    "p_qualify": float(np.mean(vote_shares_matrix[:, p_idx] >= 4.0)),
                    "p_any_seats": float(np.mean(p_draws > 0)),
                }

                party_summaries.append({
                    "seed": seed,
                    "election_year": int(year_str),
                    "horizon_days": h,
                    "as_of": as_of.isoformat(),
                    **party_metrics[p],
                })

            sim_median_mae = float(np.mean(np.abs(median_seats_vec - actual_seat_vec)))
            sim_mean_mae = float(np.mean(np.abs(mean_seats_vec - actual_seat_vec)))
            sim_mean_crps = float(np.mean([m["crps"] for m in party_metrics.values()]))

            # 5. Group Majorities
            tido_summary = sim_res.summarize_group(["M", "SD", "KD", "L"])
            rgc_summary = sim_res.summarize_group(["S", "V", "MP", "C"])

            # 6. Uncertainty Diagnostics
            uncertainty_diag = calculate_seat_uncertainty_diagnostics(vote_shares_matrix, seats_matrix)

            case_record = {
                "seed": seed,
                "model_version": sim_res.manifest.get("model_version"),
                "source_git_commit": sim_res.manifest.get("source_git_commit"),
                "source_worktree_clean": sim_res.manifest.get("source_worktree_clean"),
                "input_hashes": {
                    key: sim_res.manifest.get(key)
                    for key in ("poll_data_hash", "election_data_hash", "mandate_data_hash", "geography_data_hash")
                },
                "model_config_hash": sim_res.manifest.get("model_config_hash"),
                "election_year": int(year_str),
                "election_date": elec_date.isoformat(),
                "horizon_days": h,
                "as_of": as_of.isoformat(),
                "geography_baseline_year": base_geo_year,
                "geography_mode": geography_mode,
                "point_baseline_mae": round(point_mae, 3),
                "baseline_crps": round(baseline_crps, 3),
                "simulator_median_mae": round(sim_median_mae, 3),
                "simulator_mean_mae": round(sim_mean_mae, 3),
                "simulator_mean_crps": round(sim_mean_crps, 4),
                "baseline_energy_score": round(baseline_energy_score, 4),
                "joint_energy_score": round(es, 4),
                "party_metrics": party_metrics,
                "bloc_majorities": {
                    "tido_mean_seats": round(tido_summary.mean_seats, 2),
                    "tido_prob_majority": round(tido_summary.prob_majority, 4),
                    "rgc_mean_seats": round(rgc_summary.mean_seats, 2),
                    "rgc_prob_majority": round(rgc_summary.prob_majority, 4),
                },
                "uncertainty_diagnostics": uncertainty_diag,
            }
            all_results.append(case_record)

    return all_results, party_summaries


def run_multi_seed_seat_hindcasts(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    samples: int = DEFAULT_SAMPLES,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Execute complete multi-seed historical seat hindcast evaluation."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================================")
    print("RUNNING MULTI-SEED HISTORICAL SEAT HINDCASTS (SeatHindcast v1)")
    print(f"Seeds: {seeds} | Samples per run: {samples:,} | Horizons: {horizons}")
    print("==========================================================================================")

    t_start = time.perf_counter()

    all_seed_results = []
    all_party_summaries = []

    for s in seeds:
        print(f"\n>>> Running Hindcasts for Seed {s} ...")
        t_s = time.perf_counter()
        cases, party_records = run_seat_hindcast_single_seed(
            seed=s,
            samples=samples,
            horizons=horizons,
            geography_mode="chronological",
        )
        all_seed_results.extend(cases)
        all_party_summaries.extend(party_records)
        print(f"  Seed {s} completed in {time.perf_counter() - t_s:.2f} s")

    total_time = time.perf_counter() - t_start
    print(f"\n==========================================================================================")
    print(f"ALL {len(seeds)} SEEDS ({len(all_seed_results)} CASES) COMPLETED IN {total_time:.2f} s")
    print(f"==========================================================================================")

    df_cases = pd.DataFrame(all_seed_results)
    df_parties = pd.DataFrame(all_party_summaries)

    # Compute per-seed and overall aggregate metrics
    seed_stability_table = []
    for s in seeds:
        sub = df_cases[df_cases["seed"] == s]
        seed_stability_table.append({
            "seed": s,
            "baseline_crps": round(float(sub["baseline_crps"].mean()), 4),
            "baseline_energy_score": round(float(sub["baseline_energy_score"].mean()), 4),
            "simulator_crps": round(float(sub["simulator_mean_crps"].mean()), 4),
            "energy_score": round(float(sub["joint_energy_score"].mean()), 4),
            "simulator_median_mae": round(float(sub["simulator_median_mae"].mean()), 3),
            "simulator_mean_mae": round(float(sub["simulator_mean_mae"].mean()), 3),
            "cov_50": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_50"] for p in PARLIAMENTARY_PARTIES_8]) for _, c in sub.iterrows()])), 3),
            "cov_80": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_80"] for p in PARLIAMENTARY_PARTIES_8]) for _, c in sub.iterrows()])), 3),
            "cov_90": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_90"] for p in PARLIAMENTARY_PARTIES_8]) for _, c in sub.iterrows()])), 3),
        })

    df_seed_stab = pd.DataFrame(seed_stability_table)

    # Primary Seed 12345 Detailed Breakdown
    primary_cases = [c for c in all_seed_results if c["seed"] == DEFAULT_SEED]
    df_prim = df_cases[df_cases["seed"] == DEFAULT_SEED]

    provenance_rows = [
        {
            "model_version": row.get("model_version"),
            "source_git_commit": row.get("source_git_commit"),
            "source_worktree_clean": row.get("source_worktree_clean"),
            "input_hashes": row.get("input_hashes"),
            "model_config_hash": row.get("model_config_hash"),
        }
        for row in all_seed_results
    ]
    provenance = provenance_rows[0] if provenance_rows else {}
    # The chronological hindcast deliberately uses 2014 geography for the
    # 2018 target and 2018 geography for the 2022 target.  That changes the
    # model-config hash by election, while source code, cleanliness, and raw
    # input hashes must remain identical across all cases.
    for row in provenance_rows[1:]:
        if row.get("source_git_commit") != provenance.get("source_git_commit"):
            raise RuntimeError("SeatHindcast cases were generated with inconsistent source commits")
        if row.get("source_worktree_clean") != provenance.get("source_worktree_clean"):
            raise RuntimeError("SeatHindcast cases were generated with inconsistent source cleanliness")
        if row.get("input_hashes") != provenance.get("input_hashes"):
            raise RuntimeError("SeatHindcast cases were generated with inconsistent input hashes")
    config_hashes_by_election = {
        str(year): sorted({row.get("model_config_hash") for row in all_seed_results if str(row.get("election_year")) == year})
        for year in sorted({str(row.get("election_year")) for row in all_seed_results})
    }
    if any(len(values) != 1 for values in config_hashes_by_election.values()):
        raise RuntimeError("SeatHindcast cases were generated with inconsistent config hashes within an election")

    summary_report = {
        "metadata": {
            "model": "ElectionSimulator_v1.0-rc1",
            "artifact_schema_version": "1.0",
            "interpretation": "Retrospective historical evaluation (not independent holdout validation)",
            "validation_note": (
                "The model-family choices and polling calibration used evidence from the same 2018/2022 period; "
                "coverage and horizon patterns are descriptive, not formal calibration or monotonicity claims."
            ),
            "samples_per_case": samples,
            "primary_seed": DEFAULT_SEED,
            "seeds_evaluated": list(seeds),
            "elections": [2018, 2022],
            "horizons": list(horizons),
            "geography_mode": "chronological",
            "total_evaluations": len(all_seed_results),
            "generated_at": date.today().isoformat(),
            "source_git_commit": provenance.get("source_git_commit"),
            "source_worktree_clean": provenance.get("source_worktree_clean"),
            "input_hashes": provenance.get("input_hashes"),
            "model_config_hashes_by_election": {
                year: values[0] for year, values in config_hashes_by_election.items()
            },
        },
        "multi_seed_stability": {
            "per_seed": seed_stability_table,
            "aggregated": {
                "simulator_crps_mean": round(float(df_seed_stab["simulator_crps"].mean()), 4),
                "simulator_crps_std": round(float(df_seed_stab["simulator_crps"].std()), 4),
                "simulator_crps_min": round(float(df_seed_stab["simulator_crps"].min()), 4),
                "simulator_crps_max": round(float(df_seed_stab["simulator_crps"].max()), 4),
                "energy_score_mean": round(float(df_seed_stab["energy_score"].mean()), 4),
                "energy_score_std": round(float(df_seed_stab["energy_score"].std()), 4),
                "energy_score_min": round(float(df_seed_stab["energy_score"].min()), 4),
                "energy_score_max": round(float(df_seed_stab["energy_score"].max()), 4),
                "baseline_energy_score_mean": round(float(df_seed_stab["baseline_energy_score"].mean()), 4),
                "median_mae_mean": round(float(df_seed_stab["simulator_median_mae"].mean()), 3),
                "mean_mae_mean": round(float(df_seed_stab["simulator_mean_mae"].mean()), 3),
                "cov_50_mean": round(float(df_seed_stab["cov_50"].mean()), 3),
                "cov_80_mean": round(float(df_seed_stab["cov_80"].mean()), 3),
                "cov_90_mean": round(float(df_seed_stab["cov_90"].mean()), 3),
            },
        },
        "primary_seed_performance": {
            "overall": {
                "baseline_point_mae": round(float(df_prim["point_baseline_mae"].mean()), 3),
                "baseline_crps": round(float(df_prim["baseline_crps"].mean()), 4),
                "baseline_energy_score": round(float(df_prim["baseline_energy_score"].mean()), 4),
                "simulator_median_mae": round(float(df_prim["simulator_median_mae"].mean()), 3),
                "simulator_mean_mae": round(float(df_prim["simulator_mean_mae"].mean()), 3),
                "simulator_crps": round(float(df_prim["simulator_mean_crps"].mean()), 4),
                "energy_score": round(float(df_prim["joint_energy_score"].mean()), 4),
                "coverage_50": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_50"] for p in PARLIAMENTARY_PARTIES_8]) for c in primary_cases])), 3),
                "coverage_80": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_80"] for p in PARLIAMENTARY_PARTIES_8]) for c in primary_cases])), 3),
                "coverage_90": round(float(np.mean([np.mean([c["party_metrics"][p]["cov_90"] for p in PARLIAMENTARY_PARTIES_8]) for c in primary_cases])), 3),
            },
            "by_election": {
                "2018": {
                    "baseline_point_mae": round(float(df_prim[df_prim["election_year"] == 2018]["point_baseline_mae"].mean()), 3),
                    "baseline_crps": round(float(df_prim[df_prim["election_year"] == 2018]["baseline_crps"].mean()), 4),
                    "baseline_energy_score": round(float(df_prim[df_prim["election_year"] == 2018]["baseline_energy_score"].mean()), 4),
                    "simulator_median_mae": round(float(df_prim[df_prim["election_year"] == 2018]["simulator_median_mae"].mean()), 3),
                    "simulator_mean_mae": round(float(df_prim[df_prim["election_year"] == 2018]["simulator_mean_mae"].mean()), 3),
                    "simulator_crps": round(float(df_prim[df_prim["election_year"] == 2018]["simulator_mean_crps"].mean()), 4),
                    "energy_score": round(float(df_prim[df_prim["election_year"] == 2018]["joint_energy_score"].mean()), 4),
                },
                "2022": {
                    "baseline_point_mae": round(float(df_prim[df_prim["election_year"] == 2022]["point_baseline_mae"].mean()), 3),
                    "baseline_crps": round(float(df_prim[df_prim["election_year"] == 2022]["baseline_crps"].mean()), 4),
                    "baseline_energy_score": round(float(df_prim[df_prim["election_year"] == 2022]["baseline_energy_score"].mean()), 4),
                    "simulator_median_mae": round(float(df_prim[df_prim["election_year"] == 2022]["simulator_median_mae"].mean()), 3),
                    "simulator_mean_mae": round(float(df_prim[df_prim["election_year"] == 2022]["simulator_mean_mae"].mean()), 3),
                    "simulator_crps": round(float(df_prim[df_prim["election_year"] == 2022]["simulator_mean_crps"].mean()), 4),
                    "energy_score": round(float(df_prim[df_prim["election_year"] == 2022]["joint_energy_score"].mean()), 4),
                },
            },
            "by_horizon": {
                str(h): {
                    "baseline_crps": round(float(df_prim[df_prim["horizon_days"] == h]["baseline_crps"].mean()), 4),
                    "baseline_energy_score": round(float(df_prim[df_prim["horizon_days"] == h]["baseline_energy_score"].mean()), 4),
                    "simulator_median_mae": round(float(df_prim[df_prim["horizon_days"] == h]["simulator_median_mae"].mean()), 3),
                    "simulator_mean_mae": round(float(df_prim[df_prim["horizon_days"] == h]["simulator_mean_mae"].mean()), 3),
                    "simulator_crps": round(float(df_prim[df_prim["horizon_days"] == h]["simulator_mean_crps"].mean()), 4),
                    "energy_score": round(float(df_prim[df_prim["horizon_days"] == h]["joint_energy_score"].mean()), 4),
                } for h in horizons
            },
        },
        "cases": primary_cases,
    }

    # Hash the complete deterministic report payload, excluding only metadata
    # fields that are intentionally run-specific or self-referential.  This
    # prevents a derived headline table from drifting away from stored cases.
    hash_payload = json.loads(json.dumps(summary_report))
    hash_metadata = hash_payload.get("metadata", {})
    hash_metadata.pop("generated_at", None)
    hash_metadata.pop("payload_sha256", None)
    payload_str = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    summary_report["metadata"]["payload_sha256"] = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    # Save to JSON
    json_path = out_dir / "seat_hindcast_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    # Save detailed CSV
    csv_path = out_dir / "seat_hindcast_parties_detail.csv"
    df_parties.to_csv(csv_path, index=False)

    print(f"\nSaved multi-seed seat hindcast summary to {json_path}")
    print(f"Saved party details table to {csv_path}")

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Run SeatHindcast v1 Historical Benchmarks")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Monte Carlo samples per hindcast")
    parser.add_argument("--multi-seed", action="store_true", default=True, help="Run all 5 predetermined seeds")
    args = parser.parse_args()

    run_multi_seed_seat_hindcasts(samples=args.samples)


if __name__ == "__main__":
    main()
