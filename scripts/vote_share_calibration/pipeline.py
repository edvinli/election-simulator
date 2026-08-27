"""Complete pipeline runner for Final Generic Vote-Share Calibration Experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .config import HIGH_SAMPLES, INITIAL_SAMPLES, INITIAL_SEED, STABILITY_SEEDS
from .forward_eval import run_exact_forward_evaluation
from .hindcast import run_vote_share_hindcasts
from .stability import run_multi_seed_stability_audit


def run_full_calibration_experiment(
    initial_samples: int = INITIAL_SAMPLES,
    high_samples: int = HIGH_SAMPLES,
    initial_seed: int = INITIAL_SEED,
    stability_seeds: Sequence[int] = STABILITY_SEEDS,
) -> int:
    print(">>> Step 1: Running exact standalone forward evaluation from 14d polling consensus (2010-2022) ...")
    fwd_res = run_exact_forward_evaluation()
    print(f"Forward evaluation completed! Saved to {fwd_res['paths']['cases_csv']}")

    print(f"\n>>> Step 2: Running paired full-pipeline hindcasts at N={initial_samples} (Seed {initial_seed}) ...")
    hind_res = run_vote_share_hindcasts(samples=initial_samples, seed=initial_seed)

    print(f"\n>>> Step 3: Running high-sample stability audit at N={high_samples} across seeds {list(stability_seeds)} ...")
    stab_res = run_multi_seed_stability_audit(seeds=stability_seeds, samples=high_samples)

    fwd_report = fwd_res["report"]
    hind_summ = hind_res["summary"]
    stab_report = stab_res["report"]

    print("\n==========================================================================================")
    print("1. STANDALONE FORWARD EVALUATION FROM 14-DAY POLLING CONSENSUS (2010-2022)")
    print("==========================================================================================")
    print("Model               | Overall MAE (8p) | Overall CRPS (8p) | Energy Score (All 9) | Mean Lambda")
    print("--------------------+------------------+-------------------+----------------------+------------")
    for r in fwd_report["by_model_overall"]:
        print(f"{r['model']:<19s} |      {r['overall_MAE_8parties']:>5.2f}%      |      {r['overall_CRPS_8parties']:>7.4f}      |       {r['overall_EnergyScore_all9']:>7.4f}        |   {r['overall_mean_lambda']:>6.4f}")

    print("\n------------------------------------------------------------------------------------------")
    print("STANDALONE FORWARD EVALUATION BY ELECTION")
    print("------------------------------------------------------------------------------------------")
    print("Year | Pool | Model               | MAE (8p) | CRPS (8p) | Energy Score | Lambda")
    print("-----+------+---------------------+----------+-----------+--------------+-------")
    for r in fwd_report["by_model_election"]:
        print(f"{r['election_year']} |  {r['training_pool_size']}   | {r['model']:<19s} |  {r['MAE_8parties']:>5.2f}%  |  {r['CRPS_8parties']:>7.4f}  |   {r['EnergyScore_all9']:>7.4f}    | {r['mean_lambda']:>5.3f}")

    print("\n==========================================================================================")
    print(f"2. FULL PIPELINE HINDCASTS WITH STATE + DYNAMICS (N={initial_samples}, Seed {initial_seed})")
    print("==========================================================================================")
    print("Model               |  MAE(8p)  | CRPS(8p) | Energy Score | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) | Mean Lam")
    print("--------------------+-----------+----------+--------------+---------------+---------------+---------------+---------")
    for r in hind_summ["by_model_overall"]:
        print(
            f"{r['model']:<19s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} |   {r['EnergyScore_all9']:>8.4f}   | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f}) |  {r['mean_lambda']:>6.4f}"
        )

    print("\n------------------------------------------------------------------------------------------")
    print("FULL HINDCASTS BY ELECTION (2018 vs 2022)")
    print("------------------------------------------------------------------------------------------")
    print("Election   | Model               |  MAE(8p)  | CRPS(8p) | Energy Score | Cov 90% (W90)")
    print("-----------+---------------------+-----------+----------+--------------+--------------")
    for r in hind_summ["by_election_model"]:
        print(
            f"{r['election_date']:<10s} | {r['model']:<19s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} |   {r['EnergyScore_all9']:>8.4f}   | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    print("\n==========================================================================================")
    print(f"3. MULTI-SEED STABILITY AUDIT AT N={high_samples} SAMPLES")
    print("==========================================================================================")
    print("Seed   | Model               | CRPS(8p) Overall | CRPS (2018) | CRPS (2022) | Energy Score | Cov 90% (W90)")
    print("-------+---------------------+------------------+-------------+-------------+--------------+--------------")
    for r in stab_report["by_seed_and_model"]:
        print(
            f"{r['seed']:<5d}  | {r['model']:<19s} |      {r['CRPS_8parties_overall']:>7.4f}     |   {r['CRPS_8parties_2018']:>7.4f}    |   {r['CRPS_8parties_2022']:>7.4f}    |   {r['EnergyScore_all9']:>8.4f}   | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    print("\n------------------------------------------------------------------------------------------")
    print("MULTI-SEED AVERAGE SUMMARY (ACROSS ALL SEEDS)")
    print("------------------------------------------------------------------------------------------")
    for avg in stab_report["model_averages_across_seeds"]:
        print(
            f"{avg['model']:<19s}: Mean CRPS(8p) = {avg['mean_CRPS_8parties_across_seeds']:.4f} (2018: {avg['mean_CRPS_2018_across_seeds']:.4f}, 2022: {avg['mean_CRPS_2022_across_seeds']:.4f}) | Energy Score = {avg['mean_EnergyScore_across_seeds']:.4f}"
        )

    print(f"\nFinal calibration experiment complete! Results saved under {hind_res['paths']['cases_csv']}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Final Generic Vote-Share Calibration Experiment Runner."
    )
    parser.add_argument(
        "--initial-samples",
        type=int,
        default=INITIAL_SAMPLES,
        help="Initial Monte Carlo sample count (default: 5000).",
    )
    parser.add_argument(
        "--high-samples",
        type=int,
        default=HIGH_SAMPLES,
        help="High-sample stability count (default: 20000).",
    )
    parser.add_argument(
        "--initial-seed",
        type=int,
        default=INITIAL_SEED,
        help="Base seed (default: 12345).",
    )
    args = parser.parse_args(args_list)
    return run_full_calibration_experiment(
        initial_samples=args.initial_samples,
        high_samples=args.high_samples,
        initial_seed=args.initial_seed,
    )


if __name__ == "__main__":
    sys.exit(main())
