"""Pipeline runner for Election Result Layer v2 (percentage-point transfers)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .forward_eval import run_forward_election_layer_evaluation
from .hindcast import run_election_layer_v2_hindcasts


def run_full_v2_pipeline(samples: int = 5000, seed: int = 12345) -> int:
    print(">>> Step 1: Running forward chronological election-layer evaluation (2010-2022) ...")
    fwd_res = run_forward_election_layer_evaluation()
    print(f"Forward evaluation completed! Saved to {fwd_res['paths']['cases_csv']}")

    print("\n>>> Step 2: Running full-pipeline paired hindcasts (2018 + 2022) ...")
    hind_res = run_election_layer_v2_hindcasts(samples=samples, seed=seed)

    fwd_summ = fwd_res["report"]
    hind_summ = hind_res["summary"]

    print("\n==========================================================================================")
    print("STANDALONE FORWARD EVALUATION FROM 14-DAY POLLING CONSENSUS (2010-2022)")
    print("==========================================================================================")
    print("Variant             | Overall MAE (8p) | Overall CRPS (8p) | Mean Lambda ")
    print("--------------------+------------------+-------------------+-------------")
    for r in fwd_summ["by_variant_overall"]:
        print(f"{r['variant']:<19s} |      {r['overall_MAE_8parties']:>5.2f}%      |      {r['overall_mean_CRPS_8parties']:>7.4f}      |    {r['overall_mean_lambda']:>6.4f}")

    print("\n------------------------------------------------------------------------------------------")
    print("FORWARD EVALUATION BY ELECTION")
    print("------------------------------------------------------------------------------------------")
    print("Year | Pool Size | Variant             | MAE (8p) | CRPS (8p) | Lambda")
    print("-----+-----------+---------------------+----------+-----------+-------")
    for r in fwd_summ["by_variant_election"]:
        print(f"{r['election_year']} |     {r['training_pool_size']}     | {r['variant']:<19s} |  {r['MAE_8parties']:>5.2f}%  |  {r['mean_CRPS_8parties']:>7.4f}  | {r['mean_lambda']:>5.3f}")

    print("\n==========================================================================================")
    print("FULL PIPELINE HINDCASTS WITH STATE + DYNAMICS (2018 + 2022)")
    print("==========================================================================================")
    print("Variant             |  MAE(8p)  | CRPS(8p) | CRPS(all9) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) | Mean Lam (p05) | Lam<0.90")
    print("--------------------+-----------+----------+------------+---------------+---------------+---------------+----------------+---------")
    for r in hind_summ["by_variant_overall"]:
        print(
            f"{r['variant']:<19s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} |   {r['mean_CRPS_all9']:>8.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f}) | "
            f"   {r['mean_lambda']:>6.4f}      |  {r['fraction_lambda_lt_0_90']:>5.1%}"
        )

    print("\n------------------------------------------------------------------------------------------")
    print("FULL HINDCASTS BY ELECTION")
    print("------------------------------------------------------------------------------------------")
    print("Election   | Variant             |  MAE(8p)  | CRPS(8p) | Cov 50% (W50) | Cov 80% (W80) | Cov 90% (W90) ")
    print("-----------+---------------------+-----------+----------+---------------+---------------+---------------")
    for r in hind_summ["by_election_variant"]:
        print(
            f"{r['election_date']:<10s} | {r['variant']:<19s} |   {r['MAE_8parties']:>5.2f}   |  {r['mean_CRPS_8parties']:>7.4f} | "
            f"{r['coverage_50']:>5.1%} ({r['mean_width_50']:>4.2f}) | {r['coverage_80']:>5.1%} ({r['mean_width_80']:>4.2f}) | {r['coverage_90']:>5.1%} ({r['mean_width_90']:>4.2f})"
        )

    print(f"\nPipeline completed! Output saved to {hind_res['paths']['cases_csv']}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Election Result Layer v2 (Percentage-Point Transfers) Runner."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Monte Carlo sample count (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )
    args = parser.parse_args(args_list)
    return run_full_v2_pipeline(samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
