"""Pipeline runner executing robustness audit and election layer hindcasts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .hindcast import run_election_layer_hindcasts
from .robustness import run_window_robustness_audit


def run_full_pipeline(samples: int = 5000, seed: int = 12345) -> int:
    print(">>> Step 1: Running residual-window robustness audit (7d, 14d, 21d) ...")
    rob_res = run_window_robustness_audit()
    print(f"Robustness audit completed! Saved to {rob_res['csv_path']}")

    print("\n>>> Step 2: Running 4-variant election-layer hindcasts across 2018 and 2022 ...")
    hind_res = run_election_layer_hindcasts(samples=samples, seed=seed)

    summ = hind_res["summary"]
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

    print(f"\nPipeline successfully completed! Output saved to {hind_res['paths']['cases_csv']}")
    return 0


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Residual Robustness + Election Result Layer v1 Runner."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Monte Carlo samples count (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )
    args = parser.parse_args(args_list)
    return run_full_pipeline(samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
