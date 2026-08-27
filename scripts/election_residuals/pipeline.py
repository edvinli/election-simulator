"""CLI runner and pipeline execution for Historical Poll-to-Election Residual Study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .residuals import calculate_residuals_study


def main(args_list: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Historical Poll-to-Election Residual Study for Swedish Riksdag Elections (2002-2022)."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory for processed residual tables.",
    )
    parser.add_argument(
        "--polls-file",
        default=None,
        help="Custom polls file path.",
    )
    parser.add_argument(
        "--elections-file",
        default=None,
        help="Custom election targets file path.",
    )

    args = parser.parse_args(args_list)

    print(">>> Running Historical Poll-to-Election Residual Study (2002, 2006, 2010, 2014, 2018, 2022) ...")
    res = calculate_residuals_study(
        polls_file=args.polls_file,
        elections_file=args.elections_file,
        output_dir=args.output_dir,
    )

    summ = res["summary"]
    print("\n==========================================================================================")
    print("ELECTION-LEVEL RESIDUAL SUMMARY")
    print("==========================================================================================")
    print("Election   | Pollsters | MAE (8 parties) | MAE (All 9) | Max Miss Party (Diff pp) | REST Diff pp")
    print("-----------+-----------+-----------------+-------------+--------------------------+-------------")
    for r in summ["by_election"]:
        print(
            f"{r['election_date']} |     {r['pollster_count']:>2d}    |     {r['MAE_8parties']:>5.2f}%     |    {r['MAE_all9']:>5.2f}%   | "
            f"{r['max_miss_party']:<4s} ({r['max_miss_pp']:>+5.2f} pp)         |   {r['REST_residual_pp']:>+5.2f} pp"
        )

    print("\n==========================================================================================")
    print("PARTY-LEVEL RESIDUAL SUMMARY ACROSS ALL 6 ELECTIONS (2002-2022)")
    print("==========================================================================================")
    print("Party | Mean Diff (pp) | Median (pp) | Std (pp) | MAE (pp) | Sign (+ / - / 0) | Mean CLR Diff")
    print("------+----------------+-------------+----------+----------+------------------+--------------")
    for r in summ["by_party"]:
        print(
            f"{r['party']:<5s} |    {r['mean_residual_pp']:>+5.2f} pp    |   {r['median_residual_pp']:>+5.2f} pp  |  {r['std_residual_pp']:>5.2f}   |  {r['MAE_pp']:>5.2f}%  | "
            f"{r['sign_consistency']:<16s} |    {r['mean_residual_clr']:>+6.3f}"
        )

    print("\n==========================================================================================")
    print("THRESHOLD-LEVEL DIAGNOSTICS (NEAR 4% vs AWAY)")
    print("==========================================================================================")
    tc = summ["threshold_comparison"]
    print(f"Near Threshold (within 1.5 pp of 4%): {tc['near_threshold_cases_count']} cases | MAE = {tc['near_threshold_MAE_pp']:.2f} pp | Mean Bias = {tc['near_threshold_mean_residual_pp']:>+5.2f} pp")
    print(f"Away from Threshold:                   {tc['away_threshold_cases_count']} cases | MAE = {tc['away_threshold_MAE_pp']:.2f} pp | Mean Bias = {tc['away_threshold_mean_residual_pp']:>+5.2f} pp")

    print("\n==========================================================================================")
    print("POLITICAL BLOC CO-MOVEMENT DIAGNOSTICS")
    print("==========================================================================================")
    print("Year | Left/Green (S+V+MP) Diff | Right/Alliansen (M+L+C+KD) Diff | SD Diff")
    print("-----+--------------------------+---------------------------------+---------")
    for b in summ["bloc_diagnostics"]:
        print(f"{b['election_year']} |         {b['left_green_residual_pp']:>+5.2f} pp        |             {b['alliansen_residual_pp']:>+5.2f} pp              | {b['sd_residual_pp']:>+5.2f} pp")

    print(f"\nResidual study successfully completed! Outputs saved to {res['paths']['summary_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
