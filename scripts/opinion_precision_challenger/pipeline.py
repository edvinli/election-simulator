"""CLI and pipeline entry point for OpinionState Precision Challenger (Experiment 2)."""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    BOOTSTRAP_REPLICATIONS,
    DEFAULT_ORIGIN_STEP_DAYS,
    EVALUATION_DRAWS_COUNT,
    PROCESSED_DIR,
)
from .qa import run_full_opinion_precision_qa


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OpinionState empirical pollster precision challenger pipeline (Experiment 2)."
    )
    parser.add_argument("--step-days", type=int, default=DEFAULT_ORIGIN_STEP_DAYS, help="Origin stepping in days.")
    parser.add_argument("--draws", type=int, default=EVALUATION_DRAWS_COUNT, help="Number of Monte Carlo draws per case.")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPLICATIONS, help="Bootstrap replications.")
    args = parser.parse_args()

    print("=== Starting Experiment 2: OpinionState Precision Challenger Pipeline ===")
    report = run_full_opinion_precision_qa(
        processed_dir=PROCESSED_DIR,
        origin_step_days=args.step_days,
        m_draws=args.draws,
        n_bootstrap_replications=args.bootstrap_reps,
    )

    print("\n==========================================================================================")
    print("EXPERIMENT 2: PRECISION CHALLENGER FINAL OUTCOME")
    print("==========================================================================================")
    print(f"Final Decision:  {report['final_decision']}")
    print(f"Rolling Gate:    {report['rolling_decision_gate']['rolling_gate_passed']}")
    print(f"Horizons Won:    {report['rolling_decision_gate']['horizons_won_on_es']}")
    print(f"Relative ES Imp: {report['pooled_scores']['relative_es_improvement_pct']}% (95% CI: {report['calendar_block_bootstrap_6m']['relative_es_improvement']['ci_95_pct']}%)")
    print(f"Relative CRPS:   {report['pooled_scores']['relative_crps_improvement_pct']}%")
    print(f"Summary:         {report['decision_summary']}")
    print("==========================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
