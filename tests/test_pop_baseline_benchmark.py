"""Focused tests for matched PoPBaseline versus Candidate-A evidence."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.pop_baseline.benchmark import (
    run_election_vote_benchmark,
    run_final_poll_experiment,
    run_rolling_dynamics_benchmark,
)
from scripts.pop_baseline.metrics import continuous_crps, energy_score, score_vote_draws, threshold_brier
from scripts.pop_baseline.threshold import classify_threshold_band, run_threshold_support_diagnostic
from scripts.pop_baseline.diagnostics import (
    attribute_national_variance,
    compare_coverage_rows,
    run_candidate_a_variance_diagnostic,
)


class PoPBaselineBenchmarkTests(unittest.TestCase):
    def test_metrics_match_basic_invariants(self) -> None:
        draws = np.asarray([[3.0, 97.0], [5.0, 95.0]], dtype=float)
        self.assertAlmostEqual(continuous_crps(draws[:, 0], 4.0), 0.5)
        self.assertAlmostEqual(threshold_brier(draws[:, 0], 4.0), 0.25)
        self.assertAlmostEqual(energy_score(draws, np.asarray([4.0, 96.0])), np.sqrt(2.0) / 2.0)
        scores = score_vote_draws(draws, np.asarray([4.0, 96.0]), ("L", "REST"), threshold_parties=("L",))
        self.assertIn("vote_crps_mean_9parties", scores)
        self.assertIsNone(scores["per_party"]["REST"]["threshold_brier"])

    def test_rolling_benchmark_has_explicit_skips_and_paired_models(self) -> None:
        cases = run_rolling_dynamics_benchmark(
            start_date=date(2015, 1, 1),
            end_date=date(2015, 3, 1),
            horizons=(7,),
            origin_step_days=28,
            samples=16,
            seed=42,
        )
        self.assertGreater(len(cases), 0)
        scored = [case for case in cases if case["status"] == "SCORED"]
        self.assertGreater(len(scored), 0)
        self.assertEqual(
            set(scored[0]["models"]),
            {"pop_baseline_v1", "election_simulator_v1_rc1_dynamics"},
        )
        self.assertEqual(scored[0]["samples"], 16)

    def test_election_benchmark_uses_same_origin_and_marks_seats_unavailable(self) -> None:
        cases = run_election_vote_benchmark(
            elections=(date(2022, 9, 11),),
            horizons=(7,),
            samples=16,
            seed=42,
        )
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["status"], "SCORED")
        self.assertEqual(case["origin_date"], "2022-09-04")
        for metrics in case["models"].values():
            self.assertIn("threshold_brier_mean_8parties", metrics)

    def test_final_poll_experiment_is_diagnostic_and_not_fabricated(self) -> None:
        result = run_final_poll_experiment(elections=(date(2010, 9, 19), date(2022, 9, 11)))
        self.assertIn(result["status"], {"COMPLETE_DIAGNOSTIC_ONLY", "PARTIAL", "NOT_RUN"})
        self.assertTrue(all("status" in row for row in result["records"]))

    def test_threshold_bands_are_predeclared_and_support_rule_is_fail_closed(self) -> None:
        self.assertEqual(classify_threshold_band(2.999), "<3")
        self.assertEqual(classify_threshold_band(3.0), "3-3.5")
        self.assertEqual(classify_threshold_band(3.5), "3.5-4")
        self.assertEqual(classify_threshold_band(4.0), "4-4.5")
        self.assertEqual(classify_threshold_band(5.0), ">5")
        report = run_threshold_support_diagnostic(
            elections=(date(2018, 9, 9), date(2022, 9, 11)),
            min_observations_per_band=8,
            min_independent_elections=4,
        )
        self.assertEqual(report["decision"], "KEEP_RC1")
        self.assertEqual(report["status"], "INSUFFICIENT_HISTORICAL_SUPPORT")

    def test_variance_diagnostic_is_sequential_and_non_adaptive(self) -> None:
        state = np.asarray([[0.20, 0.80], [0.21, 0.79], [0.19, 0.81]], dtype=float)
        dynamic = state + np.asarray([[0.01, -0.01], [-0.01, 0.01], [0.00, 0.00]])
        final = dynamic + np.asarray([[0.005, -0.005], [-0.005, 0.005], [0.00, 0.00]])
        result = attribute_national_variance(
            opinion_state_draws=state,
            base_comp_matrix=dynamic,
            nat_shares_matrix=final,
            party_order=("L", "REST"),
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertIn("election_residual", result["layers"])
        self.assertIn("order-dependent", result["attribution_warning"])
        coverage = compare_coverage_rows([], model_ids=("pop_baseline_v1",))
        self.assertEqual(coverage["pop_baseline_v1"]["scored_cases"], 0)

    def test_candidate_a_variance_diagnostic_is_report_only(self) -> None:
        report = run_candidate_a_variance_diagnostic(
            processed_root=Path("data/processed"),
            as_of="2026-08-23",
            samples=8,
            seed=42,
        )
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(report["election_noise_scale_variant"]["decision"], "KEEP_RC1")
        self.assertEqual(report["coverage_comparison"]["pop_baseline_v1"]["scored_cases"], 0)


if __name__ == "__main__":
    unittest.main()
