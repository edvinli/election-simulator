"""Focused tests for the final threshold-attribution research cycle."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.pop_baseline.metrics import score_vote_draws
from scripts.pop_baseline.paired_precision import (
    case_identity_hash,
    classify_skip_reason,
    run_paired_precision_benchmark,
    summarize_skip_cases,
    validate_paired_case_set,
)
from scripts.pop_baseline.threshold import run_threshold_support_diagnostic
from scripts.pop_baseline.threshold_attribution import (
    apply_attribution_gate,
    build_component_attribution,
)
from scripts.pop_baseline.threshold_metrics import (
    build_threshold_brier_breakdown,
    probability_bin,
    summarize_threshold_by_dimensions,
)
from scripts.pop_baseline.variants import (
    VARIANT_A,
    VARIANT_B,
    VARIANT_C,
    VARIANT_D,
    VARIANT_E,
    VARIANT_F,
    VARIANT_ORDER,
    generate_variant_draws,
)
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares


class ThresholdAttributionTests(unittest.TestCase):
    def _synthetic_case(self) -> dict:
        actual = np.asarray([19.0, 4.1, 7.0, 5.0, 30.0, 7.0, 5.0, 20.0, 2.9], dtype=float)
        draws = np.tile(actual, (4, 1))
        draws[:, 1] = np.asarray([3.8, 4.2, 4.2, 3.8])
        metrics = score_vote_draws(
            draws,
            actual,
            ("M", "L", "C", "KD", "S", "V", "MP", "SD", "REST"),
            threshold_parties=("M", "L", "C", "KD", "S", "V", "MP", "SD"),
        )
        return {
            "evaluation": "synthetic",
            "status": "SCORED",
            "origin_date": "2022-09-04",
            "target_date": "2022-09-11",
            "horizon_days": 7,
            "samples": 4,
            "actual_vote_share_pct": {party: float(actual[i]) for i, party in enumerate(("M", "L", "C", "KD", "S", "V", "MP", "SD", "REST"))},
            "models": {VARIANT_A: metrics, VARIANT_B: metrics},
        }

    def test_threshold_breakdown_has_required_dimensions_and_does_not_add_rest(self) -> None:
        rows = build_threshold_brier_breakdown([self._synthetic_case()])
        self.assertEqual(len(rows), 16)
        self.assertNotIn("REST", {row["party"] for row in rows})
        self.assertTrue({"election_year", "horizon_days", "party", "forecast_probability", "actual_above_threshold"}.issubset(rows[0]))
        summary = summarize_threshold_by_dimensions(rows)
        self.assertGreater(len(summary["by_probability_bin"]), 0)
        self.assertEqual(sum(row["observation_count"] for row in summary["by_outcome"]), 32)

    def test_probability_bins_are_fixed_and_inclusive_at_one(self) -> None:
        self.assertEqual(probability_bin(0.0), "[0,0.1)")
        self.assertEqual(probability_bin(0.1), "[0.1,0.2)")
        self.assertEqual(probability_bin(1.0), "[0.9,1]")
        with self.assertRaises(ValueError):
            probability_bin(1.01)

    def test_paired_case_identity_rejects_duplicate_scored_case(self) -> None:
        case = self._synthetic_case()
        validation = validate_paired_case_set([case], model_a=VARIANT_A, model_b=VARIANT_B)
        self.assertEqual(validation["scored_case_count"], 1)
        self.assertEqual(case_identity_hash([case]), validation["case_identity_hash"])
        with self.assertRaises(ValueError):
            validate_paired_case_set([case, case], model_a=VARIANT_A, model_b=VARIANT_B)

    def test_variant_surfaces_are_isolated_and_b_matches_frozen_engine(self) -> None:
        root = Path("data/processed")
        variants = generate_variant_draws(
            origin_date=date(2022, 9, 4),
            election_date=date(2022, 9, 11),
            samples=8,
            seed=42,
            processed_root=root,
        )
        self.assertEqual(set(variants), set(VARIANT_ORDER))
        self.assertTrue(all(variants[variant].status == "RUN" for variant in VARIANT_ORDER))
        direct = generate_national_vote_shares(
            as_of=date(2022, 9, 4),
            election_date=date(2022, 9, 11),
            samples=8,
            seed=42,
            data_dir=root,
        )
        np.testing.assert_array_equal(variants[VARIANT_B].samples_pct, direct.nat_shares_matrix * 100.0)
        np.testing.assert_array_equal(variants[VARIANT_D].samples_pct, direct.base_comp_matrix)
        self.assertEqual(
            variants[VARIANT_A].diagnostics["raw_path_sha256"],
            variants[VARIANT_F].diagnostics["raw_path_sha256"],
        )
        self.assertEqual(
            variants[VARIANT_B].diagnostics["dynamics_draws_sha256"],
            variants[VARIANT_C].diagnostics["shared_dynamics_sha256"],
        )
        self.assertEqual(
            variants[VARIANT_B].diagnostics["state_draws_sha256"],
            variants[VARIANT_E].diagnostics["shared_state_draws_sha256"],
        )
        self.assertEqual(
            variants[VARIANT_C].diagnostics["center_preservation_check"]["expected_center_source"],
            "OpinionState.mean_pct/rest_pct",
        )
        self.assertTrue(variants[VARIANT_E].diagnostics["shared_state_draws"])

    def test_six_election_final_poll_evidence_is_not_probabilistic_brier_validation(self) -> None:
        result = run_threshold_support_diagnostic()
        self.assertEqual(result["elections_observed"], [2002, 2006, 2010, 2014, 2018, 2022])
        self.assertEqual(result["probabilistic_evaluation"]["status"], "NOT_RUN")
        self.assertGreater(result["focus_3_to_5_pct"]["observation_count"], 0)
        self.assertEqual(result["focus_3_to_5_pct"]["fail_count"], 0)

    def test_attribution_gate_can_select_only_a_declared_variant(self) -> None:
        base = {
            "threshold_brier_mean_8parties": 0.10,
            "vote_crps_mean_8parties": 0.50,
            "joint_vote_energy_score_9parties": 1.00,
            "median_vote_mae_8parties": 0.50,
        }
        models = {VARIANT_B: base, VARIANT_A: base}
        for variant in (VARIANT_C, VARIANT_D, VARIANT_E, VARIANT_F):
            models[variant] = dict(base)
        models[VARIANT_C] = {
            **base,
            "threshold_brier_mean_8parties": 0.08,
        }
        case = self._synthetic_case()
        case["models"] = models
        decision = apply_attribution_gate([case])
        self.assertEqual(decision["selected_variant"], VARIANT_C)
        self.assertFalse(decision["automatic_adoption"])

    def test_component_attribution_contains_support_reference_pairings(self) -> None:
        case = self._synthetic_case()
        base = case["models"][VARIANT_B]
        case["models"] = {
            VARIANT_A: {**base, "threshold_brier_mean_8parties": 0.10, "vote_crps_mean_8parties": 0.50, "joint_vote_energy_score_9parties": 1.00},
            VARIANT_B: {**base, "threshold_brier_mean_8parties": 0.20, "vote_crps_mean_8parties": 0.60, "joint_vote_energy_score_9parties": 2.00},
            VARIANT_C: {**base, "threshold_brier_mean_8parties": 0.19, "vote_crps_mean_8parties": 0.61, "joint_vote_energy_score_9parties": 2.01},
            VARIANT_D: {**base, "threshold_brier_mean_8parties": 0.21, "vote_crps_mean_8parties": 0.65, "joint_vote_energy_score_9parties": 2.05},
            VARIANT_E: {**base, "threshold_brier_mean_8parties": 0.18, "vote_crps_mean_8parties": 0.62, "joint_vote_energy_score_9parties": 2.02},
            VARIANT_F: {**base, "threshold_brier_mean_8parties": 0.15, "vote_crps_mean_8parties": 0.55, "joint_vote_energy_score_9parties": 1.80},
        }
        rows = {row["component"]: row for row in build_component_attribution([case])}
        self.assertEqual(
            {row["delta_label"] for row in rows.values()},
            {"F_minus_A", "C_minus_B", "D_minus_B", "E_minus_B"},
        )
        self.assertEqual(rows["support_transfer"]["candidate_variant"], VARIANT_F)
        self.assertEqual(rows["support_transfer"]["reference_variant"], VARIANT_A)
        self.assertGreater(rows["support_transfer"]["threshold_brier_delta_candidate_minus_reference"], 0.0)
        self.assertLess(rows["opinion_state_uncertainty"]["threshold_brier_delta_candidate_minus_reference"], 0.0)
        self.assertGreater(rows["pp_centered_noise"]["threshold_brier_delta_candidate_minus_reference"], 0.0)
        self.assertLess(rows["dynamics"]["threshold_brier_delta_candidate_minus_reference"], 0.0)
        self.assertEqual(rows["support_transfer"]["threshold_brier_case_win_rate_candidate"], 0.0)

    def test_precision_report_collapses_seed_rows_before_threshold_reliability(self) -> None:
        report = run_paired_precision_benchmark(
            run_rolling=False,
            run_elections=True,
            seeds=(12345, 24680),
            election_samples=8,
            horizons=(7,),
        )
        self.assertTrue(report["case_set"]["same_case_set_across_seeds"])
        self.assertEqual(report["threshold_brier"]["raw_seed_row_count"], 64)
        self.assertEqual(report["threshold_brier"]["row_count"], 32)
        self.assertNotIn("normal_95ci_low", report["paired_case_deltas"][0])
        self.assertIn("not independent", report["paired_case_deltas"][0]["interpretation"])

    def test_variant_benchmark_marks_missing_origins_explicitly(self) -> None:
        from scripts.pop_baseline.variants import run_variant_election_benchmark

        cases = run_variant_election_benchmark(
            elections=(date(2018, 9, 9),),
            horizons=(1461,),
            samples=8,
            seed=42,
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["status"], "SKIPPED")
        self.assertEqual(cases[0]["reason"], "missing_exact_stored_pop_origin")

    def test_variant_failure_is_local_and_does_not_erase_b_or_d(self) -> None:
        from scripts.pop_baseline import variants as variant_module

        with patch.object(variant_module, "estimate_opinion", side_effect=ValueError("synthetic C failure")):
            variants = generate_variant_draws(
                origin_date=date(2022, 9, 4),
                election_date=date(2022, 9, 11),
                samples=8,
                seed=42,
                processed_root=Path("data/processed"),
            )
        self.assertEqual(variants[VARIANT_B].status, "RUN")
        self.assertEqual(variants[VARIANT_D].status, "RUN")
        self.assertEqual(variants[VARIANT_C].status, "NOT_RUN")
        self.assertEqual(variants[VARIANT_E].status, "RUN")

    def test_skip_reason_accounting_is_explanatory_and_fail_closed(self) -> None:
        detail = classify_skip_reason("insufficient_rc1_transitions:15<30")
        self.assertEqual(detail["class"], "chronological_history_gap")
        self.assertIn("cannot be eliminated", detail["resolution"])
        summary = summarize_skip_cases([
            {"status": "SKIPPED", "evaluation": "rolling", "reason": "missing_exact_origin_or_target_observation"},
            {"status": "SKIPPED", "evaluation": "rolling", "reason": "insufficient_rc1_transitions:15<30"},
        ])
        self.assertEqual(summary["skipped_case_count"], 2)
        self.assertEqual(summary["by_evaluation"]["rolling"], 2)


if __name__ == "__main__":
    unittest.main()
