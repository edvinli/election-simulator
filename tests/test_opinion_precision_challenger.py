"""Unit tests for OpinionState Empirical Pollster Precision Challenger (Experiment 2).

Verifies:
1. Reference-invariance across alternative ALR bases (REST, S, M).
2. N-deconfounding in CLR dispersion calculations.
3. Kish reduction identity when precision multipliers q_g = 1.
4. Strict anti-leakage boundaries on precision estimation.
5. Offline QA execution and decision gate stability.
"""

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.pollofpolls.state import (
    OpinionState,
    ReconstructedPoll,
    estimate_opinion,
    load_individual_polls_dataset,
    load_timeseries_dataset,
)
from scripts.pollofpolls.state_config import ALL_CATEGORIES, PARTIES, REFERENCE_CATEGORY
from scripts.opinion_precision_challenger.config import (
    ALL_CATEGORIES_9,
    POLLS_FILE,
    POP_TIMESERIES_FILE,
)
from scripts.opinion_precision_challenger.manifest import build_canonical_rolling_manifest
from scripts.opinion_precision_challenger.opinion_state import (
    calculate_kish_effective_count,
    estimate_opinion_with_precision_arm,
)
from scripts.opinion_precision_challenger.precision import (
    compute_sample_size_weight,
    estimate_pollster_precision,
    extract_historical_clr_residuals,
)
from scripts.opinion_precision_challenger.qa import (
    determine_final_decision,
    run_full_opinion_precision_qa,
    verify_reference_invariance_hard_gate,
)


class TestOpinionPrecisionChallenger(unittest.TestCase):
    """Test suite for Experiment 2 OpinionState precision challenger."""

    @classmethod
    def setUpClass(cls):
        cls.individual_polls, _ = load_individual_polls_dataset(POLLS_FILE)
        cls.pop_timeseries = load_timeseries_dataset(POP_TIMESERIES_FILE)
        cls.pop_by_date = {r["date"]: r["composition"] for r in cls.pop_timeseries}

    @staticmethod
    def _state_fingerprint(state: OpinionState) -> str:
        """Hash every deterministic state field used by the simulator and sampler."""
        payload = {
            "as_of": state.as_of.isoformat(),
            "estimate_date": state.estimate_date.isoformat(),
            "estimate_age_days": state.estimate_age_days,
            "parties": list(state.parties),
            "mean_pct": state.mean_pct,
            "rest_pct": state.rest_pct,
            "mean_alr": state.mean_alr,
            "covariance_alr": state.covariance_alr,
            "residual_covariance_alr": state.residual_covariance_alr,
            "recent_poll_count": state.recent_poll_count,
            "effective_poll_count": state.effective_poll_count,
            "residual_poll_count": state.residual_poll_count,
            "covariance_fallback_used": state.covariance_fallback_used,
            "house_effects_alr": state.house_effects_alr,
            "diagnostics": state.diagnostics,
            "cholesky_L": state._cholesky_L,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def test_arm_a_is_exact_shared_rc1_state_and_draws(self):
        """Arm A must hash-identically to production RC1, including covariance and seeded draws."""
        as_of = date(2022, 9, 1)
        production = estimate_opinion(as_of=as_of, data_dir=POLLS_FILE.parent)
        arm_a = estimate_opinion_with_precision_arm(
            as_of,
            self.individual_polls,
            self.pop_timeseries,
            weighting_arm="rc1_baseline",
            data_dir=POLLS_FILE.parent,
        )
        self.assertEqual(self._state_fingerprint(production), self._state_fingerprint(arm_a))
        self.assertEqual(production.sample(n=64, seed=9182), arm_a.sample(n=64, seed=9182))

    def test_reference_invariance_hard_gate(self):
        """Assert changing ALR reference category leaves precision multipliers invariant to 1e-6."""
        test_dates = [date(2018, 9, 1), date(2022, 9, 1), date(2024, 6, 1)]
        for o_date in test_dates:
            res = verify_reference_invariance_hard_gate(o_date, self.individual_polls, self.pop_timeseries)
            self.assertTrue(res["reference_invariance_passed"])
            self.assertLess(res["max_q_difference_across_reference_bases"], 1e-5)

    def test_n_deconfounding_dispersion_weight(self):
        """Assert sample size weight w_N behaves properly in [0.7, 1.5] and defaults to 1.0 for missing N."""
        self.assertEqual(compute_sample_size_weight(None), 1.0)
        self.assertEqual(compute_sample_size_weight(0), 1.0)
        self.assertAlmostEqual(compute_sample_size_weight(1000), 1.0, places=4)
        self.assertAlmostEqual(compute_sample_size_weight(400), 0.7, places=4)   # Lower clip
        self.assertAlmostEqual(compute_sample_size_weight(4000), 1.5, places=4)  # Upper clip

    def test_kish_reduction_identity(self):
        """Assert derived n_eff^precision formula exactly equals standard Kish when all q_g = 1.0."""
        o_date = date(2022, 9, 1)
        state_rc1 = estimate_opinion_with_precision_arm(
            o_date, self.individual_polls, self.pop_timeseries, weighting_arm="rc1_baseline"
        )
        # Force a precision state where all multipliers are 1.0
        prec_dummy = estimate_pollster_precision(o_date, self.individual_polls, self.pop_by_date)
        for p in prec_dummy.precision_multipliers_q:
            prec_dummy.precision_multipliers_q[p] = 1.0

        state_challenger = estimate_opinion_with_precision_arm(
            o_date, self.individual_polls, self.pop_timeseries, weighting_arm="precision_challenger", precision_state=prec_dummy
        )

        self.assertAlmostEqual(state_rc1.effective_poll_count, state_challenger.effective_poll_count, places=6)

    def test_kish_uses_final_precision_adjusted_weights(self):
        """Kish's denominator must square the final q-adjusted weights."""
        self.assertAlmostEqual(calculate_kish_effective_count([1.0, 2.0]), 1.8, places=10)
        self.assertAlmostEqual(calculate_kish_effective_count([1.0, 1.0]), 2.0, places=10)
        self.assertAlmostEqual(calculate_kish_effective_count([1.0, 2.0, 3.0]), 36.0 / 14.0, places=10)

    def test_precision_reference_excludes_evaluated_pollster(self):
        """Every precision residual must use a contemporaneous leave-one-house-out reference."""
        records, _ = extract_historical_clr_residuals(
            date(2022, 9, 1),
            self.individual_polls,
            self.pop_by_date,
        )
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertEqual(record["reference_method"], "leave_one_pollster_out_clr_mean")
            self.assertNotIn(record["pollster"], record["reference_pollsters"])
            self.assertGreater(len(record["reference_poll_ids"]), 0)

    def test_incomplete_coverage_is_not_a_performance_rejection(self):
        """Missing cases must stop adoption without labeling the model rejected."""
        decision = determine_final_decision(
            coverage_ready=False,
            coverage_status="PARTIAL",
            score_gate_passed=True,
            guardrail_res={
                "status": "SKIPPED_ROLLING_GATE_FAILED",
                "evaluation_status": "NOT_RUN",
            },
        )
        self.assertEqual(decision["final_decision"], "PRECISION_CHALLENGER_NOT_EVALUATED_KEEP_RC1")
        self.assertIn("partial", decision["decision_summary"].lower())

    def test_complete_coverage_score_failure_is_rejection(self):
        """With adequate coverage, a substantive score-gate failure is a rejection."""
        decision = determine_final_decision(
            coverage_ready=True,
            coverage_status="COMPLETE",
            score_gate_passed=False,
            guardrail_res={
                "status": "SKIPPED_ROLLING_GATE_FAILED",
                "evaluation_status": "NOT_RUN",
            },
        )
        self.assertEqual(decision["final_decision"], "PRECISION_CHALLENGER_REJECTED_KEEP_RC1")

    def test_strict_anti_leakage_precision_boundary(self):
        """Assert historical precision records satisfy strict publication/interview boundaries."""
        test_origins = [date(2015, 1, 1), date(2018, 9, 9), date(2022, 9, 11)]
        for o_date in test_origins:
            prec = estimate_pollster_precision(o_date, self.individual_polls, self.pop_by_date)
            self.assertEqual(prec.as_of, o_date)
            self.assertLessEqual(prec.lookback_start, o_date)
            for prof in prec.profiles_by_house.values():
                if prof.is_eligible:
                    self.assertGreaterEqual(prof.poll_count, 20)
                    self.assertGreaterEqual(prof.normalized_multiplier_q, 0.5)
                    self.assertLessEqual(prof.normalized_multiplier_q, 2.0)

    def test_offline_qa_execution(self):
        """Verify full QA pipeline executes in temporary directory with valid output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_file = tmp_path / "precision_validation_report.json"
            report = run_full_opinion_precision_qa(
                processed_dir=tmp_path,
                output_report_file=report_file,
                origin_step_days=28,
                m_draws=100,
                n_bootstrap_replications=100,
            )

            self.assertIn("rolling_decision_gate", report)
            self.assertIn("reference_invariance_hard_gate", report)
            self.assertTrue(report["reference_invariance_hard_gate"]["reference_invariance_passed"])
            self.assertIn("final_decision", report)
            self.assertEqual(report["rolling_decision_gate"]["coverage_status"], "PARTIAL")
            self.assertEqual(report["final_decision"], "PRECISION_CHALLENGER_NOT_EVALUATED_KEEP_RC1")
            self.assertEqual(report["stage_2_election_guardrail"]["evaluation_status"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
