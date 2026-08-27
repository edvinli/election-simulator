"""Unit tests for PoP state-dependence diagnostic pipeline (Step 4A).

Tests verify CLR compositional invertibility, strict anti-leakage candidate pool filters,
standardized Euclidean distance properties, common random number determinism,
vote-share scoring, and offline QA reproducibility.
"""
from datetime import date
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from scripts.pollofpolls.clr import clr_to_composition, composition_to_clr
from scripts.pop_baseline.metrics import continuous_crps, energy_score
from scripts.pop_state_diagnostics.config import (
    ALL_CATEGORIES_9,
    POP_TIMESERIES_FILE,
    PROCESSED_DATA_DIR,
    THRESHOLD_STARTING_BINS,
)
from scripts.pop_state_diagnostics.evaluation import (
    compute_threshold_starting_state_distributions,
    evaluate_single_case,
    run_calendar_block_bootstrap,
    sample_transitions_to_vote_shares,
    uniform_indices_from_unit_draws,
)
from scripts.pop_state_diagnostics.qa import run_full_state_diagnostics_qa
from scripts.pop_state_diagnostics.similarity import (
    compute_standardized_clr_distance,
    rank_candidate_transitions,
)
from scripts.pop_state_diagnostics.transitions import (
    DailyState,
    StateTransition,
    build_all_exact_transitions,
    compute_historical_clr_stds_as_of,
    get_leakage_safe_candidate_pool,
    load_canonical_pop_series,
)


class TestPoPStateDiagnostics(unittest.TestCase):
    """Test suite for Step 4A state-dependence diagnostic engine and decision gate."""

    @classmethod
    def setUpClass(cls):
        cls.daily_states, cls.by_date = load_canonical_pop_series(POP_TIMESERIES_FILE)
        cls.transitions_by_horizon = build_all_exact_transitions(cls.daily_states, cls.by_date)

    def test_clr_composition_exact_invertibility(self):
        """Assert exact round-trip mapping: comp -> clr -> comp."""
        for st in self.daily_states[:10]:
            clr_vec, _ = composition_to_clr(st.composition)
            recovered_comp = clr_to_composition(clr_vec)
            for p in ALL_CATEGORIES_9:
                self.assertAlmostEqual(st.composition[p], recovered_comp[p], places=4)

    def test_strict_anti_leakage_candidate_pool(self):
        """Assert that candidate historical transitions strictly satisfy end_date <= origin_date."""
        test_origins = [date(2016, 1, 1), date(2018, 9, 9), date(2022, 9, 11), date(2025, 1, 1)]
        for o_date in test_origins:
            for h in [7, 14, 28, 56, 84, 112]:
                pool = get_leakage_safe_candidate_pool(o_date, h, self.transitions_by_horizon)
                for tr in pool:
                    self.assertLessEqual(tr.end_date, o_date)
                    self.assertEqual(tr.horizon_days, h)

    def test_standardized_clr_distance_metric_properties(self):
        """Assert distance symmetry, non-negativity, and identity of indiscernibles."""
        stds = np.ones(9, dtype=float)
        vec1 = np.array([0.1, 0.2, -0.3, 0.4, -0.1, -0.2, 0.0, 0.1, -0.2])
        vec2 = np.array([-0.2, 0.1, 0.0, 0.3, 0.2, -0.1, -0.1, 0.0, -0.2])
        
        d_self = compute_standardized_clr_distance(vec1, vec1, stds)
        self.assertAlmostEqual(d_self, 0.0, places=6)
        
        d12 = compute_standardized_clr_distance(vec1, vec2, stds)
        d21 = compute_standardized_clr_distance(vec2, vec1, stds)
        self.assertAlmostEqual(d12, d21, places=6)
        self.assertGreater(d12, 0.0)

    def test_nearest_neighbor_ranking_and_diagnostics(self):
        """Assert candidate ranking produces sorted distances and accurate audit records."""
        o_date = date(2020, 6, 1)
        origin_st = self.by_date[o_date]
        stds = compute_historical_clr_stds_as_of(self.daily_states, o_date)
        cand_pool = get_leakage_safe_candidate_pool(o_date, 14, self.transitions_by_horizon)
        
        sorted_nn, sorted_recent, audit_recs = rank_candidate_transitions(origin_st, cand_pool, stds)
        
        self.assertEqual(len(sorted_nn), len(cand_pool))
        self.assertEqual(len(sorted_recent), len(cand_pool))
        self.assertEqual(len(audit_recs), 50)
        
        # Verify distance monotonicity
        distances = [d for _, d in sorted_nn]
        self.assertTrue(all(distances[i] <= distances[i+1] for i in range(len(distances)-1)))
        
        # Verify recency monotonicity
        end_dates = [tr.end_date for tr in sorted_recent]
        self.assertTrue(all(end_dates[i] >= end_dates[i+1] for i in range(len(end_dates)-1)))

    def test_sample_transitions_to_vote_shares_properties(self):
        """Assert simulated vote shares sum to exact 100% and are non-negative."""
        o_st = self.daily_states[100]
        cand_pool = get_leakage_safe_candidate_pool(o_st.observation_date, 14, self.transitions_by_horizon)
        
        # The sampler consumes candidate-pool indices.  Generate those
        # indices from continuous unit draws so this property test remains
        # valid for pools smaller than the number of requested draws and does
        # not rely on the modulo-based sampling that the production path
        # deliberately avoids.
        unit_draws = np.linspace(0.0, np.nextafter(1.0, 0.0), 100)
        draw_indices = uniform_indices_from_unit_draws(unit_draws, len(cand_pool))
        sign_flips = np.ones(100)
        shares_mat = sample_transitions_to_vote_shares(o_st.clr, cand_pool, draw_indices, sign_flips)
        
        self.assertEqual(shares_mat.shape, (100, 9))
        self.assertTrue((shares_mat >= 0.0).all())
        row_sums = np.sum(shares_mat, axis=1)
        np.testing.assert_allclose(row_sums, 100.0, atol=1e-5)

    def test_uniform_pool_indices_are_unbiased_and_deterministic(self):
        """Shared unit draws must map to valid indices without integer-modulo bias."""
        draws = np.array([0.0, 0.199999, 0.2, 0.999999], dtype=float)
        expected = np.array([0, 0, 1, 4], dtype=np.int64)
        np.testing.assert_array_equal(uniform_indices_from_unit_draws(draws, 5), expected)
        np.testing.assert_array_equal(uniform_indices_from_unit_draws(draws, 5), expected)

    def test_energy_score_and_crps_correctness(self):
        """Assert Energy Score and CRPS evaluation on identical and shifted distributions."""
        actual = np.array([20.0, 5.0, 4.0, 5.0, 30.0, 8.0, 5.0, 20.0, 3.0])
        exact_samples = np.tile(actual, (100, 1))
        
        es_zero = energy_score(exact_samples, actual)
        self.assertAlmostEqual(es_zero, 0.0, places=4)
        
        crps_zero = continuous_crps(exact_samples[:, 0], actual[0])
        self.assertAlmostEqual(crps_zero, 0.0, places=4)
        
        noisy_samples = exact_samples + np.random.normal(0, 1, size=exact_samples.shape)
        noisy_samples = 100.0 * np.exp(noisy_samples) / np.sum(np.exp(noisy_samples), axis=1, keepdims=True)
        es_noisy = energy_score(noisy_samples, actual)
        self.assertGreater(es_noisy, 0.0)

    def test_threshold_starting_bins_definitions(self):
        """Assert threshold starting bins are strictly half-open and non-overlapping."""
        for i in range(len(THRESHOLD_STARTING_BINS) - 1):
            low_curr, high_curr, _ = THRESHOLD_STARTING_BINS[i]
            low_next, _, _ = THRESHOLD_STARTING_BINS[i+1]
            self.assertEqual(high_curr, low_next)

    def test_offline_qa_execution_and_decision_gate(self):
        """Verify full QA pipeline executes in temporary directory and asserts all assertions pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_file = tmp_path / "state_diagnostics_validation_report.json"
            report = run_full_state_diagnostics_qa(
                processed_dir=tmp_path,
                output_report_file=report_file,
                n_bootstrap_replications=100,
                m_draws=100,
                origin_step_days=28,
            )
            
            self.assertTrue(report["assertions"]["all_assertions_passed"])
            self.assertEqual(report["step_4b_gate_decision"], "REJECT_STATE_DYNAMICS_KEEP_RC1")
            self.assertFalse(report["gate_passed"])
            self.assertIn("pooled_scores", report)
            self.assertIn("calendar_block_bootstrap_6m", report)


if __name__ == "__main__":
    unittest.main()
