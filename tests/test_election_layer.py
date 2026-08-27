"""Unit and regression tests for Residual Robustness and Election Result Layer v1."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest
import numpy as np

from scripts.election_layer.config import CANONICAL_WINDOW_DAYS
from scripts.election_layer.hindcast import run_election_layer_hindcasts
from scripts.election_layer.models import (
    apply_election_layer_variants,
    derive_election_layer_seed,
    load_chronological_training_residuals,
)
from scripts.election_layer.robustness import run_window_robustness_audit
from scripts.pollofpolls.clr import composition_to_clr


class ChronologicalTrainingPoolTests(unittest.TestCase):
    """Test strict chronological election boundaries and mathematical residual centering."""

    def test_2018_chronological_training_set(self) -> None:
        pool_2018 = load_chronological_training_residuals(target_election_year=2018)
        # Must be exactly {2002, 2006, 2010, 2014}
        self.assertEqual(pool_2018.training_years, (2002, 2006, 2010, 2014))
        self.assertNotIn(2018, pool_2018.training_years)
        self.assertNotIn(2022, pool_2018.training_years)
        self.assertEqual(pool_2018.residuals_matrix.shape, (4, 9))

    def test_2022_chronological_training_set(self) -> None:
        pool_2022 = load_chronological_training_residuals(target_election_year=2022)
        # Must be exactly {2002, 2006, 2010, 2014, 2018}
        self.assertEqual(pool_2022.training_years, (2002, 2006, 2010, 2014, 2018))
        self.assertNotIn(2022, pool_2022.training_years)
        self.assertEqual(pool_2022.residuals_matrix.shape, (5, 9))

    def test_mathematical_centering_of_residuals(self) -> None:
        for target_year in (2018, 2022):
            pool = load_chronological_training_residuals(target_election_year=target_year)
            # Mathematical mean of centered residuals must be strictly zero to floating-point precision
            mean_centered = np.mean(pool.centered_residuals_matrix, axis=0)
            np.testing.assert_allclose(mean_centered, np.zeros(9), atol=1e-12)


class ElectionLayerModelVariantsTests(unittest.TestCase):
    """Test paired Monte Carlo application, residual difference isolation, and simplex validity."""

    def test_paired_residual_difference_equals_mean_bias(self) -> None:
        pool_2022 = load_chronological_training_residuals(target_election_year=2022)
        samples_count = 100
        seed = 42

        # Create dummy base CLR matrix
        base_clr = np.zeros((samples_count, 9))

        # Sample indices directly
        rng = np.random.default_rng(seed)
        sampled_indices = rng.integers(0, len(pool_2022.training_years), size=samples_count)

        noise_only_clr = base_clr + pool_2022.centered_residuals_matrix[sampled_indices]
        bias_plus_noise_clr = base_clr + pool_2022.residuals_matrix[sampled_indices]

        # For every single sample i, (bias_plus_noise - noise_only) must equal mean_bias_clr exactly!
        diff_clr = bias_plus_noise_clr - noise_only_clr
        for i in range(samples_count):
            np.testing.assert_allclose(diff_clr[i], pool_2022.mean_bias_clr, atol=1e-12)

    def test_simplex_validity_across_all_variants(self) -> None:
        pool_2018 = load_chronological_training_residuals(target_election_year=2018)
        samples_count = 50
        seed = 123

        # Dummy positive composition mapped to CLR
        dummy_comp = {"M": 20.0, "L": 5.0, "C": 8.0, "KD": 6.0, "S": 30.0, "V": 8.0, "MP": 4.0, "SD": 17.0, "REST": 2.0}
        clr_vec, _ = composition_to_clr(dummy_comp)
        base_clr_mat = np.tile(clr_vec, (samples_count, 1))

        variants = apply_election_layer_variants(base_clr_mat, pool_2018, samples_count, seed)
        self.assertEqual(set(variants.keys()), {"base", "bias_only", "noise_only", "bias_plus_noise"})

        for var_name, comp_mat in variants.items():
            self.assertEqual(comp_mat.shape, (samples_count, 9))
            # Every row must sum strictly to 100.0%
            row_sums = np.sum(comp_mat, axis=1)
            np.testing.assert_allclose(row_sums, 100.0, atol=1e-6)
            # All party shares must be strictly positive
            self.assertTrue(np.all(comp_mat > 0.0))


class RobustnessAuditTests(unittest.TestCase):
    """Test 7-day, 14-day, and 21-day consensus window execution."""

    def test_robustness_windows_execution(self) -> None:
        res = run_window_robustness_audit(windows=(7, 14, 21))
        self.assertEqual(len(res["report"]["windows_evaluated"]), 3)
        self.assertEqual(len(res["window_overall_df"]), 3)


if __name__ == "__main__":
    unittest.main()
