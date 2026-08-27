"""Unit and regression tests for Election Result Layer v2 (percentage-point transfers)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest
import numpy as np

from scripts.election_layer_v2.config import (
    ALL_HISTORICAL_ELECTIONS,
    CANONICAL_WINDOW_DAYS,
    MIN_SHARE_PCT,
)
from scripts.election_layer_v2.forward_eval import (
    compute_discrete_crps,
    run_forward_election_layer_evaluation,
)
from scripts.election_layer_v2.residuals_pool import (
    load_chronological_pp_residuals,
)
from scripts.election_layer_v2.transfer import (
    apply_batch_simplex_transfer,
    apply_simplex_transfer,
    compute_simplex_transfer_scale,
    summarize_lambda_diagnostics,
)


class SimplexSafeTransferTests(unittest.TestCase):
    """Test bounded lambda transfer scaling, edge cases, and simplex invariants."""

    def test_lambda_when_no_negative_residuals(self) -> None:
        base = np.array([20.0, 30.0, 50.0])
        res = np.array([0.0, 0.0, 0.0])
        lam = compute_simplex_transfer_scale(base, res, eps=0.01)
        self.assertEqual(lam, 1.0)

    def test_lambda_unattenuated_when_feasible(self) -> None:
        base = np.array([20.0, 30.0, 50.0])
        res = np.array([-2.0, +1.0, +1.0])
        lam = compute_simplex_transfer_scale(base, res, eps=0.01)
        self.assertEqual(lam, 1.0)
        x_p, lam_val = apply_simplex_transfer(base, res, eps=0.01)
        np.testing.assert_allclose(x_p, np.array([18.0, 31.0, 51.0]), atol=1e-10)
        self.assertEqual(lam_val, 1.0)

    def test_lambda_correct_attenuation_when_donor_exhausted(self) -> None:
        base = np.array([2.0, 48.0, 50.0])
        res = np.array([-5.0, +2.5, +2.5])  # Party 0 only has 2.0%, cannot donate 5.0%
        eps = 0.01
        expected_lam = (2.0 - 0.01) / 5.0  # 1.99 / 5.0 = 0.398
        lam = compute_simplex_transfer_scale(base, res, eps=eps)
        self.assertAlmostEqual(lam, expected_lam, places=6)

        x_p, lam_val = apply_simplex_transfer(base, res, eps=eps)
        self.assertAlmostEqual(lam_val, expected_lam, places=6)
        self.assertAlmostEqual(x_p[0], 0.01, places=6)  # Donor hits exactly epsilon
        np.testing.assert_allclose(np.sum(x_p), 100.0, atol=1e-8)
        self.assertTrue(np.all(x_p >= eps))

    def test_batch_transfer_and_diagnostics(self) -> None:
        base_mat = np.array([
            [20.0, 30.0, 50.0],
            [1.0, 49.0, 50.0],
        ])
        res_vec = np.array([-2.0, 1.0, 1.0])
        out_mat, lams = apply_batch_simplex_transfer(base_mat, res_vec, eps=0.01)
        self.assertEqual(len(lams), 2)
        self.assertEqual(lams[0], 1.0)
        self.assertAlmostEqual(lams[1], (1.0 - 0.01) / 2.0, places=6)

        diag = summarize_lambda_diagnostics(lams)
        self.assertAlmostEqual(diag["mean_lambda"], (1.0 + 0.495) / 2.0, places=3)
        self.assertEqual(diag["fraction_lambda_lt_0_99"], 0.5)


class ChronologicalPPResidualPoolTests(unittest.TestCase):
    """Test strict chronological training pools and zero-sum invariants."""

    def test_chronological_training_sets(self) -> None:
        p10 = load_chronological_pp_residuals(2010)
        self.assertEqual(p10.training_years, (2002, 2006))

        p14 = load_chronological_pp_residuals(2014)
        self.assertEqual(p14.training_years, (2002, 2006, 2010))

        p18 = load_chronological_pp_residuals(2018)
        self.assertEqual(p18.training_years, (2002, 2006, 2010, 2014))

        p22 = load_chronological_pp_residuals(2022)
        self.assertEqual(p22.training_years, (2002, 2006, 2010, 2014, 2018))

    def test_residual_zero_sum_invariant(self) -> None:
        for yr in (2010, 2014, 2018, 2022):
            pool = load_chronological_pp_residuals(yr)
            for row in pool.residuals_matrix:
                np.testing.assert_allclose(np.sum(row), 0.0, atol=1e-10)
            np.testing.assert_allclose(np.sum(pool.mean_bias_pp), 0.0, atol=1e-10)
            for row in pool.centered_residuals_matrix:
                np.testing.assert_allclose(np.sum(row), 0.0, atol=1e-10)

    def test_mathematical_centering_of_residuals(self) -> None:
        for yr in (2010, 2014, 2018, 2022):
            pool = load_chronological_pp_residuals(yr)
            mean_centered = np.mean(pool.centered_residuals_matrix, axis=0)
            np.testing.assert_allclose(mean_centered, np.zeros(9), atol=1e-12)


class DiscreteCRPSTests(unittest.TestCase):
    """Test exact discrete CRPS implementation."""

    def test_discrete_crps_single_point(self) -> None:
        pts = np.array([25.0])
        self.assertEqual(compute_discrete_crps(pts, 28.0), 3.0)

    def test_discrete_crps_two_points(self) -> None:
        # pts = [20, 30], actual = 25
        # first_term = mean(|20-25|, |30-25|) = 5
        # diff_matrix: (0 + 10 + 10 + 0)/4 = 5
        # second_term = 0.5 * 5 = 2.5
        # CRPS = 5 - 2.5 = 2.5
        pts = np.array([20.0, 30.0])
        self.assertEqual(compute_discrete_crps(pts, 25.0), 2.5)


if __name__ == "__main__":
    unittest.main()
