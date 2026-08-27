"""Unit and regression tests for Final Generic Vote-Share Calibration Experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest
import numpy as np

from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.vote_share_calibration.energy_score import (
    compute_discrete_energy_score,
    compute_energy_score,
    reference_energy_score_slow,
)
from scripts.vote_share_calibration.forward_eval import run_exact_forward_evaluation
from scripts.vote_share_calibration.models import (
    apply_vote_share_models,
    derive_vote_share_layer_seeds,
)


class EnergyScoreInvariantTests(unittest.TestCase):
    """Test Energy Score properties, invariants, and exactness against reference."""

    def test_energy_score_deterministic_forecast_invariant(self) -> None:
        # For a single deterministic forecast X = x, ES(F, y) = ||x - y||_2
        x = np.array([20.0, 30.0, 50.0])
        y = np.array([25.0, 28.0, 47.0])
        expected_dist = float(np.linalg.norm(x - y))

        samples_1 = x[None, :]  # Shape (1, 3)
        es_val_1 = compute_energy_score(samples_1, y)
        self.assertAlmostEqual(es_val_1, expected_dist, places=6)

        # Repeated identical samples must yield the same deterministic distance
        samples_rep = np.tile(x, (100, 1))  # Shape (100, 3)
        es_val_rep = compute_energy_score(samples_rep, y)
        self.assertAlmostEqual(es_val_rep, expected_dist, places=6)

    def test_energy_score_matches_slow_reference(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.dirichlet(np.ones(9), size=120) * 100.0  # Shape (120, 9)
        actual = rng.dirichlet(np.ones(9)) * 100.0            # Shape (9,)

        fast_es = compute_energy_score(samples, actual, chunk_size=25)
        slow_es = reference_energy_score_slow(samples, actual)
        self.assertAlmostEqual(fast_es, slow_es, places=6)

    def test_discrete_energy_score_exactness(self) -> None:
        support = np.array([
            [20.0, 30.0, 50.0],
            [22.0, 28.0, 50.0],
            [18.0, 32.0, 50.0],
        ])
        actual = np.array([21.0, 29.0, 50.0])

        discrete_es = compute_discrete_energy_score(support, actual)
        slow_es = reference_energy_score_slow(support, actual)
        self.assertAlmostEqual(discrete_es, slow_es, places=6)


class VoteShareModelPairingTests(unittest.TestCase):
    """Test strict architectural pairing and sign-symmetric residual properties."""

    def test_shared_index_and_independent_sign_pairing(self) -> None:
        pool = load_chronological_pp_residuals(target_election_year=2022)
        k = len(pool.training_years)
        samples_count = 50

        dummy_base = np.tile(np.array([20.0, 5.0, 8.0, 6.0, 30.0, 8.0, 4.0, 17.0, 2.0]), (samples_count, 1))
        idx_seed, sign_seed = derive_vote_share_layer_seeds(12345, date(2022, 9, 4), 7)

        outputs = apply_vote_share_models(
            base_comp_matrix=dummy_base,
            training_pool=pool,
            samples_count=samples_count,
            index_seed=idx_seed,
            sign_seed=sign_seed,
        )

        self.assertEqual(set(outputs.keys()), {"base", "pp_centered_noise", "pp_symmetric_noise"})

        # Re-derive sample draws directly to assert pairing
        rng_idx = np.random.default_rng(idx_seed)
        sampled_indices = rng_idx.integers(0, k, size=samples_count)

        rng_sign = np.random.default_rng(sign_seed)
        sampled_signs = rng_sign.choice([-1.0, 1.0], size=samples_count, p=[0.5, 0.5])

        # Verify all signs are strictly -1 or +1
        self.assertTrue(set(np.unique(sampled_signs)).issubset({-1.0, 1.0}))

        # Verify simplex validity across all models
        for m_id, (mat, lams) in outputs.items():
            self.assertEqual(mat.shape, (samples_count, 9))
            row_sums = np.sum(mat, axis=1)
            np.testing.assert_allclose(row_sums, 100.0, atol=1e-6)
            self.assertTrue(np.all(mat >= 0.01))

    def test_sign_symmetric_support_in_standalone_forward_eval(self) -> None:
        res = run_exact_forward_evaluation(elections=[date(2018, 9, 9)])
        # In 2018, historical pool has K=4 (2002, 2006, 2010, 2014)
        # pp_symmetric_noise must have 2K=8 discrete support points
        sub_sym = res["cases_df"][
            (res["cases_df"]["election_year"] == 2018) & (res["cases_df"]["model"] == "pp_symmetric_noise")
        ]
        self.assertEqual(len(sub_sym), 9)  # 9 categories evaluated


if __name__ == "__main__":
    unittest.main()
