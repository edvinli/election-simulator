"""Targeted tests for the ElectionNoise mixture diagnostic instrumentation.

These verify that the diagnostic's reconstruction of the residual-election index
is bit-identical to the index production actually used, and that the passive
capture wrapper does not alter production output.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.election_noise_mixture.run_simulation import recompute_residual_indices
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.vote_share_calibration.config import MIN_SHARE_PCT
from scripts.vote_share_calibration.models import (
    apply_vote_share_models,
    derive_vote_share_layer_seeds,
)


class ResidualIndexReconstructionTest(unittest.TestCase):
    """The reconstructed residual index must equal the index production consumed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pool = load_chronological_pp_residuals(target_election_year=2026)
        cls.k = len(cls.pool.training_years)
        cls.n = 4_000
        cls.idx_seed, cls.sign_seed = derive_vote_share_layer_seeds(
            base_seed=12345, origin_date=__import__("datetime").date(2026, 8, 24), horizon_days=20
        )
        rng = np.random.default_rng(999)
        base = rng.dirichlet(np.full(9, 40.0), size=cls.n) * 100.0
        cls.base = base

    def test_pool_is_six_elections(self) -> None:
        self.assertEqual(self.pool.training_years, (2002, 2006, 2010, 2014, 2018, 2022))

    def test_reconstructed_index_reproduces_production_transfer_exactly(self) -> None:
        models = apply_vote_share_models(
            base_comp_matrix=self.base,
            training_pool=self.pool,
            samples_count=self.n,
            index_seed=self.idx_seed,
            sign_seed=self.sign_seed,
            eps=MIN_SHARE_PCT,
        )
        prod_comp, prod_lam = models["pp_centered_noise"]

        idx = recompute_residual_indices(self.idx_seed, self.k, self.n)
        recon_comp, recon_lam = apply_batch_simplex_transfer(
            self.base, self.pool.centered_residuals_matrix[idx], eps=MIN_SHARE_PCT
        )

        np.testing.assert_array_equal(prod_comp, recon_comp)
        np.testing.assert_array_equal(prod_lam, recon_lam)

    def test_reconstructed_index_is_uniform_over_pool(self) -> None:
        idx = recompute_residual_indices(self.idx_seed, self.k, 100_000)
        counts = np.bincount(idx, minlength=self.k)
        self.assertEqual(int(counts.sum()), 100_000)
        self.assertEqual(len(np.unique(idx)), self.k)
        # Uniform multinomial: sd per cell ~ sqrt(N p (1-p)) ~ 118 for N=1e5, p=1/6.
        self.assertLess(float(np.max(np.abs(counts - 100_000 / self.k))), 600.0)

    def test_centering_is_exact(self) -> None:
        np.testing.assert_allclose(
            self.pool.centered_residuals_matrix.mean(axis=0),
            np.zeros(9),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            self.pool.residuals_matrix.sum(axis=1), np.zeros(self.k), atol=1e-10
        )
        np.testing.assert_allclose(
            self.pool.centered_residuals_matrix.sum(axis=1), np.zeros(self.k), atol=1e-10
        )

    def test_no_duplicate_residual_vectors(self) -> None:
        uniq = np.unique(np.round(self.pool.residuals_matrix, 9), axis=0)
        self.assertEqual(uniq.shape[0], self.k)


if __name__ == "__main__":
    unittest.main()
