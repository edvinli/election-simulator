"""Challenger B: synthetic distributional and structural validation.

Synthetic fixtures only. No target-election truth is loaded and no score is computed.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from diagnostics.election_noise_v2.challengers import challenger_b as b_mod
from diagnostics.election_noise_v2.challengers.challenger_b import (
    NonPSDCovariance,
    draw_challenger_b,
    fit_challenger_b,
    symmetric_factor,
    zero_sum_projector,
    zero_sum_rank,
)


def pool(k: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=1.5, size=(k, 9))
    c = c - c.mean(axis=1, keepdims=True)
    return c - c.mean(axis=0)


POOLS = [pool(k=3, seed=2), pool(k=4, seed=5), pool(k=5, seed=7), pool(k=6, seed=9)]


class Structure(unittest.TestCase):
    def test_sigma_tilde_is_symmetric(self):
        for c in POOLS:
            s = fit_challenger_b(c).sigma_tilde
            self.assertLess(np.abs(s - s.T).max(), 1e-14)

    def test_sigma_tilde_annihilates_the_ones_vector(self):
        for c in POOLS:
            s = fit_challenger_b(c).sigma_tilde
            self.assertLess(np.abs(s @ np.ones(9)).max(), 1e-12)

    def test_delta_is_in_the_unit_interval(self):
        for c in POOLS:
            d = fit_challenger_b(c).delta
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_b_sq_never_exceeds_d_sq(self):
        for c in POOLS:
            f = fit_challenger_b(c)
            self.assertLessEqual(f.b_sq, f.d_sq + 1e-15)
            self.assertEqual(f.b_sq, min(f.bbar_sq, f.d_sq))

    def test_sigma_tilde_is_positive_semidefinite(self):
        for c in POOLS:
            w = np.linalg.eigvalsh(fit_challenger_b(c).sigma_tilde)
            self.assertGreater(w.min(), -1e-12)

    def test_rank_on_the_zero_sum_subspace_is_eight(self):
        """delta > 0 and tau^2 > 0 must give full rank on the 8-dim zero-sum subspace."""
        for c in POOLS:
            f = fit_challenger_b(c)
            self.assertGreater(f.delta, 0.0)
            self.assertGreater(f.tau_sq, 0.0)
            self.assertEqual(zero_sum_rank(f.sigma_tilde), 8, f"K={f.k}")

    def test_b_is_full_rank_where_a_is_singular(self):
        """The scientific contrast: at K=3 A spans <=2 dimensions, B spans 8."""
        c = pool(k=3, seed=2)
        self.assertEqual(zero_sum_rank(fit_challenger_b(c).sigma_tilde), 8)
        self.assertLessEqual(np.linalg.matrix_rank(c, tol=1e-9), 2)

    def test_degenerate_pool_gives_delta_one(self):
        """d^2 = 0 -> delta := 1, the preregistered limit."""
        tau = 0.7
        # A pool whose S_P is exactly tau^2 * P9 makes S_P - T vanish.
        p9 = zero_sum_projector(9)
        w, v = np.linalg.eigh(p9)
        basis = v[:, w > 0.5].T                        # 8 orthonormal zero-sum directions
        c = basis * np.sqrt(8.0 * tau)                 # S_P = C^T C / 8
        f = fit_challenger_b(c)
        self.assertLess(f.d_sq, 1e-20)
        self.assertEqual(f.delta, 1.0)
        np.testing.assert_allclose(f.sigma_tilde, (8 / 7) * f.target, atol=1e-10)

    def test_no_tunable_parameter_exists(self):
        sig = inspect.signature(fit_challenger_b)
        self.assertEqual(list(sig.parameters), ["centered"],
                         "Challenger B must take the pool and nothing else")
        fields = set(b_mod.ChallengerBFit.__dataclass_fields__)
        self.assertNotIn("h", fields)
        for banned in ("nu", "df", "ridge", "shrinkage_floor", "tail_multiplier", "weights"):
            self.assertNotIn(banned, fields)

    def test_pool_of_one_is_refused(self):
        with self.assertRaises(ValueError):
            fit_challenger_b(pool(k=1, seed=3)[:1])


class Sampling(unittest.TestCase):
    N = 400_000

    def test_empirical_covariance_approaches_sigma_tilde(self):
        for c in (pool(k=3, seed=2), pool(k=5, seed=7)):
            f = fit_challenger_b(c)
            r = draw_challenger_b(f, self.N, np.random.default_rng(31))
            self.assertLess(np.abs(np.cov(r.T, bias=True) - f.sigma_tilde).max(), 0.05)

    def test_empirical_mean_approaches_zero(self):
        f = fit_challenger_b(pool())
        r = draw_challenger_b(f, self.N, np.random.default_rng(32))
        self.assertLess(np.abs(r.mean(axis=0)).max(), 0.02)

    def test_draws_are_zero_sum(self):
        for c in POOLS:
            f = fit_challenger_b(c)
            r = draw_challenger_b(f, 20000, np.random.default_rng(33))
            self.assertLess(np.abs(r.sum(axis=1)).max(), 1e-10)

    def test_factor_reproduces_sigma_tilde(self):
        for c in POOLS:
            f = fit_challenger_b(c)
            L = symmetric_factor(f.sigma_tilde)
            np.testing.assert_allclose(L @ L.T, f.sigma_tilde, atol=1e-12)

    def test_draws_are_gaussian_not_heavy_tailed(self):
        """Excess kurtosis of a Gaussian marginal is ~0; a t-tail would show > 0."""
        f = fit_challenger_b(pool())
        r = draw_challenger_b(f, 300_000, np.random.default_rng(34))
        x = r[:, 0] / r[:, 0].std()
        self.assertLess(abs(float(np.mean(x ** 4) - 3.0)), 0.15)


class NumericalPolicy(unittest.TestCase):
    def test_materially_negative_eigenvalue_raises_rather_than_clipping(self):
        bad = np.eye(9) * 1.0
        bad[0, 0] = -0.5                      # a real negative direction, not round-off
        with self.assertRaises(NonPSDCovariance):
            symmetric_factor(bad)

    def test_asymmetric_input_raises(self):
        bad = np.eye(9)
        bad[0, 1] = 0.5
        with self.assertRaises(NonPSDCovariance):
            symmetric_factor(bad)

    def test_round_off_level_zero_is_accepted(self):
        """The structural null direction may sit at -1e-18; that must not raise."""
        f = fit_challenger_b(pool())
        s = f.sigma_tilde.copy()
        w, v = np.linalg.eigh(s)
        w[0] = -1e-18
        s = (v * w) @ v.T
        s = (s + s.T) / 2
        L = symmetric_factor(s)
        self.assertEqual(L.shape, (9, 9))

    def test_sigma_tilde_is_not_modified_by_factorisation(self):
        f = fit_challenger_b(pool())
        before = f.sigma_tilde.copy()
        symmetric_factor(f.sigma_tilde)
        np.testing.assert_array_equal(f.sigma_tilde, before)


if __name__ == "__main__":
    unittest.main()
