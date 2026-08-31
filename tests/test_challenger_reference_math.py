"""Independent reference mathematics for the preregistered challengers.

Every reference quantity here is recomputed from the preregistration formulas using
plain loops and ``numpy`` primitives only. **No reference calculation calls the
challenger helper it is checking** - not ``pool_covariance``, not
``zero_sum_projector``, not ``_normalized_frobenius_sq``, not ``fit_challenger_a``
and not ``fit_challenger_b``. The production implementation is then compared against
the reference at tight tolerances.

No target-election data is loaded and no score is computed.
"""

from __future__ import annotations

import unittest

import numpy as np

from diagnostics.election_noise_v2.challengers.challenger_a import (
    FROZEN_H_GRID,
    fit_challenger_a,
    pool_covariance,
)
from diagnostics.election_noise_v2.challengers.challenger_b import (
    ZERO_SUM_RANK,
    fit_challenger_b,
    zero_sum_projector,
)

TIGHT = 1e-12


# --------------------------------------------------------------------------- #
# Independent reference implementations - loops only, no challenger helpers.
# --------------------------------------------------------------------------- #

def ref_center(raw: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Column mean (zero-sum cleaned) and the centered pool, by explicit loops."""
    k, d = len(raw), len(raw[0])
    mean = [sum(raw[i][j] for i in range(k)) / k for j in range(d)]
    s = sum(mean)
    if abs(s) > 1e-12:
        mean = [m - s / d for m in mean]
    centered = [[raw[i][j] - mean[j] for j in range(d)] for i in range(k)]
    return mean, centered


def ref_s_p(centered: list[list[float]]) -> list[list[float]]:
    """S_P = C^T C / K, by explicit triple loop."""
    k, d = len(centered), len(centered[0])
    return [[sum(centered[i][a] * centered[i][b] for i in range(k)) / k
             for b in range(d)] for a in range(d)]


def ref_projector(d: int) -> list[list[float]]:
    """P9 = I - 11^T / 9, by explicit loop."""
    return [[(1.0 if a == b else 0.0) - 1.0 / d for b in range(d)] for a in range(d)]


def ref_norm_sq(a: list[list[float]], rank: int = 8) -> float:
    """||A||^2 = tr(A A^T)/8, by explicit loop."""
    return sum(a[i][j] * a[i][j] for i in range(len(a)) for j in range(len(a[0]))) / rank


def ref_sub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def ref_scale(a, s):
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]


def ref_challenger_b(raw_centered: list[list[float]]) -> dict:
    """The whole Challenger B closed form, independently."""
    k, d = len(raw_centered), len(raw_centered[0])
    s_p = ref_s_p(raw_centered)
    p9 = ref_projector(d)
    tau_sq = sum(s_p[i][i] for i in range(d)) / ZERO_SUM_RANK
    target = ref_scale(p9, tau_sq)
    d_sq = ref_norm_sq(ref_sub(s_p, target))
    bbar_sq = 0.0
    for j in range(k):
        cj = raw_centered[j]
        outer = [[cj[a] * cj[b] for b in range(d)] for a in range(d)]
        bbar_sq += ref_norm_sq(ref_sub(outer, s_p))
    bbar_sq /= (k * k)
    b_sq = min(bbar_sq, d_sq)
    delta = 1.0 if d_sq == 0.0 else b_sq / d_sq
    sigma_lw = [[delta * target[a][b] + (1.0 - delta) * s_p[a][b] for b in range(d)]
                for a in range(d)]
    sigma_tilde = ref_scale(sigma_lw, k / (k - 1))
    return {"s_p": s_p, "p9": p9, "tau_sq": tau_sq, "target": target, "d_sq": d_sq,
            "bbar_sq": bbar_sq, "b_sq": b_sq, "delta": delta,
            "sigma_lw": sigma_lw, "sigma_tilde": sigma_tilde}


# --------------------------------------------------------------------------- #
# Hand-checkable fixtures.
# --------------------------------------------------------------------------- #

def fixture_hand() -> np.ndarray:
    """K = 2, a single M<->L transfer of 1 pp each way. Already centered."""
    return np.array([
        [1.0, -1.0, 0, 0, 0, 0, 0, 0, 0],
        [-1.0, 1.0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)


def fixture_small() -> np.ndarray:
    """K = 3 raw (uncentered), zero-sum rows, small integers - hand-checkable."""
    return np.array([
        [2.0, -1.0, 1.0, -2.0, 3.0, -1.0, 0.0, -1.0, -1.0],
        [-1.0, 2.0, -2.0, 1.0, -1.0, 0.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 1.0, 1.0, -2.0, 1.0, -2.0, 2.0, 0.0],
    ], dtype=float)


def fixture_pool(k: int = 5, seed: int = 7) -> np.ndarray:
    """A larger centered zero-sum pool."""
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=1.5, size=(k, 9))
    c = c - c.mean(axis=1, keepdims=True)   # rows zero-sum
    return c - c.mean(axis=0)               # columns centered


class ReferenceFixtures(unittest.TestCase):
    def test_fixtures_are_zero_sum(self):
        for f in (fixture_hand(), fixture_small(), fixture_pool()):
            self.assertLess(np.abs(f.sum(axis=1)).max(), TIGHT)


class ChallengerAReferenceMath(unittest.TestCase):
    def test_centering_matches_reference(self):
        raw = fixture_small()
        _, ref_c = ref_center(raw.tolist())
        got = raw - raw.mean(axis=0)
        np.testing.assert_allclose(got, np.array(ref_c), atol=TIGHT)
        self.assertLess(np.abs(got.sum(axis=0)).max(), TIGHT, "centered pool must sum to 0")

    def test_s_p_matches_reference(self):
        for pool in (fixture_hand(), fixture_pool(), fixture_pool(k=3, seed=11)):
            ref = np.array(ref_s_p(pool.tolist()))
            np.testing.assert_allclose(pool_covariance(pool), ref, atol=TIGHT)

    def test_s_p_uses_divisor_K_not_bessel(self):
        """The binding convention: divisor K, no Bessel correction."""
        pool = fixture_pool()
        k = pool.shape[0]
        np.testing.assert_allclose(pool_covariance(pool), pool.T @ pool / k, atol=TIGHT)
        bessel = pool.T @ pool / (k - 1)
        self.assertGreater(np.abs(pool_covariance(pool) - bessel).max(), 1e-6,
                           "S_P must NOT carry a Bessel correction")

    def test_epsilon_covariance_equals_s_p(self):
        """eps = (1/sqrt K) sum_j z_j c_j has covariance (1/K) sum_j c_j c_j^T = S_P."""
        for pool in (fixture_small() - fixture_small().mean(axis=0), fixture_pool()):
            k = pool.shape[0]
            ref = np.zeros((9, 9))
            for j in range(k):
                ref += np.outer(pool[j], pool[j])
            ref /= k
            np.testing.assert_allclose(ref, np.array(ref_s_p(pool.tolist())), atol=TIGHT)
            np.testing.assert_allclose(pool_covariance(pool), ref, atol=TIGHT)

    def test_final_covariance_is_s_p_for_every_frozen_h(self):
        """Cov(R) = (S_P + h^2 S_P)/(1+h^2) = S_P, computed independently per h."""
        pool = fixture_pool()
        ref_sp = np.array(ref_s_p(pool.tolist()))
        for h in FROZEN_H_GRID:
            ref_final = (ref_sp + h * h * ref_sp) / (1.0 + h * h)
            np.testing.assert_allclose(ref_final, ref_sp, atol=TIGHT)
            fit = fit_challenger_a(pool, h)
            np.testing.assert_allclose(fit.theoretical_covariance, ref_final, atol=TIGHT)
            np.testing.assert_allclose(fit.theoretical_mean, np.zeros(9), atol=TIGHT)

    def test_variance_correction_denominator_is_binding(self):
        """Without /sqrt(1+h^2) the covariance would be (1+h^2) S_P, not S_P."""
        pool = fixture_pool()
        ref_sp = np.array(ref_s_p(pool.tolist()))
        for h in FROZEN_H_GRID:
            uncorrected = ref_sp + h * h * ref_sp          # Cov(c_k + h*eps)
            np.testing.assert_allclose(uncorrected, (1 + h * h) * ref_sp, atol=TIGHT)
            self.assertGreater(np.abs(uncorrected - ref_sp).max(), 1e-6,
                               f"h={h}: the correction must change the covariance")

    def test_every_frozen_h_is_reachable_and_grid_is_exact(self):
        self.assertEqual(FROZEN_H_GRID, (0.25, 0.50, 0.75, 1.00))
        pool = fixture_pool()
        for h in FROZEN_H_GRID:
            self.assertEqual(fit_challenger_a(pool, h).h, h)
        for bad in (0.0, 0.1, 0.3, 1.5, 2.0):
            with self.assertRaises(ValueError):
                fit_challenger_a(pool, bad)


class ChallengerBReferenceMath(unittest.TestCase):
    def _check(self, pool: np.ndarray):
        ref = ref_challenger_b(pool.tolist())
        fit = fit_challenger_b(pool)
        np.testing.assert_allclose(fit.s_p, np.array(ref["s_p"]), atol=TIGHT)
        np.testing.assert_allclose(fit.p9, np.array(ref["p9"]), atol=TIGHT)
        self.assertAlmostEqual(fit.tau_sq, ref["tau_sq"], delta=TIGHT)
        np.testing.assert_allclose(fit.target, np.array(ref["target"]), atol=TIGHT)
        self.assertAlmostEqual(fit.d_sq, ref["d_sq"], delta=TIGHT)
        self.assertAlmostEqual(fit.bbar_sq, ref["bbar_sq"], delta=TIGHT)
        self.assertAlmostEqual(fit.b_sq, ref["b_sq"], delta=TIGHT)
        self.assertAlmostEqual(fit.delta, ref["delta"], delta=TIGHT)
        np.testing.assert_allclose(fit.sigma_lw, np.array(ref["sigma_lw"]), atol=TIGHT)
        np.testing.assert_allclose(fit.sigma_tilde, np.array(ref["sigma_tilde"]), atol=TIGHT)

    def test_matches_reference_on_every_fixture(self):
        for pool in (fixture_hand(), fixture_small() - fixture_small().mean(axis=0),
                     fixture_pool(k=3, seed=2), fixture_pool(k=4, seed=5), fixture_pool(k=5)):
            with self.subTest(k=pool.shape[0]):
                self._check(pool)

    def test_hand_computed_values(self):
        """K=2 fixture: S_P has a single 2x2 block, tr(S_P)=2, tau^2 = 2/8 = 0.25."""
        pool = fixture_hand()
        fit = fit_challenger_b(pool)
        expected_sp = np.zeros((9, 9))
        expected_sp[0, 0] = expected_sp[1, 1] = 1.0
        expected_sp[0, 1] = expected_sp[1, 0] = -1.0
        np.testing.assert_allclose(fit.s_p, expected_sp, atol=TIGHT)
        self.assertAlmostEqual(float(np.trace(fit.s_p)), 2.0, delta=TIGHT)
        self.assertAlmostEqual(fit.tau_sq, 0.25, delta=TIGHT)
        self.assertAlmostEqual(fit.bessel_factor, 2.0, delta=TIGHT)

    def test_projector_matches_reference_and_is_idempotent(self):
        p9 = zero_sum_projector(9)
        np.testing.assert_allclose(p9, np.array(ref_projector(9)), atol=TIGHT)
        np.testing.assert_allclose(p9 @ p9, p9, atol=TIGHT)
        np.testing.assert_allclose(p9 @ np.ones(9), np.zeros(9), atol=TIGHT)
        self.assertAlmostEqual(float(np.trace(p9)), float(ZERO_SUM_RANK), delta=1e-12)

    def test_delta_is_invariant_to_the_norm_normalisation(self):
        """The 1/8 cancels in delta = b^2/d^2, so the unnormalized norm gives the same delta."""
        for pool in (fixture_pool(k=3, seed=2), fixture_pool(k=5)):
            ref_n = ref_challenger_b(pool.tolist())          # normalized by 1/8
            k, d = pool.shape
            s_p = np.array(ref_n["s_p"]); t = np.array(ref_n["target"])
            d2_raw = float(np.sum((s_p - t) ** 2))           # plain Frobenius
            bb_raw = sum(float(np.sum((np.outer(c, c) - s_p) ** 2)) for c in pool) / (k * k)
            delta_raw = min(bb_raw, d2_raw) / d2_raw
            self.assertAlmostEqual(delta_raw, ref_n["delta"], delta=1e-12)

    def test_bessel_correction_appears_exactly_once(self):
        for pool in (fixture_pool(k=3, seed=2), fixture_pool(k=4, seed=5), fixture_pool(k=5)):
            fit = fit_challenger_b(pool)
            k = pool.shape[0]
            np.testing.assert_allclose(fit.sigma_tilde, (k / (k - 1)) * fit.sigma_lw, atol=TIGHT)
            # and NOT twice
            self.assertGreater(np.abs(fit.sigma_tilde - (k / (k - 1)) ** 2 * fit.sigma_lw).max(),
                               1e-9)
            # tau^2, d^2, bbar^2 and delta must be free of it
            ref = ref_challenger_b(pool.tolist())
            self.assertAlmostEqual(fit.tau_sq, ref["tau_sq"], delta=TIGHT)
            self.assertAlmostEqual(fit.delta, ref["delta"], delta=TIGHT)


if __name__ == "__main__":
    unittest.main()
