"""Fast unit tests of the calibration-2030 T3-rev noise laws and harness helpers."""

from __future__ import annotations

import hashlib
import json
import math
import unittest

import numpy as np

from diagnostics.calibration_2030 import noise_laws as N
from diagnostics.calibration_2030 import t3rev_noise as H
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b, symmetric_factor

AMENDMENT_001 = N.ROOT / "data" / "processed" / "calibration_2030" / "amendments" / "001-t3-flag-interim-beta-half.json"


def synthetic_pool(k: int, seed: int, shift: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(k, 9))
    if shift is not None:
        r = r + shift
    r = r - r.mean(axis=1, keepdims=True)
    consensus = rng.uniform(2.0, 30.0, size=(k, 9))
    return {"residuals": r, "consensus": consensus, "unreported": np.zeros((k, 9), dtype=bool),
            "years": list(range(2000, 2000 + 4 * k, 4))}


class ProductionParityTest(unittest.TestCase):
    def test_lw0_equals_production_byte_for_byte(self):
        prod = load_chronological_pp_residuals(2026)
        pool = N.current_rule_pool(list(prod.training_years))
        self.assertTrue(np.array_equal(pool["residuals"], prod.residuals_matrix))
        z = np.random.default_rng(20260929).standard_normal((2_000, 9))
        expected = z @ symmetric_factor(fit_election_noise_b(prod.centered_residuals_matrix).sigma_tilde).T
        law = N.fit_law("LW0", pool)
        got = law.draw(z, np.full((2_000, 9), 11.0))
        self.assertEqual(got.tobytes(), expected.tobytes())
        self.assertEqual(law.summary()["beta"], 0.0)
        self.assertEqual(law.summary()["gamma"], 0.0)

    def test_n0_is_amendment_001_on_2002_2026(self):
        interim = json.loads(AMENDMENT_001.read_text(encoding="utf-8"))["interim_2030_law"]
        s = N.fit_law("N0", N.corrected_pool(N.ELECTION_DAYS)).summary()
        np.testing.assert_allclose(s["mean_pp"], interim["mean_shift_pp"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(s["sd_pp"], interim["sd_pp"], rtol=0, atol=1e-12)

    def test_n0_is_half_mean_plus_lw0(self):
        pool = synthetic_pool(5, 1, shift=np.linspace(-1, 1, 9))
        z = np.random.default_rng(2).standard_normal((300, 9))
        lw, n0 = N.fit_law("LW0", pool), N.fit_law("N0", pool)
        r_bar = pool["residuals"].mean(axis=0)
        np.testing.assert_allclose(n0.draw(z, None), 0.5 * (r_bar - r_bar.mean()) + lw.draw(z, None), atol=1e-12)

    def test_n0_matches_t3_gaussian_beta_half_byte_for_byte(self):
        from diagnostics.calibration_2030.t3_noise import pool_law

        pool = N.corrected_pool([2002, 2006, 2010, 2014, 2018])
        mean_bias, factor, _ = pool_law(pool["residuals"])
        z = np.random.default_rng(13).standard_normal((1_000, 9))
        self.assertEqual(N.fit_law("N0", pool).draw(z, None).tobytes(), (0.5 * mean_bias + z @ factor.T).tobytes())


class HierarchicalLawTest(unittest.TestCase):
    def test_nh0_is_lw_up_to_mean_and_inflation(self):
        pool = synthetic_pool(5, 3, shift=np.linspace(-2, 2, 9))
        z = np.random.default_rng(4).standard_normal((500, 9))
        lw, nh = N.fit_law("LW0", pool), N.fit_law("NH0", pool)
        self.assertGreater(nh.beta, 0.0)
        u_bar = np.array(nh.summary()["u_bar"])
        # Same covariance as LW0 (Σ̂ = Σ̃ at γ = 0 without unreported cells). The draws are compared
        # through NH0's own factor: Σ̃ has a degenerate eigenspace, so eigh's basis, and hence the
        # factor, can rotate under a last-bit change of Σ. The law, not the factor, is the claim.
        np.testing.assert_allclose(nh.sigma, lw.sigma, atol=1e-12)
        np.testing.assert_allclose(nh.factor_t.T @ nh.factor_t, lw.sigma, atol=1e-12)
        expected = nh.beta * u_bar + math.sqrt(1.0 + nh.beta / nh.K) * (z @ nh.factor_t)
        np.testing.assert_allclose(nh.draw(z, None), expected, atol=1e-12)
        self.assertAlmostEqual(nh.summary()["variance_inflation"], 1.0 + nh.beta / nh.K, places=14)

    def test_nh0_beta_zero_limit_equals_lw(self):
        a, b = np.random.default_rng(5).normal(size=(2, 9))
        rows = np.array([a, -a, b, -b])
        rows = rows - rows.mean(axis=1, keepdims=True)
        pool = {"residuals": rows, "consensus": np.full((4, 9), 10.0), "unreported": np.zeros((4, 9), bool),
                "years": [2002, 2006, 2010, 2014]}
        nh, lw = N.fit_law("NH0", pool), N.fit_law("LW0", pool)
        self.assertEqual(nh.kappa_hat, 0.0)
        self.assertEqual(nh.beta, 0.0)
        self.assertEqual(nh.scale, 1.0)
        np.testing.assert_allclose(nh.mean, np.zeros(9), atol=0)
        np.testing.assert_allclose(nh.sigma, lw.sigma, atol=1e-12)
        np.testing.assert_allclose(nh.factor_t.T @ nh.factor_t, lw.factor_t.T @ lw.factor_t, atol=1e-12)
        z = np.random.default_rng(6).standard_normal((500, 9))
        np.testing.assert_allclose(nh.draw(z, None), z @ nh.factor_t, atol=1e-12)

    def test_kappa_formula(self):
        for seed, shift in ((7, np.linspace(-3, 3, 9)), (8, None)):
            pool = synthetic_pool(6, seed, shift=shift)
            law = N.fit_law("NH0.5", pool)
            u = pool["residuals"] / np.maximum(pool["consensus"], 1.0) ** 0.5
            u = u - u.mean(axis=1, keepdims=True)
            u_bar = u.mean(axis=0)
            u_bar = u_bar - u_bar.mean()
            sigma = fit_election_noise_b(u - u_bar).sigma_tilde
            kappa = max(0.0, float(u_bar @ u_bar) / float(np.trace(sigma)) - 1.0 / 6)
            self.assertAlmostEqual(law.kappa_hat, kappa, places=12)
            self.assertAlmostEqual(law.beta, 6 * kappa / (6 * kappa + 1), places=12)
            np.testing.assert_allclose(law.mean, law.beta * u_bar, atol=1e-12)
            self.assertAlmostEqual(law.scale, math.sqrt(1 + law.beta / 6), places=12)

    def test_kappa_recovers_between_election_variance(self):
        rng = np.random.default_rng(9)
        p9 = np.eye(9) - np.ones((9, 9)) / 9
        kappa_true, k, estimates = 0.5, 40, []
        for _ in range(200):
            mu = math.sqrt(kappa_true) * rng.normal(size=9) @ p9
            r = (mu + rng.normal(size=(k, 9))) @ p9
            pool = {"residuals": r, "consensus": np.full((k, 9), 10.0), "unreported": np.zeros((k, 9), bool),
                    "years": list(range(k))}
            estimates.append(N.fit_law("NH0", pool).kappa_hat)
        self.assertAlmostEqual(float(np.mean(estimates)), kappa_true, delta=0.08)


class SizeScalingTest(unittest.TestCase):
    def test_zero_sum_restoration(self):
        pool = synthetic_pool(5, 10, shift=np.linspace(-1, 1, 9))
        rng = np.random.default_rng(11)
        z = rng.standard_normal((400, 9))
        base = rng.dirichlet(np.ones(9), size=400) * 100.0
        for name in N.LAWS:
            r = N.fit_law(name, pool).draw(z, base)
            self.assertEqual(r.shape, (400, 9))
            self.assertLess(float(np.max(np.abs(r.sum(axis=1)))), 1e-10, name)

    def test_restoration_formula(self):
        u = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])
        c = np.array([20.0, 5.0, 0.5, 9.0, 30.0, 8.0, 4.0, 18.0, 5.5])
        s = np.maximum(c, 1.0) ** 0.5
        raw = u * s
        np.testing.assert_allclose(N.rescale(u, c, 0.5), raw - raw.sum() * s / s.sum(), atol=1e-14)

    def test_gamma_scaling_round_trips(self):
        rng = np.random.default_rng(12)
        c = rng.uniform(0.0, 40.0, size=(50, 9))
        for gamma in (0.0, 0.5, 1.0):
            u = rng.normal(size=(50, 9))
            u = u - u.mean(axis=1, keepdims=True)
            np.testing.assert_allclose(N.standardize(N.rescale(u, c, gamma), c, gamma), u, atol=1e-12)
            r = rng.normal(size=(50, 9))
            r = r - r.mean(axis=1, keepdims=True)
            np.testing.assert_allclose(N.rescale(N.standardize(r, c, gamma), c, gamma), r, atol=1e-12)

    def test_unreported_rule_touches_only_sd_2002(self):
        pool = N.corrected_pool(N.ELECTION_DAYS)
        self.assertEqual([(pool["years"][i], N.ALL_CATEGORIES[j]) for i, j in np.argwhere(pool["unreported"])],
                         [(2002, "SD")])
        self.assertEqual(pool["consensus"][0, 7], 0.0)
        for gamma in (0.0, 0.5, 1.0):
            u = N.standardize(pool["residuals"], pool["consensus"], gamma, pool["unreported"])
            raw = pool["residuals"] / np.maximum(pool["consensus"], 1.0) ** gamma
            raw[0, 7] = 0.0
            np.testing.assert_allclose(u, raw - raw.mean(axis=1, keepdims=True), atol=1e-14)
            plain = N.standardize(pool["residuals"], pool["consensus"], gamma)
            np.testing.assert_allclose(u[1:], plain[1:], atol=0)
        n0 = N.fit_law("N0", N.corrected_pool([2002, 2006, 2010]))
        self.assertNotIn("unreported_cells", n0.summary())


class HarnessHelpersTest(unittest.TestCase):
    def test_seed_strings(self):
        self.assertEqual(H.case_seed(2014, "rolling", ""),
                         int(hashlib.sha256(b"calibration-2030:T3rev:2014:rolling").hexdigest()[:8], 16))
        self.assertEqual(H.case_seed(2026, "heldout", ":3"),
                         int(hashlib.sha256(b"calibration-2030:T3rev:2026:heldout:3").hexdigest()[:8], 16))

    def test_designs(self):
        d = {(design, target): years for design, target, years in H.designs()}
        self.assertEqual([len(d[("rolling", t)]) for t in (2014, 2018, 2022)], [3, 4, 5])
        self.assertTrue(all(len(d[("loo", t)]) == 5 and t not in d[("loo", t)] for t in H.PRE_2026))
        self.assertEqual(d[("heldout", 2026)], list(H.PRE_2026))
        self.assertEqual(len(d), 10)

    def test_brier_threshold(self):
        draws = np.zeros((4, 9))
        draws[:, :8] = [[5, 5, 3, 3, 5, 5, 5, 4], [5, 3, 3, 3, 5, 5, 5, 4], [5, 5, 3, 3, 5, 5, 5, 4], [5, 5, 3, 5, 5, 5, 5, 4]]
        truth = np.array([5, 5, 5, 3, 5, 5, 3.99, 4.0, 0.0])
        p = np.array([1, 0.75, 0, 0.25, 1, 1, 1, 1])
        y = np.array([1, 1, 1, 0, 1, 1, 0, 1])
        self.assertAlmostEqual(H.brier_threshold(draws, truth), float(np.mean((p - y) ** 2)), places=14)


if __name__ == "__main__":
    unittest.main()
