"""Fast unit tests for the T2 arms of the calibration-2030 final round (diagnostics/calibration_2030/t2_arms.py)."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from diagnostics.calibration_2030 import t2_arms as A
from scripts.poll_aggregate.model import multinomial_shape

_CTX = None


def shared_ctx() -> dict:
    global _CTX
    if _CTX is None:
        _CTX = A.t2_context(n_draws=64)
    return _CTX


class SimplexMapTest(unittest.TestCase):
    def test_completes_clips_and_renormalizes(self):
        eight = np.array([[20.0, 5.0, 5.0, 5.0, 30.0, 8.0, 4.0, 20.0],   # REST = 3
                          [-1.0, 0.0, 10.0, 10.0, 40.0, 10.0, 10.0, 25.0]])  # REST = −4 → clipped
        out = A.simplex_map(eight)
        self.assertEqual(out.shape, (2, 9))
        np.testing.assert_allclose(out.sum(axis=1), 100.0)
        np.testing.assert_allclose(out[0], np.append(eight[0], 3.0))
        clipped = np.array([0.01, 0.01, 10, 10, 40, 10, 10, 25, 0.01])
        np.testing.assert_allclose(out[1], 100.0 * clipped / clipped.sum())
        self.assertTrue((out >= 0.01 * 100.0 / clipped.sum() - 1e-12).all())


class PureFitsTest(unittest.TestCase):
    def test_theta_fit_recovers_known_theta(self):
        sigma2, lags = 1.2e-5, np.array(A.VARIOGRAM_LAGS, dtype=float)
        for theta in (2e-4, 3e-3, 2e-2):
            gamma = sigma2 / theta * (1.0 - np.exp(-theta * lags))
            self.assertAlmostEqual(A.fit_theta(lags, gamma, sigma2) / theta, 1.0, places=5)
        # Three lags only, and a random walk hits the lower bound.
        self.assertAlmostEqual(A.fit_theta(lags[:3], sigma2 / 3e-3 * (1 - np.exp(-3e-3 * lags[:3])), sigma2) / 3e-3, 1.0, places=5)
        self.assertEqual(A.fit_theta(lags, sigma2 * lags * 1.1, sigma2), 1e-5)

    def test_k_solves_its_equation(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(0.1, 1.0, (200, 8))
        d = rng.uniform(0.0, 0.5, (200, 8))
        om = rng.uniform(0.2, 0.8, (200, 8))
        e2 = (1.7 ** 2 * s + d + om) * rng.chisquare(1, (200, 8))
        k = A.solve_k(e2, s, d, om)
        self.assertTrue(0.5 < k < 3.0)
        self.assertAlmostEqual(float(np.mean(np.mean(e2 / (k * k * s + d + om), axis=1))), 1.0, places=9)
        # Exactly calibrated synthetic errors give the true k.
        e2_exact = 1.3 ** 2 * s + d + om
        self.assertAlmostEqual(A.solve_k(e2_exact, s, d, om), 1.3, places=9)
        # No root: the nearer bound.
        self.assertEqual(A.solve_k(np.zeros_like(s), s, d, om), 0.5)
        self.assertEqual(A.solve_k(1e6 * np.ones_like(s), s, d, om), 3.0)

    def test_ou_factor_never_exceeds_random_walk(self):
        for theta in (1e-5, 1e-3, 1e-1):
            for days in (7, 28, 365, 1460):
                self.assertLessEqual(A.ou_variance_factor(theta, days), days)


class ContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = shared_ctx()

    def test_poll_target_matches_hand_computation(self):
        ctx = self.ctx
        day = date(2018, 3, 15)
        num, den, count = [0.0] * 8, 0.0, 0
        for o in ctx["polls"]:
            if day - timedelta(days=7) <= o.obs_date <= day + timedelta(days=7):
                count += 1
                den += o.sample_size
                for p in range(8):
                    num[p] += o.sample_size * o.shares[p]
        self.assertGreaterEqual(count, 3)
        np.testing.assert_allclose(A.poll_target(ctx, day), [x / den for x in num], rtol=0, atol=1e-10)
        # known_by drops later publications; a day before every window poll was published gives none.
        self.assertIsNone(A.poll_target(ctx, day, known_by=day - timedelta(days=8)))

    def test_target_count_on_pinned_inputs(self):
        ctx = self.ctx
        cells = [(o, h) for o in A.grid_origins() for h in A.HORIZONS
                 if o + timedelta(days=h + 7) <= A.POLL_CUTOFF and A.poll_target(ctx, o + timedelta(days=h)) is not None]
        self.assertEqual(len(A.grid_origins()), 421)
        self.assertEqual(len(cells), 2020)
        self.assertEqual(sum(o + timedelta(days=h) < date(2026, 9, 13) for o, h in cells), 2003)

    def test_reading_reproduces_v03_row(self):
        st = A.origin_state(self.ctx, date(2015, 5, 3))  # raises if it did not
        np.testing.assert_array_equal(st["mean"], self.ctx["row_array"][date(2015, 5, 3)])

    def test_election_shocks_use_the_open_interval(self):
        ctx, origin, e = self.ctx, date(2014, 8, 3), date(2014, 9, 14)
        st = A.origin_state(ctx, origin)
        per = st["theta"]["s_shock"] * multinomial_shape(st["latent"]) + st["theta"]["h_reset"] * st["house_shape"]
        self.assertFalse(np.any(A.election_shock_cov(ctx, origin, e)))               # E = D: no shock
        np.testing.assert_allclose(A.election_shock_cov(ctx, origin, e + timedelta(days=1)), per)
        np.testing.assert_allclose(A.election_shock_cov(ctx, origin, date(2018, 9, 10)), 2 * per)
        on_election = date(2018, 9, 9)
        if on_election in ctx["row_array"]:
            # origin = E: not in the open interval either
            self.assertFalse(np.any(A.election_shock_cov(ctx, on_election, on_election + timedelta(days=7))))
        # Ua's dynamics carries the shock exactly once.
        no = A.ua_dynamics_cov(ctx, origin, e)
        yes = A.ua_dynamics_cov(ctx, origin, e + timedelta(days=1))
        _, c_e = st["model"].consensus_reading_at(st["state"], e, st["weights"])
        _, c_e1 = st["model"].consensus_reading_at(st["state"], e + timedelta(days=1), st["weights"])
        np.testing.assert_allclose(yes - no, per + (c_e1 - c_e), atol=1e-12)

    def test_ua_dynamics_matches_the_spec_formula(self):
        ctx, origin, target = self.ctx, date(2016, 1, 17), date(2016, 7, 17)
        st = A.origin_state(ctx, origin)
        days = (target - origin).days
        formula = days * (st["theta"]["process"] * multinomial_shape(st["latent"])
                          + st["theta"]["house_drift"] * st["house_shape"])
        np.testing.assert_allclose(A.ua_dynamics_cov(ctx, origin, target), formula, rtol=1e-9, atol=1e-12)

    def test_ub1_variance_never_exceeds_ua(self):
        ctx, origin = self.ctx, date(2020, 2, 2)
        for theta in (1e-5, 1e-3, 5e-2):
            for h in A.HORIZONS:
                target = origin + timedelta(days=h)
                diff = A.ua_dynamics_cov(ctx, origin, target) - A.ub_dynamics_cov(ctx, origin, target, theta)
                self.assertGreaterEqual(float(np.linalg.eigvalsh(diff).min()), -1e-12)
        # And in the draws: shared normals, same state, the Ub1 spread is not larger.
        n, h = 2000, 365
        target = origin + timedelta(days=h)
        z_state, z_dyn, _, _ = A.cell_normals(origin, h, n)
        kw = dict(z_state=z_state, z_dyn=z_dyn, boot_rng=None, d0_base_seed=None, k=1.0)
        ua = A.state_dynamics_draws("Ua", ctx, origin, target, **kw)
        ub1 = A.state_dynamics_draws("Ub1", ctx, origin, target, **kw)
        self.assertTrue((ub1[:, :8].var(axis=0) <= ua[:, :8].var(axis=0) + 1e-9).all())

    def test_u_draws_shape_and_seed_order(self):
        ctx, origin, h = self.ctx, date(2019, 3, 10), 91
        target = origin + timedelta(days=h)
        z_state, z_dyn, z_poll, rng = A.cell_normals(origin, h, 50)
        rng2 = np.random.default_rng(A.t2_seed(origin, h))
        np.testing.assert_array_equal(z_state, rng2.standard_normal((50, 8)))
        np.testing.assert_array_equal(z_dyn, rng2.standard_normal((50, 8)))
        np.testing.assert_array_equal(z_poll, rng2.standard_normal((50, 8)))
        for arm in A.U_ARMS:
            out = A.state_dynamics_draws(arm, ctx, origin, target, z_state=z_state, z_dyn=z_dyn,
                                         boot_rng=np.random.default_rng(1), d0_base_seed=None, k=1.2)
            self.assertEqual(out.shape, (50, 9))
            np.testing.assert_allclose(out.sum(axis=1), 100.0)

    def test_d0_matches_production_national_engine(self):
        import scripts.simulator  # noqa: F401  (resolves national_engine's import cycle, as other tests do)
        from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

        ctx, origin, n, seed = self.ctx, date(2018, 5, 6), 40, A.t2_seed(date(2018, 5, 6), 126)
        target = date(2018, 9, 9)  # h = 126: production's 112-day cap applies
        with tempfile.TemporaryDirectory() as tmp:
            agg = Path(tmp) / "poll_aggregate" / "v0_2"
            agg.mkdir(parents=True)
            os.symlink(ctx["v03_file"], agg / "swedishpolls_aggregate_timeseries.csv")
            os.symlink(ctx["polls_file"], agg / "individual_polls.csv")
            prod = generate_national_vote_shares(as_of=origin, election_date=target, samples=n, seed=seed, data_dir=tmp,
                                                 polls_file=A.ROOT / "data/processed/pollofpolls/swedishpolls_individual_polls.csv",
                                                 elections_file=A.ROOT / "data/processed/elections/riksdag_election_results.csv")
        ours = A.state_dynamics_draws("D0", ctx, origin, target, z_state=np.zeros((n, 8)), z_dyn=np.zeros((n, 8)),
                                      boot_rng=None, d0_base_seed=seed)
        self.assertEqual(prod.diagnostics["dynamics_eval_horizon"], A.d0_eligible(ctx, origin, 126)[1])
        np.testing.assert_allclose(ours, A.simplex_map(prod.base_comp_matrix[:, :8]), rtol=0, atol=1e-10)
        np.testing.assert_allclose(ours, prod.base_comp_matrix, rtol=0, atol=1e-8)

    def test_caches_are_identical_recomputations(self):
        ctx = self.ctx
        origin = date(2013, 6, 30)
        first = A.omega_at(ctx, origin)
        theta = A.theta_at(ctx, origin)
        ctx["cache"]["omega"].pop(origin)
        ctx["cache"]["theta"].pop(origin)
        again = A.omega_at(ctx, origin)
        np.testing.assert_array_equal(first[0], again[0])
        self.assertEqual(first[1], again[1])
        self.assertEqual(theta, A.theta_at(ctx, origin))
        self.assertTrue(math.isfinite(theta))


if __name__ == "__main__":
    unittest.main()
