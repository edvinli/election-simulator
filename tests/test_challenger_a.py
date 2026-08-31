"""Challenger A: synthetic distributional validation and nested LOEO-FIT.

Synthetic fixtures only. No target-election truth is loaded and no score against any
certified outcome is computed. Monte Carlo is used solely to confirm that the
implemented sampler realises the algebraic law.
"""

from __future__ import annotations

from datetime import date
import unittest
from unittest import mock

import numpy as np

from diagnostics.election_noise_v2.challengers import loeo as loeo_mod
from diagnostics.election_noise_v2.challengers.challenger_a import (
    FROZEN_H_GRID,
    PoolTooSmall,
    draw_challenger_a,
    fit_challenger_a,
    pool_covariance,
)
from diagnostics.election_noise_v2.challengers.loeo import (
    InnerPoolTooSmall,
    OuterPoolTooSmall,
    loeo_select_bandwidth,
    production_centering,
    select_smallest_on_tie,
)
from diagnostics.election_noise_v2.challengers.rng import A_INDEX, A_KERNEL, challenger_rng

ORIGIN = date(2014, 9, 14)
HORIZON = 14


def pool(k: int = 5, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=1.5, size=(k, 9))
    c = c - c.mean(axis=1, keepdims=True)
    return c - c.mean(axis=0)


class SyntheticMoments(unittest.TestCase):
    N = 400_000

    def test_empirical_mean_approaches_zero(self):
        c = pool()
        for h in FROZEN_H_GRID:
            fit = fit_challenger_a(c, h)
            r, _ = draw_challenger_a(fit, self.N, np.random.default_rng(1),
                                     np.random.default_rng(2))
            self.assertLess(np.abs(r.mean(axis=0)).max(), 0.02, f"h={h}")

    def test_empirical_covariance_approaches_s_p(self):
        c = pool()
        s_p = pool_covariance(c)
        for h in FROZEN_H_GRID:
            fit = fit_challenger_a(c, h)
            r, _ = draw_challenger_a(fit, self.N, np.random.default_rng(3),
                                     np.random.default_rng(4))
            emp = np.cov(r.T, bias=True)
            self.assertLess(np.abs(emp - s_p).max(), 0.05, f"h={h}")

    def test_uncorrected_law_inflates_covariance_by_one_plus_h_squared(self):
        """Empirical proof that sqrt(1+h^2) is doing real work."""
        c = pool()
        s_p = pool_covariance(c)
        for h in FROZEN_H_GRID:
            fit = fit_challenger_a(c, h)
            r, _ = draw_challenger_a(fit, self.N, np.random.default_rng(5),
                                     np.random.default_rng(6))
            uncorrected = r * np.sqrt(1 + h * h)          # undo the correction
            emp = np.cov(uncorrected.T, bias=True)
            self.assertLess(np.abs(emp - (1 + h * h) * s_p).max(), 0.08, f"h={h}")

    def test_draws_are_zero_sum(self):
        c = pool()
        for h in FROZEN_H_GRID:
            fit = fit_challenger_a(c, h)
            r, _ = draw_challenger_a(fit, 5000, np.random.default_rng(7),
                                     np.random.default_rng(8))
            self.assertLess(np.abs(r.sum(axis=1)).max(), 1e-10, f"h={h}")

    def test_every_frozen_h_is_reachable_and_distinct(self):
        c = pool()
        spreads = []
        for h in FROZEN_H_GRID:
            fit = fit_challenger_a(c, h)
            r, idx = draw_challenger_a(fit, 20000, np.random.default_rng(9),
                                       np.random.default_rng(10))
            self.assertEqual(fit.h, h)
            self.assertEqual(sorted(set(idx.tolist())), list(range(c.shape[0])))
            # distance from the nearest atom grows with h
            d = np.linalg.norm(r - fit.centered[idx] / np.sqrt(1 + h * h), axis=1)
            spreads.append(float(d.mean()))
        self.assertEqual(spreads, sorted(spreads), "smoothing must increase with h")

    def test_support_is_continuous_but_spans_only_the_pool(self):
        """Disclosed property: eps lies in span{c_j}, so A is singular."""
        c = pool(k=3, seed=4)
        fit = fit_challenger_a(c, 1.00)
        r, _ = draw_challenger_a(fit, 3000, np.random.default_rng(11),
                                 np.random.default_rng(12))
        self.assertLessEqual(np.linalg.matrix_rank(r, tol=1e-9), 3)

    def test_pool_too_small_fails_loudly(self):
        with self.assertRaises(PoolTooSmall):
            fit_challenger_a(pool(k=1, seed=3)[:1], 0.25)


class NestedLoeoFit(unittest.TestCase):
    SEEDS = (12345, 24680)
    DRAWS = 400

    def test_k_inner_two_is_allowed(self):
        """K_outer = 3 necessarily produces K_inner = 2 folds; explicitly permitted."""
        res = loeo_select_bandwidth(pool(k=3, seed=21), ORIGIN, HORIZON,
                                    seeds=self.SEEDS, draws=self.DRAWS)
        self.assertEqual(res.k_outer, 3)
        self.assertIn(res.h_star, FROZEN_H_GRID)
        self.assertEqual(set(res.scores), set(FROZEN_H_GRID))

    def test_k_outer_below_three_is_excluded_outright(self):
        with self.assertRaises(OuterPoolTooSmall):
            loeo_select_bandwidth(pool(k=2, seed=22), ORIGIN, HORIZON,
                                  seeds=self.SEEDS, draws=self.DRAWS)

    def test_k_inner_one_is_prohibited_and_fails_loudly(self):
        """With the outer guard lifted, a K_outer=2 pool must still refuse K_inner=1."""
        with mock.patch.object(loeo_mod, "MIN_K_OUTER", 2):
            with self.assertRaises(InnerPoolTooSmall):
                loeo_select_bandwidth(pool(k=2, seed=23), ORIGIN, HORIZON,
                                      seeds=self.SEEDS, draws=self.DRAWS)

    def test_inner_fits_never_see_the_held_out_residual(self):
        """White-box: fold j is fitted on exactly P\\{j}, re-centered on P\\{j}."""
        raw = pool(k=4, seed=24)
        seen: list[np.ndarray] = []
        real_fit = loeo_mod.fit_challenger_a

        def spy(centered, h):
            seen.append(np.array(centered, copy=True))
            return real_fit(centered, h)

        with mock.patch.object(loeo_mod, "fit_challenger_a", spy):
            loeo_select_bandwidth(raw, ORIGIN, HORIZON, seeds=(12345,), draws=100)

        self.assertEqual(len(seen), len(FROZEN_H_GRID) * 4)
        for i, got in enumerate(seen):
            j = i % 4
            expected = production_centering(np.delete(raw, j, axis=0))[1]
            self.assertEqual(got.shape[0], 3, "K_inner must be K_outer - 1")
            np.testing.assert_allclose(got, expected, atol=1e-12)
            # the held-out row is absent from the inner pool
            for row in got:
                self.assertGreater(np.abs(row - raw[j]).max(), 1e-9)

    def test_held_out_target_uses_the_inner_centering(self):
        raw = pool(k=3, seed=25)
        for j in range(3):
            mean_bias, inner = production_centering(np.delete(raw, j, axis=0))
            held = raw[j] - mean_bias
            self.assertLess(abs(float(held.sum())), 1e-9, "held-out target stays zero-sum")
            self.assertLess(np.abs(inner.mean(axis=0)).max(), 1e-12)

    def test_production_centering_matches_the_production_algorithm(self):
        raw = pool(k=4, seed=26)
        mb, centered = production_centering(raw)
        np.testing.assert_allclose(mb, raw.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(centered, raw - raw.mean(axis=0), atol=1e-12)

    def test_exact_tie_selects_the_smallest_h(self):
        tied = {h: 1.0 for h in FROZEN_H_GRID}
        h, was_tie = select_smallest_on_tie(tied)
        self.assertEqual(h, 0.25)
        self.assertTrue(was_tie)

    def test_partial_tie_selects_the_smallest_among_the_minima(self):
        scores = {0.25: 2.0, 0.50: 1.0, 0.75: 1.0, 1.00: 3.0}
        h, was_tie = select_smallest_on_tie(scores)
        self.assertEqual(h, 0.50)
        self.assertTrue(was_tie)

    def test_strict_minimum_is_selected_when_there_is_no_tie(self):
        for winner in FROZEN_H_GRID:
            scores = {h: (0.5 if h == winner else 1.0) for h in FROZEN_H_GRID}
            h, was_tie = select_smallest_on_tie(scores)
            self.assertEqual(h, winner)
            self.assertFalse(was_tie)

    def test_selection_is_deterministic_and_seed_stable(self):
        raw = pool(k=4, seed=27)
        a = loeo_select_bandwidth(raw, ORIGIN, HORIZON, seeds=self.SEEDS, draws=self.DRAWS)
        b = loeo_select_bandwidth(raw, ORIGIN, HORIZON, seeds=self.SEEDS, draws=self.DRAWS)
        self.assertEqual(a.h_star, b.h_star)
        self.assertEqual(a.scores, b.scores)
        self.assertEqual(a.per_seed_scores, b.per_seed_scores)

    def test_all_five_seed_values_are_retained(self):
        raw = pool(k=3, seed=28)
        res = loeo_select_bandwidth(raw, ORIGIN, HORIZON, seeds=(1, 2, 3, 4, 5), draws=200)
        for h in FROZEN_H_GRID:
            self.assertEqual(len(res.per_seed_scores[h]), 5)

    def test_score_is_the_mean_over_folds(self):
        raw = pool(k=3, seed=29)
        res = loeo_select_bandwidth(raw, ORIGIN, HORIZON, seeds=(12345,), draws=200)
        for h in FROZEN_H_GRID:
            self.assertAlmostEqual(res.scores[h], float(np.mean(res.fold_scores[h])), places=12)
            self.assertEqual(len(res.fold_scores[h]), 3)


class ChronologicalLeakage(unittest.TestCase):
    """The outer pool itself must be strictly earlier than the target."""

    def test_production_pools_contain_no_future_residual(self):
        from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
        expected = {2014: (2002, 2006, 2010),
                    2018: (2002, 2006, 2010, 2014),
                    2022: (2002, 2006, 2010, 2014, 2018)}
        for target, years in expected.items():
            p = load_chronological_pp_residuals(target_election_year=target)
            self.assertEqual(tuple(int(y) for y in p.training_years), years)
            self.assertTrue(all(y < target for y in p.training_years))
            self.assertGreaterEqual(len(p.training_years), 3, "K_outer >= 3 required")


if __name__ == "__main__":
    unittest.main()
