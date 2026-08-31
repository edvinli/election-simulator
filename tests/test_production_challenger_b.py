"""Production ElectionNoise B must be the frozen Challenger B law, exactly.

The production implementation in ``scripts/vote_share_calibration/election_noise_b.py``
is written independently of the frozen research implementation in
``diagnostics/election_noise_v2/challengers/challenger_b.py``. These tests assert
the two agree bit-for-bit on every intermediate quantity, on the generated draws,
and after the downstream simplex transfer — so promotion carries no scientific
change, only a change of location.

They also assert the promotion is additive: no file inside either freeze changed.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from diagnostics.election_noise_v2.challengers.challenger_b import (
    draw_challenger_b,
    fit_challenger_b,
    symmetric_factor as frozen_factor,
    zero_sum_projector as frozen_p9,
)
from diagnostics.election_noise_v2.challengers.rng import B_NORMAL, challenger_rng
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
from scripts.vote_share_calibration.election_noise_b import (
    LEGACY_MODEL_ID,
    MODEL_ID,
    NonPSDCovariance,
    derive_election_noise_b_seed,
    draw_election_noise_b,
    election_noise_b_residuals,
    fit_election_noise_b,
    symmetric_factor,
    zero_sum_projector,
)


def synth(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=1.5, size=(k, 9))
    c = c - c.mean(axis=1, keepdims=True)
    return c - c.mean(axis=0)


POOLS = [synth(3, 2), synth(4, 5), synth(5, 7), synth(6, 9)]


class FrozenEquivalence(unittest.TestCase):
    """Every intermediate quantity must be bit-identical to frozen Challenger B."""

    def test_projector_identical(self):
        np.testing.assert_array_equal(zero_sum_projector(9), frozen_p9(9))

    def test_all_intermediates_bit_identical(self):
        for c in POOLS:
            with self.subTest(k=c.shape[0]):
                p = fit_election_noise_b(c)
                f = fit_challenger_b(c)
                np.testing.assert_array_equal(p.centered, f.centered)
                np.testing.assert_array_equal(p.s_p, f.s_p)
                np.testing.assert_array_equal(p.p9, f.p9)
                self.assertEqual(p.tau_sq, f.tau_sq)
                np.testing.assert_array_equal(p.target, f.target)
                self.assertEqual(p.d_sq, f.d_sq)
                self.assertEqual(p.bbar_sq, f.bbar_sq)
                self.assertEqual(p.b_sq, f.b_sq)
                self.assertEqual(p.delta, f.delta)
                np.testing.assert_array_equal(p.sigma_lw, f.sigma_lw)
                np.testing.assert_array_equal(p.sigma_tilde, f.sigma_tilde)
                self.assertEqual(p.k, f.k)
                self.assertEqual(p.bessel_factor, f.bessel_factor)

    def test_symmetric_factor_identical(self):
        for c in POOLS:
            p, f = fit_election_noise_b(c), fit_challenger_b(c)
            np.testing.assert_array_equal(symmetric_factor(p.sigma_tilde),
                                          frozen_factor(f.sigma_tilde))

    def test_draws_bit_identical_under_the_same_seed_convention(self):
        """The production RNG and the frozen reserved-token RNG must coincide."""
        for c in POOLS:
            for base_seed in (12345, 24680):
                origin, horizon = date(2026, 8, 24), 20
                prod, _, sub = election_noise_b_residuals(c, 2000, base_seed, origin, horizon)
                frozen = draw_challenger_b(
                    fit_challenger_b(c), 2000,
                    challenger_rng(base_seed, origin, horizon, B_NORMAL))
                np.testing.assert_array_equal(prod, frozen)
                self.assertEqual(sub, _frozen_subseed(base_seed, origin, horizon))

    def test_seed_derivation_matches_the_reserved_token(self):
        for base_seed in (12345, 24680, 98765):
            for h in (14, 20, 28):
                self.assertEqual(
                    derive_election_noise_b_seed(base_seed, date(2026, 8, 24), h),
                    _frozen_subseed(base_seed, date(2026, 8, 24), h))

    def test_simplex_transfer_results_identical(self):
        """Equivalence survives the unchanged downstream transfer."""
        c = synth(6, 11)
        base = np.tile(np.array([21.9, 6.2, 6.3, 5.2, 30.1, 8.0, 4.9, 17.4, 0.0]), (500, 1))
        base = base / base.sum(axis=1, keepdims=True) * 100.0
        prod, _, _ = election_noise_b_residuals(c, 500, 12345, date(2026, 8, 24), 20)
        frozen = draw_challenger_b(fit_challenger_b(c), 500,
                                   challenger_rng(12345, date(2026, 8, 24), 20, B_NORMAL))
        vp, lp = apply_batch_simplex_transfer(base, prod)
        vf, lf = apply_batch_simplex_transfer(base, frozen)
        np.testing.assert_array_equal(vp, vf)
        np.testing.assert_array_equal(lp, lf)

    def test_equivalence_on_the_real_2026_training_pool(self):
        pool = load_chronological_pp_residuals(target_election_year=2026)
        c = pool.centered_residuals_matrix
        p, f = fit_election_noise_b(c), fit_challenger_b(c)
        np.testing.assert_array_equal(p.sigma_tilde, f.sigma_tilde)
        prod, _, _ = election_noise_b_residuals(c, 5000, 12345, date(2026, 8, 24), 20)
        frozen = draw_challenger_b(f, 5000, challenger_rng(12345, date(2026, 8, 24), 20, B_NORMAL))
        np.testing.assert_array_equal(prod, frozen)


def _frozen_subseed(base_seed: int, origin: date, horizon: int) -> int:
    import hashlib
    token = f"{base_seed}:{origin.isoformat()}:{horizon}:{B_NORMAL}".encode()
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % 2_147_483_647


class LawProperties(unittest.TestCase):
    def test_no_tunable_parameters(self):
        import inspect
        self.assertEqual(list(inspect.signature(fit_election_noise_b).parameters), ["centered"])

    def test_zero_sum_and_psd(self):
        for c in POOLS:
            f = fit_election_noise_b(c)
            self.assertLess(np.abs(f.sigma_tilde @ np.ones(9)).max(), 1e-12)
            self.assertGreater(np.linalg.eigvalsh(f.sigma_tilde).min(), -1e-12)
            self.assertGreaterEqual(f.delta, 0.0)
            self.assertLessEqual(f.delta, 1.0)

    def test_draws_are_zero_sum(self):
        f = fit_election_noise_b(synth(6, 3))
        r = draw_election_noise_b(f, 20000, np.random.default_rng(1))
        self.assertLess(np.abs(r.sum(axis=1)).max(), 1e-10)

    def test_degenerate_pool_uses_the_preregistered_limit(self):
        p9 = zero_sum_projector(9)
        w, v = np.linalg.eigh(p9)
        c = (v[:, w > 0.5].T) * np.sqrt(8.0 * 0.7)
        f = fit_election_noise_b(c)
        self.assertLess(f.d_sq, 1e-20)
        self.assertEqual(f.delta, 1.0)

    def test_indefinite_covariance_raises_rather_than_clipping(self):
        bad = np.eye(9); bad[0, 0] = -0.5
        with self.assertRaises(NonPSDCovariance):
            symmetric_factor(bad)


class PromotionIsAdditive(unittest.TestCase):
    """No frozen file may change. Promotion adds modules; it edits nothing frozen."""

    def _locked(self):
        a2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
        ch = REPO_ROOT / "diagnostics/election_noise_v2/challengers"
        ev = json.loads((a2 / "evaluator_freeze.json").read_text())
        cf = json.loads((ch / "challenger_implementation_freeze.json").read_text())
        return (set(ev["metric_implementation_hashes"])
                | set(ev["evaluator_import_closure_hashes"])
                | set(cf["frozen_dependency_hashes"])
                | set(cf["import_closure_hashes"])
                | {r for g in cf["implementation_hashes"].values() for r in g})

    def test_new_production_modules_are_not_inside_either_freeze(self):
        locked = self._locked()
        for new in ("scripts/vote_share_calibration/election_noise_b.py",
                    "scripts/vote_share_calibration/production_national_engine.py",
                    "scripts/simulator/production_runner.py"):
            self.assertNotIn(new, locked, f"{new} must not be a frozen file")

    def test_both_freezes_still_verify(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze as ev
        from diagnostics.election_noise_v2.challengers import freeze_challengers as cf
        self.assertEqual(ev.verify()["drift"], [])
        self.assertEqual(cf.verify()["drift"], [])

    def test_legacy_control_law_is_preserved(self):
        from scripts.vote_share_calibration.models import apply_vote_share_models
        self.assertEqual(LEGACY_MODEL_ID, "pp_centered_noise")
        self.assertTrue(callable(apply_vote_share_models))
        # the legacy law is still reachable through the unmodified production engine
        from scripts.vote_share_calibration.national_engine import generate_national_vote_shares
        self.assertTrue(callable(generate_national_vote_shares))

    def test_model_ids_are_distinct(self):
        self.assertNotEqual(MODEL_ID, LEGACY_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
