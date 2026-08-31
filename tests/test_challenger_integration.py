"""Challenger RNG contract and downstream integration.

The integration smoke test drives real historical *inputs* purely as plumbing. It
never loads a certified election result and computes no energy score, CRPS, Brier,
calibration or any Challenger-vs-CONTROL comparison. It asserts only structural
validity of the pipeline output.
"""

from __future__ import annotations

from datetime import date
import unittest

import numpy as np

from diagnostics.election_noise_v2.challengers.draws import (
    MODEL_A,
    MODEL_B,
    challenger_iso_draws,
    challenger_residual_draws,
)
from diagnostics.election_noise_v2.challengers.rng import (
    A_INDEX,
    A_KERNEL,
    A_LOEO,
    B_NORMAL,
    ForbiddenSeedToken,
    challenger_rng,
    challenger_subseed,
)
from diagnostics.election_noise_v2.control_baseline.harness.rng import (
    CHALLENGER_RESERVED_TOKENS,
    CONTROL_TOKENS,
    FROZEN_SEEDS,
)

ORIGIN = date(2014, 9, 14)
HORIZON = 14
TARGETS = (2014, 2018, 2022)


class SeedContract(unittest.TestCase):
    def test_only_reserved_tokens_are_accepted(self):
        for label in (A_INDEX, A_KERNEL, A_LOEO, B_NORMAL):
            self.assertIn(label, CHALLENGER_RESERVED_TOKENS)
            self.assertIsInstance(challenger_subseed(12345, ORIGIN, HORIZON, label), int)

    def test_control_streams_are_refused(self):
        for label in CONTROL_TOKENS:
            with self.assertRaises(ForbiddenSeedToken):
                challenger_subseed(12345, ORIGIN, HORIZON, label)

    def test_unreserved_tokens_are_refused(self):
        for label in ("election_layer_v2", "opinion_state", "made_up_token", ""):
            with self.assertRaises(ForbiddenSeedToken):
                challenger_subseed(12345, ORIGIN, HORIZON, label)

    def test_subseed_matches_the_production_sha256_convention(self):
        import hashlib
        for seed in FROZEN_SEEDS:
            for label in CHALLENGER_RESERVED_TOKENS:
                token = f"{seed}:{ORIGIN.isoformat()}:{HORIZON}:{label}".encode()
                expected = int(hashlib.sha256(token).hexdigest()[:8], 16) % 2_147_483_647
                self.assertEqual(challenger_subseed(seed, ORIGIN, HORIZON, label), expected)

    def test_streams_are_mutually_distinct(self):
        subs = {l: challenger_subseed(12345, ORIGIN, HORIZON, l)
                for l in CHALLENGER_RESERVED_TOKENS}
        self.assertEqual(len(set(subs.values())), len(subs))

    def test_index_and_kernel_streams_are_independent(self):
        a = challenger_rng(12345, ORIGIN, HORIZON, A_INDEX).standard_normal(2000)
        b = challenger_rng(12345, ORIGIN, HORIZON, A_KERNEL).standard_normal(2000)
        self.assertLess(abs(float(np.corrcoef(a, b)[0, 1])), 0.06)

    def test_spawn_keys_give_distinct_reproducible_substreams(self):
        f = lambda key: challenger_rng(12345, ORIGIN, HORIZON, A_LOEO, key).standard_normal(50)
        np.testing.assert_array_equal(f((0, 1, 2, 0)), f((0, 1, 2, 0)))
        self.assertFalse(np.array_equal(f((0, 1, 2, 0)), f((0, 1, 2, 1))))
        self.assertFalse(np.array_equal(f((0, 1, 2, 0)), f((1, 1, 2, 0))))


class Determinism(unittest.TestCase):
    """Identical (model, case, seed, N) must be bit-identical."""

    def test_challenger_a_repeat_is_bit_identical(self):
        for target in TARGETS:
            for seed in (FROZEN_SEEDS[0], FROZEN_SEEDS[1]):
                a = challenger_residual_draws(MODEL_A, target, seed, 500, ORIGIN, HORIZON, h=0.50)
                b = challenger_residual_draws(MODEL_A, target, seed, 500, ORIGIN, HORIZON, h=0.50)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])

    def test_challenger_b_repeat_is_bit_identical(self):
        for target in TARGETS:
            for seed in (FROZEN_SEEDS[0], FROZEN_SEEDS[2]):
                a = challenger_residual_draws(MODEL_B, target, seed, 500, ORIGIN, HORIZON)
                b = challenger_residual_draws(MODEL_B, target, seed, 500, ORIGIN, HORIZON)
                np.testing.assert_array_equal(a[0], b[0])

    def test_different_seeds_give_different_draws(self):
        a = challenger_residual_draws(MODEL_B, 2018, FROZEN_SEEDS[0], 500, ORIGIN, HORIZON)[0]
        b = challenger_residual_draws(MODEL_B, 2018, FROZEN_SEEDS[1], 500, ORIGIN, HORIZON)[0]
        self.assertFalse(np.array_equal(a, b))

    def test_different_h_gives_different_draws(self):
        a = challenger_residual_draws(MODEL_A, 2018, 12345, 500, ORIGIN, HORIZON, h=0.25)[0]
        b = challenger_residual_draws(MODEL_A, 2018, 12345, 500, ORIGIN, HORIZON, h=1.00)[0]
        self.assertFalse(np.array_equal(a, b))

    def test_a_and_b_do_not_share_a_stream(self):
        a = challenger_residual_draws(MODEL_A, 2018, 12345, 500, ORIGIN, HORIZON, h=0.50)[0]
        b = challenger_residual_draws(MODEL_B, 2018, 12345, 500, ORIGIN, HORIZON)[0]
        self.assertFalse(np.array_equal(a, b))

    def test_full_iso_path_repeat_is_bit_identical(self):
        x = challenger_iso_draws(MODEL_B, 2014, 12345, 120)
        y = challenger_iso_draws(MODEL_B, 2014, 12345, 120)
        np.testing.assert_array_equal(x.votes_pct, y.votes_pct)
        np.testing.assert_array_equal(x.seats, y.seats)
        np.testing.assert_array_equal(x.lambdas, y.lambdas)

    def test_a_requires_explicit_h_and_b_refuses_one(self):
        with self.assertRaises(ValueError):
            challenger_residual_draws(MODEL_A, 2014, 12345, 10, ORIGIN, HORIZON, h=None)
        with self.assertRaises(ValueError):
            challenger_residual_draws(MODEL_B, 2014, 12345, 10, ORIGIN, HORIZON, h=0.5)


class IntegrationSmoke(unittest.TestCase):
    """transfer -> chronological geography -> historically correct allocator.

    Structural validity only. No certified result is read and no score is computed.
    """

    N = 120

    def _check(self, d):
        v = d.votes_pct
        self.assertEqual(v.shape, (self.N, 9))
        self.assertTrue(np.all(v >= 0.0), "vote shares must be non-negative")
        self.assertLess(np.abs(v.sum(axis=1) - 100.0).max(), 1e-9, "compositions sum to 100")
        s = d.seats
        self.assertEqual(s.shape, (self.N, 8))
        self.assertEqual(s.dtype.kind, "i")
        self.assertTrue(np.all(s >= 0))
        self.assertTrue(np.all(s.sum(axis=1) == 349), "exactly 349 seats")
        self.assertTrue(np.all((d.lambdas >= 0.0) & (d.lambdas <= 1.0)))

    def test_challenger_b_flows_through_the_full_path(self):
        for target in TARGETS:
            with self.subTest(target=target):
                self._check(challenger_iso_draws(MODEL_B, target, 12345, self.N))

    def test_challenger_a_flows_through_the_full_path(self):
        for target in TARGETS:
            for h in (0.25, 1.00):
                with self.subTest(target=target, h=h):
                    self._check(challenger_iso_draws(MODEL_A, target, 12345, self.N, h=h))

    def test_transfer_is_the_unmodified_production_function(self):
        from diagnostics.election_noise_v2.challengers import draws as dmod
        from scripts.election_layer_v2.transfer import apply_batch_simplex_transfer
        self.assertIs(dmod.apply_batch_simplex_transfer, apply_batch_simplex_transfer)

    def test_geography_and_law_dispatch_are_the_frozen_ones(self):
        from diagnostics.election_noise_v2.challengers import draws as dmod
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import isolated
        self.assertIs(dmod.votes_to_seats, isolated.votes_to_seats)
        self.assertIs(dmod.consensus_vector, isolated.consensus_vector)

    def test_oracle_geography_remains_forbidden(self):
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import isolated
        with self.assertRaises(RuntimeError):
            isolated.assert_geography_mode("oracle")


if __name__ == "__main__":
    unittest.main()
