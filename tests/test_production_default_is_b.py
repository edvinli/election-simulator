"""The ordinary production default must be the adopted Challenger-B ElectionNoise law.

These tests use small sample counts: they check the wiring, the defaults, the
metadata and that CONTROL is still explicitly reachable. The full-scale certification
that the default path reproduces the Part-6A certified forecast lives in
``diagnostics/election_noise_v2/production_promotion/default_path_certification.json``.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMO = REPO_ROOT / "diagnostics/election_noise_v2/production_promotion"

from scripts.simulator.config import (
    ADOPTED_ELECTION_NOISE_CANDIDATE,
    BENCHMARK_LINEAGE_CANDIDATE,
    MODEL_VERSION,
    RELEASE_TAG,
)
from scripts.simulator.engine import simulate_election
from scripts.vote_share_calibration.election_noise_b import (
    LEGACY_MODEL_ID,
    MODEL_ID,
    fit_election_noise_b,
)
from scripts.vote_share_calibration.national_engine import generate_national_vote_shares

AS_OF, ELECTION, N, SEED = "2026-08-24", "2026-09-13", 250, 12345


class Defaults(unittest.TestCase):
    def test_national_engine_defaults_to_the_adopted_law(self):
        d = inspect.signature(generate_national_vote_shares).parameters["noise_model"].default
        self.assertEqual(d, MODEL_ID)
        self.assertEqual(MODEL_ID, "pp_lw_gaussian")

    def test_simulate_election_defaults_to_the_adopted_law(self):
        d = inspect.signature(simulate_election).parameters["noise_model"].default
        self.assertEqual(d, MODEL_ID)

    def test_version_follows_the_repository_convention(self):
        self.assertEqual(MODEL_VERSION, "1.1.0-rc1")
        self.assertEqual(RELEASE_TAG, "election-simulator-v1.1-rc1")
        self.assertEqual(ADOPTED_ELECTION_NOISE_CANDIDATE, "B")
        self.assertEqual(BENCHMARK_LINEAGE_CANDIDATE, "A")
        self.assertTrue(MODEL_VERSION.endswith("-rc1"),
                        "the repository convention has not declared a stable release")

    def test_unknown_noise_model_is_refused(self):
        with self.assertRaises(ValueError):
            generate_national_vote_shares(as_of=AS_OF, election_date=ELECTION,
                                          samples=10, seed=SEED, noise_model="nonsense")


class DefaultPathIsB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = simulate_election(as_of=AS_OF, election_date=ELECTION,
                                        samples=N, seed=SEED)               # no override
        cls.explicit_b = simulate_election(as_of=AS_OF, election_date=ELECTION, samples=N,
                                           seed=SEED, noise_model=MODEL_ID)
        cls.control = simulate_election(as_of=AS_OF, election_date=ELECTION, samples=N,
                                        seed=SEED, noise_model=LEGACY_MODEL_ID)

    def test_default_equals_explicit_b_bit_for_bit(self):
        np.testing.assert_array_equal(self.default.vote_shares_matrix,
                                      self.explicit_b.vote_shares_matrix)
        np.testing.assert_array_equal(self.default.seats_matrix, self.explicit_b.seats_matrix)

    def test_default_is_not_control(self):
        self.assertFalse(np.array_equal(self.default.vote_shares_matrix,
                                        self.control.vote_shares_matrix))

    def test_manifest_declares_the_adopted_law_and_version(self):
        self.assertEqual(self.default.manifest["model_config"]["noise_model"], MODEL_ID)
        self.assertEqual(self.default.manifest["model_version"], "1.1.0-rc1")

    def test_control_manifest_declares_the_legacy_law(self):
        self.assertEqual(self.control.manifest["model_config"]["noise_model"], LEGACY_MODEL_ID)

    def test_deterministic_repeat_is_bit_identical(self):
        again = simulate_election(as_of=AS_OF, election_date=ELECTION, samples=N, seed=SEED)
        np.testing.assert_array_equal(self.default.vote_shares_matrix, again.vote_shares_matrix)
        np.testing.assert_array_equal(self.default.seats_matrix, again.seats_matrix)

    def test_structural_validity(self):
        for r in (self.default, self.control):
            v, s = r.vote_shares_matrix, r.seats_matrix
            self.assertTrue(np.all(np.isfinite(v)))
            self.assertTrue(np.all(v >= 0.0))
            self.assertLess(np.abs(v.sum(axis=1) - 100.0).max(), 1e-9)
            self.assertTrue(np.all(s.sum(axis=1) == 349))
            self.assertTrue(np.all(s >= 0))

    def test_lambda_in_unit_interval_under_the_default(self):
        nat = generate_national_vote_shares(as_of=AS_OF, election_date=ELECTION,
                                            samples=N, seed=SEED)
        self.assertTrue(np.all((nat.lambdas >= 0.0) & (nat.lambdas <= 1.0)))
        self.assertEqual(nat.diagnostics["election_noise_model"], MODEL_ID)

    def test_upstream_is_identical_across_the_two_laws(self):
        b = generate_national_vote_shares(as_of=AS_OF, election_date=ELECTION,
                                          samples=N, seed=SEED)
        c = generate_national_vote_shares(as_of=AS_OF, election_date=ELECTION, samples=N,
                                          seed=SEED, noise_model=LEGACY_MODEL_ID)
        np.testing.assert_array_equal(b.opinion_state_draws, c.opinion_state_draws)
        np.testing.assert_array_equal(b.dynamics_deltas, c.dynamics_deltas)
        np.testing.assert_array_equal(b.base_comp_matrix, c.base_comp_matrix)
        self.assertEqual(tuple(b.training_years), tuple(c.training_years))


class ControlIsPreserved(unittest.TestCase):
    def test_legacy_law_implementation_is_unmodified(self):
        """The mathematical CONTROL implementation must still be the frozen one."""
        import json as _json
        cf = _json.loads((REPO_ROOT / "diagnostics/election_noise_v2/challengers"
                          / "challenger_implementation_freeze.json").read_text())
        import hashlib
        recorded = {**cf["frozen_dependency_hashes"], **cf["import_closure_hashes"]}
        for rel in ("scripts/vote_share_calibration/models.py",
                    "scripts/election_layer_v2/transfer.py",
                    "scripts/election_layer_v2/residuals_pool.py",
                    "diagnostics/election_noise_v2/challengers/challenger_b.py"):
            expected = recorded[rel]["working_tree_sha256"]
            actual = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"{rel} must not change")

    def test_control_remains_selectable(self):
        r = simulate_election(as_of=AS_OF, election_date=ELECTION, samples=50, seed=SEED,
                              noise_model=LEGACY_MODEL_ID)
        self.assertEqual(r.manifest["model_config"]["noise_model"], LEGACY_MODEL_ID)
        self.assertTrue(np.all(r.seats_matrix.sum(axis=1) == 349))


class BLawUnchanged(unittest.TestCase):
    """The adopted law must be exactly the Part-6A certified mathematics."""

    def test_still_bit_identical_to_the_frozen_research_implementation(self):
        from diagnostics.election_noise_v2.challengers.challenger_b import fit_challenger_b
        rng = np.random.default_rng(0)
        c = rng.normal(size=(6, 9))
        c = c - c.mean(axis=1, keepdims=True)
        c = c - c.mean(axis=0)
        p, f = fit_election_noise_b(c), fit_challenger_b(c)
        np.testing.assert_array_equal(p.sigma_tilde, f.sigma_tilde)
        self.assertEqual(p.delta, f.delta)
        self.assertEqual(p.tau_sq, f.tau_sq)


class DefaultPathCertification(unittest.TestCase):
    """Assertions against the full-scale N=100 000 certification artifact."""

    @classmethod
    def setUpClass(cls):
        p = PROMO / "default_path_certification.json"
        if not p.exists():
            raise unittest.SkipTest("default-path certification not yet produced")
        cls.c = json.loads(p.read_text())

    def test_certified(self):
        self.assertTrue(self.c["certified"])

    def test_default_reproduced_the_part6a_forecast_exactly(self):
        self.assertTrue(self.c["default_vs_certified_B"]["identical"])
        self.assertEqual(self.c["default_vs_certified_B"]["differences"], [])

    def test_control_reproduced_the_archived_forecast_exactly(self):
        self.assertTrue(self.c["control_vs_certified_CONTROL"]["identical"])
        self.assertEqual(self.c["control_vs_certified_CONTROL"]["differences"], [])

    def test_no_noise_model_argument_was_passed(self):
        self.assertFalse(self.c["configuration"]["noise_model_argument_passed"])
        self.assertFalse(self.c["configuration"]["polling_inputs_refreshed"])
        self.assertEqual(self.c["configuration"]["as_of"], "2026-08-24")
        self.assertEqual(self.c["configuration"]["samples"], 100000)
        self.assertEqual(self.c["configuration"]["seed"], 12345)

    def test_metadata_declares_b(self):
        self.assertTrue(self.c["metadata"]["default_is_adopted_b"])
        self.assertEqual(self.c["metadata"]["default_manifest_model_version"], "1.1.0-rc1")
        self.assertEqual(self.c["metadata"]["control_manifest_noise_model"], LEGACY_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
