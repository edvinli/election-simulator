"""Model 1.2.1-rc1: corrected election-noise residuals for 2030-target forecasts only.

A forecast of the 2030 election (or later) fits ElectionNoise B on the
residuals of election-noise residual amendments 001 and 002: the certified
result minus the election-eve consensus of ``eve_consensus.py``. A forecast of
2026 or earlier keeps the unamended pool and model 1.2.0-rc1, bit for bit
(``docs/election_simulator_model_1_2_1.md``).
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.simulator.config import (
    CORRECTED_RESIDUAL_CONSENSUS_RULE,
    CORRECTED_RESIDUALS_FIRST_TARGET_YEAR,
    DEFAULT_ELECTION_DATE,
    MODEL_VERSION,
    PREVIOUS_MODEL_VERSION,
    PREVIOUS_RELEASE_TAG,
    RELEASE_TAG,
    UNAMENDED_RESIDUAL_CONSENSUS_RULE,
    model_version_for,
    release_tag_for,
    residual_consensus_rule_for,
)
from scripts.simulator.engine import simulate_election
from scripts.election_layer_v2.config import ALL_HISTORICAL_ELECTIONS
from scripts.election_layer_v2.production_residuals_pool import (
    _load_pp_residuals,
    consensus_builder_for,
    load_eve_pp_residuals,
    load_production_pp_residuals,
)
from scripts.election_layer_v2.residuals_pool import load_chronological_pp_residuals
from scripts.election_residuals.config import (
    ALL_CATEGORIES,
    DEFAULT_ELECTIONS_FILE,
    DEFAULT_POLLS_FILE,
)
from scripts.election_residuals.consensus import build_election_polling_consensus
from scripts.election_residuals.eve_consensus import RULE_ID, build_eve_polling_consensus
from scripts.elections.load import load_election_targets_for_forecasting
from scripts.forecast_history.effective_inputs import MODEL_FILES, EffectiveInputs
from scripts.vote_share_calibration import national_engine
from scripts.vote_share_calibration.election_noise_b import fit_election_noise_b

REPO_ROOT = Path(__file__).resolve().parents[1]
UNAMENDED_TARGETS = (2010, 2014, 2018, 2022, 2026)
#: ``model_config_hash`` of a default 2026-target forecast before this change
#: (origin/main 5c7e87a). A dictionary hash, so it is platform-independent.
MODEL_CONFIG_HASH_2026 = "a4d24f37b62fc2a9115ddb394c028f8739193dd5dc9152446a8cdd96b8fb6233"


def _assert_pools_identical(case: unittest.TestCase, a, b) -> None:
    case.assertEqual(a.target_election_year, b.target_election_year)
    case.assertEqual(a.training_years, b.training_years)
    for field in ("residuals_matrix", "mean_bias_pp", "centered_residuals_matrix"):
        case.assertTrue(np.array_equal(getattr(a, field), getattr(b, field)), field)
        case.assertEqual(getattr(a, field).tobytes(), getattr(b, field).tobytes(), field)


def _total_variance(pool) -> float:
    """``pool_summary``'s total variance (amendment 002 evidence), pp^2."""
    fit = fit_election_noise_b(pool.centered_residuals_matrix)
    return float(np.trace(fit.delta * fit.target + (1.0 - fit.delta) * fit.s_p))


class VersionRouting(unittest.TestCase):
    def test_current_model_and_its_predecessor(self):
        self.assertEqual(MODEL_VERSION, "1.2.1-rc1")
        self.assertEqual(RELEASE_TAG, "election-simulator-v1.2.1-rc1")
        self.assertEqual(PREVIOUS_MODEL_VERSION, "1.2.0-rc1")
        self.assertEqual(PREVIOUS_RELEASE_TAG, "election-simulator-v1.2-rc1")
        self.assertEqual(CORRECTED_RESIDUALS_FIRST_TARGET_YEAR, 2030)

    def test_the_default_target_is_forecast_by_the_current_model(self):
        year = date.fromisoformat(DEFAULT_ELECTION_DATE).year
        self.assertEqual(model_version_for(year), MODEL_VERSION)
        self.assertEqual(release_tag_for(year), RELEASE_TAG)
        self.assertEqual(residual_consensus_rule_for(year), CORRECTED_RESIDUAL_CONSENSUS_RULE)

    def test_2026_and_earlier_targets_keep_1_2_0(self):
        for year in UNAMENDED_TARGETS:
            self.assertEqual(model_version_for(year), "1.2.0-rc1", year)
            self.assertEqual(release_tag_for(year), "election-simulator-v1.2-rc1", year)
            self.assertEqual(residual_consensus_rule_for(year), UNAMENDED_RESIDUAL_CONSENSUS_RULE, year)
            self.assertIs(consensus_builder_for(year), build_election_polling_consensus)

    def test_2030_and_later_targets_use_1_2_1(self):
        for year in (2030, 2034):
            self.assertEqual(model_version_for(year), "1.2.1-rc1", year)
            self.assertEqual(residual_consensus_rule_for(year), RULE_ID, year)
            self.assertIs(consensus_builder_for(year), build_eve_polling_consensus)

    def test_the_rule_names_the_amended_builder(self):
        self.assertEqual(CORRECTED_RESIDUAL_CONSENSUS_RULE, RULE_ID)
        self.assertEqual(RULE_ID, "election-eve-consensus-amendments-001-002")
        self.assertNotEqual(UNAMENDED_RESIDUAL_CONSENSUS_RULE, CORRECTED_RESIDUAL_CONSENSUS_RULE)

    def test_a_year_must_be_an_int(self):
        for bad in ("2030", 2030.0, True):
            with self.assertRaises(TypeError):
                model_version_for(bad)
            with self.assertRaises(TypeError):
                residual_consensus_rule_for(bad)


class CorrectedPool(unittest.TestCase):
    def test_the_parameterised_loader_reproduces_the_frozen_one_exactly(self):
        """Given the unamended builder, the 1.2.1 loader is the frozen loader."""
        for year in (*UNAMENDED_TARGETS, 2030):
            _assert_pools_identical(
                self, _load_pp_residuals(build_election_polling_consensus, year),
                load_chronological_pp_residuals(year))

    def test_2026_and_earlier_targets_get_the_unamended_pool(self):
        for year in UNAMENDED_TARGETS:
            _assert_pools_identical(self, load_production_pp_residuals(year),
                                    load_chronological_pp_residuals(year))

    def test_2030_gets_the_eve_consensus_residuals_with_k_6(self):
        pool = load_production_pp_residuals(2030)
        _assert_pools_identical(self, pool, load_eve_pp_residuals(2030))
        self.assertEqual(pool.training_years, (2002, 2006, 2010, 2014, 2018, 2022))
        polls = pd.read_csv(DEFAULT_POLLS_FILE)
        targets = load_election_targets_for_forecasting(DEFAULT_ELECTIONS_FILE)
        for row, election in zip(pool.residuals_matrix, sorted(ALL_HISTORICAL_ELECTIONS)):
            consensus = build_eve_polling_consensus(election, polls)
            expected = np.array([targets[election][c] - consensus.consensus_composition[c]
                                 for c in ALL_CATEGORIES])
            expected -= expected.sum() / len(expected)
            np.testing.assert_allclose(row, expected, rtol=0, atol=1e-12)

    def test_the_corrected_pool_differs_from_the_unamended_one(self):
        corrected = load_production_pp_residuals(2030)
        unamended = load_chronological_pp_residuals(2030)
        changed = [y for y, a, b in zip(corrected.training_years, corrected.residuals_matrix,
                                        unamended.residuals_matrix) if not np.array_equal(a, b)]
        # 001 changes 2006-2022; 002 changes 2002 (and again 2006, 2022).
        self.assertEqual(changed, [2002, 2006, 2010, 2014, 2018, 2022])

    def test_total_variance_matches_the_amendment_002_evidence(self):
        self.assertAlmostEqual(_total_variance(load_production_pp_residuals(2030)), 8.389, places=3)
        self.assertAlmostEqual(_total_variance(load_production_pp_residuals(2026)), 7.2945, places=4)


class ProductionConsumers(unittest.TestCase):
    """Every production pool consumer goes through the router, nothing else."""

    CONSUMERS = (
        "scripts/vote_share_calibration/national_engine.py",
        "scripts/vote_share_calibration/production_national_engine.py",
        "scripts/forecast_history/projection_simulator.py",
        "scripts/forecast_history/campaign_paths.py",
    )

    def test_consumers_call_the_router(self):
        for rel in self.CONSUMERS:
            tree = ast.parse((REPO_ROOT / rel).read_text())
            called = {node.func.id for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
            self.assertIn("load_production_pp_residuals", called, rel)
            self.assertNotIn("load_chronological_pp_residuals", called, rel)

    def test_the_new_files_are_fingerprinted(self):
        self.assertIn("scripts/election_residuals/eve_consensus.py", MODEL_FILES)
        self.assertIn("scripts/election_layer_v2/production_residuals_pool.py", MODEL_FILES)


class NationalEngine(unittest.TestCase):
    N, SEED = 300, 12345

    def _national(self, election_date, as_of, loader=None):
        if loader is None:
            return national_engine.generate_national_vote_shares(
                as_of=as_of, election_date=election_date, samples=self.N, seed=self.SEED)
        with patch.object(national_engine, "load_production_pp_residuals", loader):
            return national_engine.generate_national_vote_shares(
                as_of=as_of, election_date=election_date, samples=self.N, seed=self.SEED)

    def test_2026_forecast_is_the_unamended_forecast_bit_for_bit(self):
        routed = self._national("2026-09-13", "2026-08-24")
        legacy = self._national("2026-09-13", "2026-08-24", load_chronological_pp_residuals)
        self.assertEqual(routed.nat_shares_matrix.tobytes(), legacy.nat_shares_matrix.tobytes())
        self.assertEqual(routed.lambdas.tobytes(), legacy.lambdas.tobytes())
        self.assertEqual(routed.diagnostics, legacy.diagnostics)
        self.assertNotIn("election_noise_residual_consensus_rule", routed.diagnostics)

    def test_2030_forecast_uses_the_corrected_pool(self):
        routed = self._national("2030-09-08", "2026-11-01")
        corrected = self._national("2030-09-08", "2026-11-01", load_eve_pp_residuals)
        unamended = self._national("2030-09-08", "2026-11-01", load_chronological_pp_residuals)
        self.assertEqual(routed.nat_shares_matrix.tobytes(), corrected.nat_shares_matrix.tobytes())
        self.assertFalse(np.array_equal(routed.nat_shares_matrix, unamended.nat_shares_matrix))
        # Only ElectionNoise moves: the opinion state and dynamics are shared.
        self.assertEqual(routed.base_comp_matrix.tobytes(), unamended.base_comp_matrix.tobytes())
        self.assertEqual(routed.diagnostics["election_noise_residual_consensus_rule"], RULE_ID)


class EngineProvenance(unittest.TestCase):
    N, SEED = 40, 12345

    def test_a_2026_forecast_records_1_2_0_and_its_unchanged_config(self):
        r = simulate_election(as_of="2026-08-24", election_date="2026-09-13",
                              samples=self.N, seed=self.SEED)
        self.assertEqual(r.manifest["model_version"], "1.2.0-rc1")
        self.assertNotIn("residual_consensus_rule", r.manifest["model_config"])
        self.assertEqual(r.manifest["model_config_hash"], MODEL_CONFIG_HASH_2026)

    def test_a_2030_forecast_records_1_2_1_and_its_consensus_rule(self):
        r = simulate_election(as_of="2026-11-01", election_date="2030-09-08",
                              samples=self.N, seed=self.SEED)
        self.assertEqual(r.manifest["model_version"], "1.2.1-rc1")
        self.assertEqual(r.manifest["model_config"]["residual_consensus_rule"], RULE_ID)


class Fingerprint(unittest.TestCase):
    """The history cache key describes the residual consensus a target uses."""

    @classmethod
    def setUpClass(cls):
        data = REPO_ROOT / "data/processed"
        cls.inputs = {year: EffectiveInputs(data, election_date=date.fromisoformat(day), seed=12345)
                      for year, day in ((2026, "2026-09-13"), (2030, "2030-09-08"))}
        cls.polls = pd.read_csv(DEFAULT_POLLS_FILE)

    def _training_pollsters(self, year, election):
        record = dict(self.inputs[year].training)[election]
        return sorted((p["pollster"], p["interview_end"]) for p in record["polls"])

    def _expected(self, builder, election):
        return sorted((p.pollster, p.interview_end.isoformat())
                      for p in builder(election, self.polls).contributing_polls)

    def test_each_target_records_its_own_consensus_polls(self):
        for election in sorted(ALL_HISTORICAL_ELECTIONS):
            self.assertEqual(self._training_pollsters(2026, election),
                             self._expected(build_election_polling_consensus, election))
            self.assertEqual(self._training_pollsters(2030, election),
                             self._expected(build_eve_polling_consensus, election))

    def test_a_revision_predating_the_new_files_never_matches_a_current_fingerprint(self):
        """Legacy recovery records the two 1.2.1 files as absent, never as current."""
        data = REPO_ROOT / "data/processed"
        new_files = frozenset({"scripts/election_residuals/eve_consensus.py",
                               "scripts/election_layer_v2/production_residuals_pool.py"})
        older = EffectiveInputs(data, election_date=date(2030, 9, 8), seed=12345,
                                absent_model_files=new_files)
        day = date(2026, 10, 1)
        self.assertNotEqual(older.fingerprint(day), self.inputs[2030].fingerprint(day))
        with self.assertRaises(ValueError):
            EffectiveInputs(data, election_date=date(2030, 9, 8), seed=12345,
                            absent_model_files=frozenset({"scripts/not_a_model_file.py"}))

    def test_no_2030_training_poll_was_fielded_or_published_on_election_day(self):
        for election, record in self.inputs[2030].training:
            for poll in record["polls"]:
                self.assertLess(poll["interview_end"], election.isoformat())
                self.assertLess(poll["publication_date"], election.isoformat())


if __name__ == "__main__":
    unittest.main()
