"""Tests for the Amendment-2 CONTROL baseline and the frozen evaluator.

Guard the properties Part-4 challenger work must not disturb: the case set, the law
dispatch, the geography restriction, the certified truth, the exact finite-support
CONTROL oracle, the Monte Carlo convergence to it, leakage-freedom of the training
pools, and byte-preservation of the superseded full-pipeline diagnostics.

Heavy Monte Carlo is avoided: the committed baseline artifacts are read rather than
recomputed, and any live computation uses the exact oracle or a small draw count.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
PART3 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline"

_HAS_BASELINE = (A2 / "control_scores_summary.json").exists()
_HAS_GEO = (
    REPO_ROOT / "diagnostics/election_noise_v2/historical_seat_extension/processed"
    / "research_geography/constituency_party_votes_2014_2022.csv"
).exists()

from scripts.mandates.law import MandateLaw, mandate_law_for_election_year
from scripts.simulator.config import PARLIAMENTARY_PARTIES_8

from tests._freeze_drift import unexpected_drift

CERTIFIED = {
    2014: {"M": 84, "L": 19, "C": 22, "KD": 16, "S": 113, "V": 21, "MP": 25, "SD": 49},
    2018: {"M": 70, "L": 20, "C": 31, "KD": 22, "S": 100, "V": 28, "MP": 16, "SD": 62},
    2022: {"M": 68, "L": 16, "C": 24, "KD": 19, "S": 107, "V": 24, "MP": 18, "SD": 73},
}
EXPECTED_K = {2014: 3, 2018: 4, 2022: 5}


@unittest.skipUnless(_HAS_GEO, "Part-2B research geography not present")
class ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.manifest import (
            build_manifest,
            validate_manifest,
        )

        cls.m = build_manifest()
        cls.validate = staticmethod(validate_manifest)

    def test_manifest_validates(self) -> None:
        self.assertEqual(self.validate(self.m), [])

    def test_final_counts(self) -> None:
        self.assertEqual(self.m["counts"]["N_T1"], 3)
        self.assertEqual(self.m["counts"]["N_seat"], 3)

    def test_tier3_iso_targets_are_exactly_2014_2018_2022(self) -> None:
        self.assertEqual(self.m["counts"]["tier3_iso_elections"], [2014, 2018, 2022])
        self.assertEqual(self.m["counts"]["tier1_elections"], [2014, 2018, 2022])
        self.assertEqual(self.m["counts"]["tier3_iso_cases"], 3)

    def test_gate_tiers_are_tier1_and_tier3_iso(self) -> None:
        self.assertEqual(self.m["gate_tiers"], ["tier1", "tier3_iso"])

    def test_2014_uses_pre_2018_and_others_post_2018(self) -> None:
        laws = {c["target_year"]: (c["mandate_law"], c["first_divisor"])
                for c in self.m["cases"]["tier3_iso"]}
        self.assertEqual(laws[2014], ("PRE_2018", "7/5"))
        self.assertEqual(laws[2018], ("POST_2018", "6/5"))
        self.assertEqual(laws[2022], ("POST_2018", "6/5"))

    def test_geography_is_chronological_and_oracle_is_forbidden(self) -> None:
        for c in self.m["cases"]["tier3_iso"]:
            self.assertEqual(c["geography_mode"], "chronological")
            self.assertIn("oracle", c["forbidden_geography_modes"])
            self.assertLess(c["geography_baseline_year"], c["target_year"])

    def test_certified_truth_vectors_are_correct(self) -> None:
        for c in self.m["cases"]["tier3_iso"]:
            self.assertEqual(c["truth_seats"], CERTIFIED[c["target_year"]])
            self.assertEqual(sum(c["truth_seats"].values()), 349)
            self.assertEqual(sum(c["fixed_seats"].values()), 310)

    def test_no_future_residual_enters_a_training_pool(self) -> None:
        for c in self.m["cases"]["tier1"] + self.m["cases"]["tier3_iso"]:
            self.assertEqual(c["k_outer"], EXPECTED_K[c["target_year"]])
            for y in c["training_residual_years"]:
                self.assertLess(y, c["target_year"], f"leak into {c['target_year']}")

    def test_2010_is_excluded(self) -> None:
        self.assertNotIn(2010, self.m["counts"]["tier1_elections"])
        self.assertNotIn(2010, self.m["counts"]["tier3_iso_elections"])

    def test_seed_and_draw_policy_frozen(self) -> None:
        self.assertEqual(self.m["monte_carlo"]["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(self.m["monte_carlo"]["draws_per_seed"], 20000)
        for c in self.m["cases"]["tier1"] + self.m["cases"]["tier3_iso"]:
            self.assertEqual(c["seeds"], [12345, 24680, 98765, 54321, 13579])
            self.assertEqual(c["draws_per_seed"], 20000)


class LawDispatchTest(unittest.TestCase):
    def test_statutory_law_per_target(self) -> None:
        self.assertIs(mandate_law_for_election_year(2014).law, MandateLaw.PRE_2018)
        self.assertEqual(mandate_law_for_election_year(2014).first_divisor, Fraction(7, 5))
        for y in (2018, 2022):
            self.assertIs(mandate_law_for_election_year(y).law, MandateLaw.POST_2018)
            self.assertEqual(mandate_law_for_election_year(y).first_divisor, Fraction(6, 5))


@unittest.skipUnless(_HAS_GEO, "Part-2B research geography not present")
class GeographyModeGuardTest(unittest.TestCase):
    def test_oracle_mode_is_rejected(self) -> None:
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
            assert_geography_mode,
        )

        with self.assertRaises(RuntimeError) as ctx:
            assert_geography_mode("oracle")
        self.assertIn("GEOGRAPHY MODE VIOLATION", str(ctx.exception))

    def test_chronological_mode_is_accepted(self) -> None:
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
            assert_geography_mode,
        )

        assert_geography_mode("chronological")

    def test_any_other_mode_is_rejected(self) -> None:
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.isolated import (
            assert_geography_mode,
        )

        with self.assertRaises(RuntimeError):
            assert_geography_mode("production")


@unittest.skipUnless(_HAS_BASELINE, "Amendment-2 baseline not present")
class ExactOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.o = json.loads((A2 / "exact_control_oracle.json").read_text())["by_election"]

    def test_support_size_is_k_3_4_5(self) -> None:
        for y, k in EXPECTED_K.items():
            self.assertEqual(self.o[str(y)]["k"], k)
            self.assertEqual(len(self.o[str(y)]["exact_vote_support"]), k)

    def test_probability_mass_sums_to_one(self) -> None:
        for y, k in EXPECTED_K.items():
            orc = self.o[str(y)]
            self.assertTrue(orc["probability_mass_sums_to_one"])
            total = sum(a["probability"] for a in orc["exact_vote_support"])
            self.assertAlmostEqual(total, 1.0, places=12)
            self.assertAlmostEqual(orc["atom_probability"], 1.0 / k, places=15)

    def test_coalition_probabilities_are_multiples_of_one_over_k(self) -> None:
        for y, k in EXPECTED_K.items():
            orc = self.o[str(y)]
            self.assertTrue(orc["all_coalition_probabilities_are_multiples_of_1_over_k"])
            for m in range(1, 255):
                p = orc["per_mask"][str(m)]["exact_probability"]
                self.assertAlmostEqual(p * k, round(p * k), places=12, msg=f"{y} mask {m}")
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_every_atom_allocates_349_seats(self) -> None:
        for y in EXPECTED_K:
            for a in self.o[str(y)]["exact_vote_support"]:
                self.assertEqual(a["seat_total"], 349)
                self.assertEqual(sum(a["seats"].values()), 349)

    def test_lambda_is_identically_one_so_the_atom_count_survives(self) -> None:
        for y in EXPECTED_K:
            self.assertTrue(self.o[str(y)]["lambda_identically_one"])

    def test_truth_matches_certified(self) -> None:
        for y in EXPECTED_K:
            self.assertEqual(self.o[str(y)]["truth_seats"], CERTIFIED[y])
            self.assertTrue(self.o[str(y)]["truth_sums_to_349"])

    def test_memoisation_was_exact(self) -> None:
        memo = json.loads((A2 / "exact_control_oracle.json").read_text())["memoisation_exactness_check"]
        for y, k in EXPECTED_K.items():
            self.assertTrue(memo[str(y)]["memoised_equals_per_draw"])
            self.assertEqual(memo[str(y)]["distinct_vote_rows"], k)
            self.assertEqual(memo[str(y)]["distinct_seat_rows"], k)


@unittest.skipUnless(_HAS_BASELINE, "Amendment-2 baseline not present")
class MonteCarloConvergesToExactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cmp = json.loads((A2 / "monte_carlo_vs_exact.json").read_text())

    def test_no_systematic_discrepancy(self) -> None:
        self.assertTrue(self.cmp["all_consistent_with_sampling_error"])
        self.assertEqual(self.cmp["blockers"], [])

    def test_coalition_probability_error_within_five_sigma(self) -> None:
        for y, v in self.cmp["by_election"].items():
            self.assertLessEqual(
                v["worst_max_abs_coalition_probability_error"], v["five_sigma_tolerance"], y
            )
            self.assertTrue(v["consistent_with_sampling_error"], y)

    def test_five_seed_mean_is_close_to_exact(self) -> None:
        for y, v in self.cmp["by_election"].items():
            f = v["five_seed"]
            self.assertLess(abs(f["coalition_brier_relative_error_pct"]), 1.0, y)
            self.assertLess(abs(f["seat_es_relative_error_pct"]), 1.0, y)

    def test_definitional_differences_are_documented(self) -> None:
        for y, v in self.cmp["by_election"].items():
            self.assertGreaterEqual(len(v["definitional_differences"]), 2, y)


@unittest.skipUnless(_HAS_BASELINE, "Amendment-2 baseline not present")
class Tier1UnchangedTest(unittest.TestCase):
    def test_tier1_is_bit_identical_to_part3(self) -> None:
        s = json.loads((A2 / "control_scores_summary.json").read_text())
        chk = s["tier1_unchanged_vs_part3"]
        self.assertTrue(chk["tier1_bit_identical_to_part3"])
        self.assertEqual(chk["mismatches"], [])
        self.assertEqual(chk["cases_compared"], 15)
        self.assertGreater(chk["metric_values_compared"], 100)


@unittest.skipUnless(_HAS_BASELINE, "Amendment-2 baseline not present")
class PreservedDiagnosticsTest(unittest.TestCase):
    """The superseded full-pipeline outputs must remain byte-for-byte unchanged."""

    def test_part3_outputs_match_their_recorded_hashes(self) -> None:
        import hashlib

        frozen = json.loads((A2 / "evaluator_freeze.json").read_text())
        recorded = frozen["preserved_part3_full_pipeline_diagnostics"]["hashes"]
        self.assertGreaterEqual(len(recorded), 8)
        for rel, expected in recorded.items():
            path = PART3 / rel
            self.assertTrue(path.exists(), rel)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"preserved diagnostic changed: {rel}")

    def test_part3_full_pipeline_rows_still_present(self) -> None:
        tiers = set()
        with open(PART3 / "control_scores_by_case_seed.csv", newline="") as f:
            for r in csv.DictReader(f):
                tiers.add(r["tier"])
        self.assertIn("tier2", tiers)
        self.assertIn("tier3", tiers)

    def test_manifest_labels_them_as_diagnostics_only(self) -> None:
        m = json.loads((A2 / "evaluation_case_manifest.json").read_text())
        pres = m["preserved_full_pipeline_diagnostics"]
        self.assertIn("DIAGNOSTICS ONLY", pres["role"])
        self.assertTrue(pres["never_recomputed_or_deleted"])
        self.assertNotIn("tier2", m["gate_tiers"])
        self.assertNotIn("tier3", m["gate_tiers"])


@unittest.skipUnless(_HAS_BASELINE, "Amendment-2 baseline not present")
class EvaluatorFreezeTest(unittest.TestCase):
    #: Production files that later, deliberate work changed after this evaluator freeze
    #: was taken: the Part-6B default flip to ElectionNoise B, the Part-7B1 metadata fix
    #: and main's purely additive party-chart parsers. The freeze artifact is never
    #: rewritten - it certifies the evaluator at its own referenced commit, where it
    #: still verifies cleanly - so this test bounds the drift rather than demanding none.
    KNOWN_POST_FREEZE_CHANGES = {
        "scripts/pollofpolls/normalize.py",
        "scripts/pollofpolls/validate.py",
        "scripts/simulator/config.py",
        "scripts/simulator/engine.py",
        "scripts/simulator/reproducibility.py",
        "scripts/vote_share_calibration/national_engine.py",
    }

    def test_freeze_self_verifies(self) -> None:
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2.freeze import verify

        res = verify()
        unexpected = unexpected_drift(res, self.KNOWN_POST_FREEZE_CHANGES)
        self.assertEqual(
            unexpected, set(),
            f"evaluator drift outside the known post-freeze set: "
            f"{sorted(unexpected)}")
        self.assertGreater(res["checks"], 30)

    def test_freeze_records_the_amendment2_hashes(self) -> None:
        frozen = json.loads((A2 / "evaluator_freeze.json").read_text())
        pre = frozen["preregistration"]
        self.assertEqual(pre["status"], "FROZEN - AMENDMENT 2")
        self.assertEqual(
            pre["body_sha256"],
            "5a9a6dc8ef6f26ce3ce152155af0ed288fb8d2d97c81a2606e513cf20e1b058b",
        )
        self.assertFalse(pre["edited_by_this_task"])

    def test_freeze_records_the_paired_randomness_contract(self) -> None:
        frozen = json.loads((A2 / "evaluator_freeze.json").read_text())
        c = frozen["paired_randomness_contract"]
        self.assertIn("residual_index", c["control_streams"])
        for t in ("election_noise_v2_a_index", "election_noise_v2_a_kernel",
                  "election_noise_v2_a_loeo", "election_noise_v2_b_normal"):
            self.assertIn(t, c["challenger_reserved_streams"])
        self.assertGreaterEqual(len(c["prohibited_for_challenger_implementations"]), 4)

    def test_freeze_records_seed_and_draw_policy(self) -> None:
        frozen = json.loads((A2 / "evaluator_freeze.json").read_text())
        mc = frozen["monte_carlo_policy"]
        self.assertEqual(mc["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(mc["draws_per_seed"], 20000)


if __name__ == "__main__":
    unittest.main()
