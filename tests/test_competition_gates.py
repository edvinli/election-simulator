"""Tests for the competition and gate machinery.

Gate logic is exercised on synthetic numbers so the rules are verified independently
of what the real scores turned out to be. Configuration tests assert the frozen
design (case set, seeds, N, bandwidths) is what the run actually used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP = REPO_ROOT / "diagnostics/election_noise_v2/competition"

from diagnostics.election_noise_v2.competition import gates as G
from diagnostics.election_noise_v2.competition.runner import CONTROL, FROZEN_H, MODELS


class Conventions(unittest.TestCase):
    def test_relative_improvement_sign_convention(self):
        """Lower is better everywhere, so a lower challenger value is positive."""
        self.assertAlmostEqual(G.rel_improvement(100.0, 98.0), 2.0)
        self.assertAlmostEqual(G.rel_improvement(100.0, 102.0), -2.0)
        self.assertAlmostEqual(G.rel_improvement(100.0, 100.0), 0.0)
        self.assertAlmostEqual(G.rel_degradation(100.0, 101.0), 1.0)

    def test_zero_control_is_refused(self):
        with self.assertRaises(ValueError):
            G.rel_improvement(0.0, 1.0)

    def test_frozen_thresholds(self):
        self.assertEqual(G.IMPROVE_PCT, 2.0)
        self.assertEqual(G.NONINFERIOR_PCT, 1.0)
        self.assertEqual(G.COVERAGE_PP, 3.0)
        self.assertEqual(G.LOO_IMPROVE_PCT, 1.0)
        self.assertEqual(G.BRIER_LOO_DEGRADE_PCT, 1.0)
        self.assertEqual(G.NOMINAL, {"50": 0.50, "80": 0.80, "90": 0.90})


class G1G2Improvement(unittest.TestCase):
    def test_exactly_two_percent_passes(self):
        self.assertEqual(G.g1_tier1_improvement(100.0, 98.0)["result"], "PASS")
        self.assertEqual(G.g2_coalition_improvement(0.100, 0.098)["result"], "PASS")

    def test_just_under_two_percent_fails(self):
        self.assertEqual(G.g1_tier1_improvement(100.0, 98.001)["result"], "FAIL")

    def test_degradation_fails(self):
        self.assertEqual(G.g1_tier1_improvement(100.0, 101.0)["result"], "FAIL")
        self.assertEqual(G.g2_coalition_improvement(0.100, 0.101)["result"], "FAIL")


class G3G4NonInferiority(unittest.TestCase):
    def _c(self, **kw):
        base = {"es_9cat": 100.0, "es_8party": 100.0, "crps_8party_mean": 100.0,
                "coverage_50": 0.50, "coverage_80": 0.80, "coverage_90": 0.90}
        base.update(kw)
        return base

    def test_one_percent_degradation_is_the_boundary(self):
        rows = G.g3_noninferiority(self._c(), self._c(es_9cat=101.0))
        self.assertEqual([r["result"] for r in rows if r["metric"] == "tier1 es_9cat"], ["PASS"])
        rows = G.g3_noninferiority(self._c(), self._c(es_9cat=101.001))
        self.assertEqual([r["result"] for r in rows if r["metric"] == "tier1 es_9cat"], ["FAIL"])

    def test_every_marginal_metric_is_checked_independently(self):
        rows = G.g3_noninferiority(self._c(), self._c(crps_8party_mean=105.0))
        self.assertIn("FAIL", [r["result"] for r in rows])

    def test_coverage_deviation_from_nominal_not_raw_coverage(self):
        """A challenger closer to nominal than CONTROL must pass even if coverage moved."""
        ctrl = self._c(coverage_50=0.30)          # 20 pp from nominal
        chal = self._c(coverage_50=0.45)          # 5 pp from nominal - better
        rows = [r for r in G.g3_noninferiority(ctrl, chal) if "coverage 50" in r["metric"]]
        self.assertEqual(rows[0]["result"], "PASS")

    def test_coverage_worsening_by_more_than_three_pp_fails(self):
        ctrl = self._c(coverage_80=0.80)          # 0 pp deviation
        chal = self._c(coverage_80=0.759)         # 4.1 pp deviation
        rows = [r for r in G.g3_noninferiority(ctrl, chal) if "coverage 80" in r["metric"]]
        self.assertEqual(rows[0]["result"], "FAIL")

    def test_coverage_worsening_by_exactly_three_pp_passes(self):
        ctrl = self._c(coverage_90=0.90)
        chal = self._c(coverage_90=0.87)
        rows = [r for r in G.g3_noninferiority(ctrl, chal) if "coverage 90" in r["metric"]]
        self.assertEqual(rows[0]["result"], "PASS")

    def test_each_nominal_level_reported_separately(self):
        rows = [r for r in G.g3_noninferiority(self._c(), self._c()) if "coverage" in r["metric"]]
        self.assertEqual(len(rows), 3)

    def test_g4_seat_non_inferiority(self):
        self.assertEqual(G.g4_seat_noninferiority(10.0, 10.1)["result"], "PASS")
        self.assertEqual(G.g4_seat_noninferiority(10.0, 10.2)["result"], "FAIL")


class G5Tier1Robustness(unittest.TestCase):
    C = {2014: 100.0, 2018: 100.0, 2022: 100.0}

    def test_all_three_loo_must_be_positive(self):
        # 2022 much worse: drop-2014 and drop-2018 aggregates still contain 2022
        chal = {2014: 90.0, 2018: 90.0, 2022: 130.0}
        rows, d = G.g5_tier1_robustness(self.C, chal)
        b1 = [r for r in rows if r["gate"] == "G5-B1"][0]
        self.assertEqual(b1["result"], "FAIL")

    def test_all_positive_and_two_above_one_percent_passes(self):
        chal = {2014: 97.0, 2018: 97.0, 2022: 99.5}
        rows, d = G.g5_tier1_robustness(self.C, chal)
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-B1"], ["PASS"])
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-B2"], ["PASS"])

    def test_only_one_loo_above_one_percent_fails_b2(self):
        # improvements 0.1% / 1.0% / 1.0% -> LOO pair means give exactly one >= 1%
        chal = {2014: 99.9, 2018: 99.0, 2022: 99.0}
        rows, d = G.g5_tier1_robustness(self.C, chal)
        b2 = [r for r in rows if r["gate"] == "G5-B2"][0]
        self.assertEqual(b2["challenger"], 1)
        self.assertEqual(b2["result"], "FAIL")

    def test_individual_election_rule_needs_two_of_three(self):
        rows, _ = G.g5_tier1_robustness(self.C, {2014: 99.0, 2018: 99.0, 2022: 101.0})
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-C"], ["PASS"])
        rows, _ = G.g5_tier1_robustness(self.C, {2014: 99.0, 2018: 101.0, 2022: 101.0})
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-C"], ["FAIL"])

    def test_loo_aggregate_is_the_unweighted_mean_of_remaining_elections(self):
        chal = {2014: 90.0, 2018: 94.0, 2022: 98.0}
        _, d = G.g5_tier1_robustness(self.C, chal)
        self.assertAlmostEqual(d["leave_one_out"]["2014"]["challenger"], (94.0 + 98.0) / 2)
        self.assertAlmostEqual(d["leave_one_out"]["2014"]["control"], 100.0)


class G5CoalitionRobustness(unittest.TestCase):
    C = {2014: 0.040, 2018: 0.020, 2022: 0.020}

    def test_needs_ceil_n_over_two_elections(self):
        self.assertEqual(math.ceil(3 / 2), 2)
        rows, _ = G.g5_coalition_robustness(self.C, {2014: 0.030, 2018: 0.010, 2022: 0.030})
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-Brier-elections"], ["PASS"])
        rows, _ = G.g5_coalition_robustness(self.C, {2014: 0.030, 2018: 0.030, 2022: 0.030})
        self.assertEqual([r["result"] for r in rows if r["gate"] == "G5-Brier-elections"], ["FAIL"])

    def test_loo_degradation_beyond_one_percent_fails(self):
        chal = {2014: 0.010, 2018: 0.0205, 2022: 0.0205}   # drop-2014 -> ~-2.5%
        rows, d = G.g5_coalition_robustness(self.C, chal)
        loo = [r for r in rows if r["gate"] == "G5-Brier-LOO"][0]
        self.assertEqual(loo["result"], "FAIL")

    def test_small_loo_degradation_within_tolerance_passes(self):
        chal = {2014: 0.010, 2018: 0.0201, 2022: 0.0201}   # drop-2014 -> ~-0.5%
        rows, _ = G.g5_coalition_robustness(self.C, chal)
        loo = [r for r in rows if r["gate"] == "G5-Brier-LOO"][0]
        self.assertEqual(loo["result"], "PASS")


class DecisionRule(unittest.TestCase):
    def test_neither_passes_retains_control(self):
        d = G.decide({"A": False, "B": False}, {"A": 1.0, "B": 1.0})
        self.assertEqual(d["decision"], "RETAIN_CONTROL")

    def test_exactly_one_passes(self):
        self.assertEqual(G.decide({"A": True, "B": False}, {"A": 1.0, "B": 1.0})["decision"],
                         "ADOPT_A")
        self.assertEqual(G.decide({"A": False, "B": True}, {"A": 1.0, "B": 1.0})["decision"],
                         "ADOPT_B")

    def test_both_pass_selects_lower_tier1_es(self):
        self.assertEqual(G.decide({"A": True, "B": True}, {"A": 3.00, "B": 3.20})["decision"],
                         "ADOPT_A")
        self.assertEqual(G.decide({"A": True, "B": True}, {"A": 3.20, "B": 3.00})["decision"],
                         "ADOPT_B")

    def test_tie_rule_prefers_fewer_parameters(self):
        d = G.decide({"A": True, "B": True}, {"A": 3.000, "B": 3.002})
        self.assertEqual(d["decision"], "ADOPT_B")
        self.assertIn("fewer free parameters", d["rule"])

    def test_coalition_brier_alone_never_decides(self):
        """The decision function only ever receives Tier-1 ES, never Brier."""
        import inspect
        self.assertEqual(list(inspect.signature(G.decide).parameters), ["passes", "tier1_es"])


class FrozenConfiguration(unittest.TestCase):
    def test_bandwidths_pinned_to_075(self):
        self.assertEqual(FROZEN_H, {2014: 0.75, 2018: 0.75, 2022: 0.75})

    def test_three_models(self):
        self.assertEqual(len(MODELS), 3)
        self.assertIn(CONTROL, MODELS)


class RunConfiguration(unittest.TestCase):
    """Assertions against the manifest the run actually pinned."""

    @classmethod
    def setUpClass(cls):
        p = COMP / "competition_manifest.json"
        if not p.exists():
            raise unittest.SkipTest("competition not yet run")
        cls.m = json.loads(p.read_text())

    def test_case_set_is_exactly_the_frozen_targets(self):
        self.assertEqual(self.m["targets"], [2014, 2018, 2022])
        self.assertEqual(sorted(c["target_year"] for c in self.m["tier1_cases"]),
                         [2014, 2018, 2022])
        self.assertEqual(sorted(c["target_year"] for c in self.m["tier3_iso_cases"]),
                         [2014, 2018, 2022])
        self.assertEqual(self.m["N_T1"], 3)
        self.assertEqual(self.m["N_seat"], 3)

    def test_seeds_and_n_exact(self):
        self.assertEqual(self.m["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(self.m["draws_per_seed"], 20000)

    def test_bandwidths_recorded_as_075(self):
        self.assertEqual(self.m["challenger_a_bandwidths"],
                         {"2014": 0.75, "2018": 0.75, "2022": 0.75})
        self.assertEqual(self.m["challenger_b_hyperparameters"], 0)

    def test_law_dispatch_and_geography(self):
        self.assertEqual(self.m["mandate_law"],
                         {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"})
        self.assertEqual(self.m["geography"]["mode"], "chronological")
        self.assertEqual(self.m["geography"]["oracle"], "forbidden")
        for c in self.m["tier3_iso_cases"]:
            self.assertEqual(c["geography_mode"], "chronological")

    def test_challenger_freeze_hash_matches_the_authoritative_value(self):
        self.assertEqual(self.m["freeze_hashes"]["challenger_implementation_freeze"],
                         self.m["freeze_hashes"]["challenger_implementation_freeze_expected"])

    def test_2026_is_not_an_input(self):
        """No 2026 election enters the case set, targets or training pools."""
        self.assertIn("not an adoption input", self.m["forecast_2026"])
        self.assertNotIn(2026, self.m["targets"])
        for group in ("tier1_cases", "tier3_iso_cases"):
            for c in self.m[group]:
                self.assertNotEqual(c["target_year"], 2026)
                self.assertTrue(all(y < c["target_year"] for y in c["training_residual_years"]))
                self.assertNotIn(2026, c["training_residual_years"])


class AllModelsOnIdenticalCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import csv
        p = COMP / "scores_by_model_case_seed.csv"
        if not p.exists():
            raise unittest.SkipTest("competition not yet run")
        cls.rows = list(csv.DictReader(p.open(newline="")))

    def test_every_model_covers_every_case_and_seed(self):
        keys = {}
        for r in self.rows:
            keys.setdefault(r["model"], set()).add(
                (r["tier"], r["target_year"], r["seed"]))
        self.assertEqual(len(keys), 3)
        sets = list(keys.values())
        for s in sets[1:]:
            self.assertEqual(s, sets[0], "models must run on identical cases")
        self.assertEqual(len(sets[0]), 2 * 3 * 5)

    def test_run_count(self):
        self.assertEqual(len(self.rows), 3 * 2 * 3 * 5)

    def test_five_seeds_per_cell(self):
        cells = {}
        for r in self.rows:
            cells.setdefault((r["model"], r["tier"], r["target_year"]), set()).add(r["seed"])
        for k, v in cells.items():
            self.assertEqual(len(v), 5, k)

    def test_challenger_a_used_the_pinned_bandwidth(self):
        for r in self.rows:
            if r["model"].startswith("CHALLENGER_A"):
                self.assertEqual(float(r["h"]), 0.75)
            else:
                self.assertEqual(r["h"], "")


if __name__ == "__main__":
    unittest.main()


class DecisionRecord(unittest.TestCase):
    """decision.json must follow mechanically from the gate table and the audit."""

    @classmethod
    def setUpClass(cls):
        p = COMP / "decision.json"
        if not p.exists():
            raise unittest.SkipTest("competition not yet decided")
        cls.d = json.loads(p.read_text())
        cls.g = json.loads((COMP / "gate_table.json").read_text())
        cls.a = json.loads((COMP / "score_audit.json").read_text())

    def test_decision_matches_the_frozen_rule_applied_to_the_gate_table(self):
        passes = self.g["all_gates_pass"]
        t1 = {m: self.d["headline_metrics"][m]["tier1_es_9cat"] for m in ("A", "B")}
        self.assertEqual(self.d["selected_model"], G.decide(passes, t1)["decision"])

    def test_pass_flags_agree_with_the_individual_gate_rows(self):
        for m in ("A", "B"):
            rows = [r for r in self.g["rows"] if r["model"] == m]
            self.assertTrue(rows)
            self.assertEqual(self.g["all_gates_pass"][m],
                             all(r["result"] == "PASS" for r in rows))

    def test_no_discretionary_override(self):
        self.assertFalse(self.d["discretionary_override"])

    def test_2026_was_not_an_adoption_input(self):
        self.assertEqual(self.d["forecast_2026_statement"],
                         "2026 forecast was not an adoption input.")
        self.assertFalse(self.d["forecast_2026_run"])

    def test_records_every_authoritative_hash(self):
        self.assertEqual(self.d["authoritative_commits"]["challenger_freeze"],
                         "1450e6f301a98d5d6e4af1357113435534b0e7a9")
        self.assertEqual(self.d["freeze_hashes"]["challenger_implementation_freeze"],
                         "2454ac15309361443656fe1d00abd5cb655d5a8efc8ddaded9e8c7164d8c1c22")
        self.assertEqual(self.d["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(self.d["draws_per_seed"], 20000)
        self.assertEqual(self.d["challenger_a_bandwidths"],
                         {"2014": 0.75, "2018": 0.75, "2022": 0.75})

    def test_audit_declared_the_run_valid(self):
        self.assertTrue(self.a["run_valid"])
        self.assertEqual(self.a["problems"], [])
        self.assertTrue(self.a["checks"]["all_reruns_bit_identical"])
        self.assertTrue(self.a["checks"]["control_reproduces_certified_baseline"])
        self.assertEqual(self.a["checks"]["evaluator_freeze"]["drift"], [])
        self.assertEqual(self.a["checks"]["challenger_freeze"]["drift"], [])
