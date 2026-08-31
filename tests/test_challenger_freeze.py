"""The challenger implementation freeze must be reconstructible from committed content.

Part 3D-R lesson, applied forward: a freeze that records working-tree-only content
can be verified on exactly one machine. Every entry here must therefore have a real
committed blob, and the working tree must equal HEAD.

These tests also assert that the freeze pins the scientific conventions the
competition depends on, and that it contains no target-election score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = (REPO_ROOT / "diagnostics/election_noise_v2/challengers"
          / "challenger_implementation_freeze.json")
EVALUATOR_REFREEZE_COMMIT = "a5b8c7a234acf60cac71ef1ab1439343fae88639"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_available() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class ChallengerFreezeReconstructible(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("challenger_implementation_freeze.json absent")
        cls.f = json.loads(FREEZE.read_text())
        cls.tables = {}
        for group, files in cls.f["implementation_hashes"].items():
            cls.tables[f"implementation:{group}"] = files
        cls.tables["frozen_dependency"] = cls.f["frozen_dependency_hashes"]
        cls.tables["test_file"] = cls.f["test_file_hashes"]
        cls.tables["import_closure"] = cls.f["import_closure_hashes"]

    def test_tables_are_not_empty(self):
        self.assertGreaterEqual(len(self.f["import_closure_hashes"]), 60)
        self.assertGreaterEqual(len(self.f["implementation_hashes"]), 6)

    def test_every_entry_has_a_committed_reference(self):
        missing = [f"{g}:{r}" for g, t in self.tables.items()
                   for r, v in t.items() if v["head_sha256"] is None]
        self.assertEqual(missing, [], f"entries with no committed blob: {missing}")

    def test_no_entry_has_an_uncommitted_local_edit(self):
        dirty = [f"{g}:{r}" for g, t in self.tables.items()
                 for r, v in t.items() if v["uncommitted_local_edit"]]
        self.assertEqual(dirty, [], f"entries recorded from a dirty tree: {dirty}")

    def test_working_tree_hash_equals_head_hash(self):
        bad = [f"{g}:{r}" for g, t in self.tables.items()
               for r, v in t.items() if v["working_tree_sha256"] != v["head_sha256"]]
        self.assertEqual(bad, [], f"working tree differs from HEAD: {bad}")

    def test_recorded_hashes_match_files_on_disk(self):
        for g, t in self.tables.items():
            for rel, v in t.items():
                p = REPO_ROOT / rel
                self.assertTrue(p.exists(), rel)
                self.assertEqual(_sha(p.read_bytes()), v["working_tree_sha256"], f"{g}:{rel}")

    def test_recorded_head_hashes_match_committed_blobs(self):
        if not _git_available():
            self.skipTest("git unavailable")
        for g, t in self.tables.items():
            for rel, v in t.items():
                blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT)
                self.assertEqual(_sha(blob), v["head_sha256"], f"{g}:{rel}")

    def test_freeze_verifier_reports_no_drift(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.challengers import freeze_challengers as fc
        res = fc.verify()
        self.assertEqual(res["drift"], [], f"challenger drift: {res['drift']}")
        self.assertTrue(res["challengers_unchanged"])

    def test_evaluator_references_are_current(self):
        a2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
        ev = self.f["evaluator"]
        self.assertEqual(ev["refreeze_commit"], EVALUATOR_REFREEZE_COMMIT)
        for key, rel in (("evaluator_freeze_sha256", "evaluator_freeze.json"),
                         ("evaluation_case_manifest_sha256", "evaluation_case_manifest.json"),
                         ("control_scores_summary_sha256", "control_scores_summary.json"),
                         ("exact_control_oracle_sha256", "exact_control_oracle.json")):
            self.assertEqual(_sha((a2 / rel).read_bytes()), ev[key], rel)

    def test_evaluator_freeze_still_verifies(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze
        res = freeze.verify()
        self.assertEqual(res["drift"], [])
        self.assertTrue(res["evaluator_unchanged"])


class ChallengerFreezePinsTheScience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("challenger_implementation_freeze.json absent")
        cls.f = json.loads(FREEZE.read_text())

    def test_contains_no_target_election_scores(self):
        self.assertTrue(self.f["contains_no_target_election_scores"])
        blob = json.dumps(self.f).lower()
        for banned in ("coalition_brier", "seat_energy_score", "es_9cat", "crps_8party",
                       "adoption_gate", "challenger_vs_control"):
            self.assertNotIn(banned, blob, f"freeze must not carry {banned}")

    def test_h_grid_is_frozen_exactly(self):
        self.assertEqual(self.f["challenger_a"]["h_grid"], [0.25, 0.50, 0.75, 1.00])
        self.assertTrue(self.f["challenger_a"]["h_zero_excluded"])
        self.assertEqual(self.f["challenger_a"]["free_parameters"], 1)

    def test_covariance_conventions_are_recorded(self):
        a = self.f["challenger_a"]
        self.assertIn("divisor K", a["covariance_convention"])
        self.assertIn("NO Bessel", a["covariance_convention"])
        self.assertIn("sqrt(1+h^2)", a["variance_correction"])
        b = self.f["challenger_b"]
        self.assertIn("once", b["bessel_correction"])
        self.assertEqual(b["free_parameters"], 0)
        self.assertEqual(b["distribution"], "Gaussian only")

    def test_tie_and_degenerate_rules_are_recorded(self):
        self.assertIn("SMALLEST h", self.f["nested_loocv"]["exact_tie_rule"])
        self.assertIn("delta := 1", self.f["challenger_b"]["d_sq_zero_rule"])
        self.assertEqual(self.f["nested_loocv"]["k_outer_minimum"], 3)
        self.assertEqual(self.f["nested_loocv"]["k_inner_minimum"], 2)
        self.assertIn("prohibited", self.f["nested_loocv"]["k_inner_one"])

    def test_seed_and_draw_policy_is_unchanged(self):
        mc = self.f["monte_carlo_policy"]
        self.assertEqual(mc["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(mc["draws_per_seed"], 20000)

    def test_reserved_rng_tokens_only(self):
        r = self.f["rng_contract"]
        self.assertEqual(sorted(r["reserved_tokens"]), sorted([
            "election_noise_v2_a_index", "election_noise_v2_a_kernel",
            "election_noise_v2_a_loeo", "election_noise_v2_b_normal"]))
        self.assertEqual(sorted(r["control_tokens_forbidden"]), ["residual_index", "sign_draw"])

    def test_transfer_hash_matches_the_production_file(self):
        p = REPO_ROOT / "scripts/election_layer_v2/transfer.py"
        self.assertEqual(_sha(p.read_bytes()),
                         self.f["downstream"]["apply_batch_simplex_transfer_sha256"])

    def test_bandwidths_are_pinned_and_on_the_grid(self):
        bw = self.f["nested_loocv"]["bandwidth_selection"]
        self.assertIsNotNone(bw, "h* must be pinned before scoring")
        for target, h in bw["h_star_by_target"].items():
            self.assertIn(h, [0.25, 0.50, 0.75, 1.00], target)
        self.assertEqual(sorted(bw["h_star_by_target"]), ["2014", "2018", "2022"])
        art = (REPO_ROOT / "diagnostics/election_noise_v2/challengers"
               / "bandwidth_selection.json")
        self.assertEqual(_sha(art.read_bytes()), bw["artifact_sha256"])

    def test_all_targeted_tests_passed_at_freeze_time(self):
        tr = self.f["targeted_test_results"]
        self.assertTrue(tr["all_passed"])
        self.assertEqual(tr["total_tests"], 71)
        self.assertTrue(tr["no_target_election_scores_computed"])


if __name__ == "__main__":
    unittest.main()
