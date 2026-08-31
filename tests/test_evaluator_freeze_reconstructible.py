"""Part 3D-R: the evaluator freeze must be reconstructible from committed content.

The original Part-3D freeze (aea30ba) recorded, for two distinct reasons, a
reference state that no clean checkout could reproduce:

1. ``scripts/pollofpolls/normalize.py`` was captured at its *working-tree* hash
   while an unrelated additive edit was uncommitted, so ``verify()`` demanded a
   blob present in no commit.
2. Seven ``control_baseline_amendment2`` modules - including ``isolated.py``,
   ``exact_oracle.py``, ``manifest.py``, ``run_control.py`` and ``freeze.py``
   itself - recorded ``head_sha256: null``, because the freeze was built from a
   commit at which that directory was still untracked. The evaluator's own core
   implementation therefore had no committed reference at all.

These tests fail if either defect returns. They deliberately assert the *strict*
property - working tree equals HEAD for every closure module - rather than
accepting a looser verifier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
A2 = REPO_ROOT / "diagnostics/election_noise_v2/control_baseline_amendment2"
FREEZE = A2 / "evaluator_freeze.json"
COMPARISON = A2 / "clean_reproduction_comparison.json"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_available() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class EvaluatorFreezeReconstructible(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("evaluator_freeze.json absent")
        cls.frozen = json.loads(FREEZE.read_text())
        cls.closure = cls.frozen["evaluator_import_closure_hashes"]

    def test_closure_is_not_empty(self):
        """Guard against a vacuous pass if the closure walk ever returns nothing."""
        self.assertGreaterEqual(len(self.closure), 60)

    def test_no_module_has_an_uncommitted_local_edit(self):
        dirty = {k: v for k, v in self.closure.items() if v["uncommitted_local_edit"]}
        self.assertEqual(dirty, {}, f"freeze depends on uncommitted working-tree content: {sorted(dirty)}")

    def test_every_module_has_a_committed_reference(self):
        """Defect 2: head_sha256 must never be null."""
        missing = sorted(k for k, v in self.closure.items() if v["head_sha256"] is None)
        self.assertEqual(missing, [], f"closure modules with no committed reference: {missing}")

    def test_working_tree_hash_equals_head_hash(self):
        """Defect 1: the recorded reference must be the committed blob."""
        mismatched = sorted(k for k, v in self.closure.items()
                            if v["working_tree_sha256"] != v["head_sha256"])
        self.assertEqual(mismatched, [], f"working tree differs from HEAD for: {mismatched}")

    def test_recorded_hashes_match_the_files_on_disk(self):
        for rel, v in self.closure.items():
            p = REPO_ROOT / rel
            self.assertTrue(p.exists(), f"{rel} missing")
            self.assertEqual(_sha256(p.read_bytes()), v["working_tree_sha256"], rel)

    def test_recorded_head_hashes_match_committed_blobs(self):
        """Reconstructible: every reference is retrievable from the commit itself."""
        if not _git_available():
            self.skipTest("git unavailable")
        for rel, v in self.closure.items():
            blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT)
            self.assertEqual(_sha256(blob), v["head_sha256"], rel)

    def test_normalize_py_is_the_committed_variant(self):
        """The specific regression: the dirty party-chart edit must not be absorbed."""
        rec = self.closure["scripts/pollofpolls/normalize.py"]
        self.assertEqual(rec["working_tree_sha256"], rec["head_sha256"])
        self.assertNotEqual(
            rec["working_tree_sha256"],
            "c6b0480d89cde0b892e1769a394e13d29bdde04d1a8abcce178cbe44cd4da09a",
            "the unrelated uncommitted party-chart edit was absorbed into the evaluator")

    def test_freeze_verify_passes_with_no_drift(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.control_baseline_amendment2.harness2 import freeze
        res = freeze.verify()
        self.assertEqual(res["drift"], [], f"evaluator drift: {res['drift']}")
        self.assertTrue(res["evaluator_unchanged"])

    def test_scientific_freeze_content_is_unchanged_by_the_remediation(self):
        """The repair must not have moved any evaluation rule, case, seed or truth."""
        f = self.frozen
        self.assertEqual(f["monte_carlo_policy"]["seeds"], [12345, 24680, 98765, 54321, 13579])
        self.assertEqual(f["monte_carlo_policy"]["draws_per_seed"], 20000)
        self.assertEqual(f["case_set"]["tier1_elections"], [2014, 2018, 2022])
        self.assertEqual(f["case_set"]["tier3_iso_elections"], [2014, 2018, 2022])
        self.assertEqual(f["case_set"]["N_T1"], 3)
        self.assertEqual(f["case_set"]["N_seat"], 3)
        self.assertEqual(f["case_set"]["geography_mode"], "chronological")
        self.assertEqual(f["case_set"]["mandate_law"],
                         {"2014": "PRE_2018", "2018": "POST_2018", "2022": "POST_2018"})


class CleanTreeReproducesCertifiedControl(unittest.TestCase):
    """The clean evaluator was shown to reproduce the certified CONTROL outputs.

    The heavy 5-seed x 20 000-draw run is not repeated here. Instead the recorded
    comparison is checked for a pass *and* pinned to the baseline files it was
    computed against, so it cannot silently go stale if an artifact changes.
    """

    @classmethod
    def setUpClass(cls):
        if not COMPARISON.exists():
            raise unittest.SkipTest("clean_reproduction_comparison.json absent")
        cls.cmp = json.loads(COMPARISON.read_text())

    def test_reproduction_matched_on_every_artifact(self):
        self.assertTrue(self.cmp["all_scientific_content_identical"])
        self.assertEqual(self.cmp["artifacts_compared"], 10)
        for rel, rec in self.cmp["files"].items():
            self.assertTrue(rec["scientific_content_identical"], rel)

    def test_reproduction_was_run_from_the_clean_committed_tree(self):
        r = self.cmp["reproduction"]
        self.assertEqual(r["base_commit"], "aea30bacfa8dfe342198d3d6ff8748e84c9ffd9c")
        self.assertEqual(r["normalize_py_sha256"],
                         "437a78560aea68163f425352348ba95e7d944d72350bd48e74e286feeec75b59")
        self.assertEqual(r["monte_carlo"]["draws_per_seed"], 20000)
        self.assertEqual(r["monte_carlo"]["seeds"], [12345, 24680, 98765, 54321, 13579])

    def test_only_wall_clock_runtime_was_exempted(self):
        self.assertEqual(self.cmp["non_scientific_fields_exempted"],
                         {"control_scores_by_case_seed.csv": ["elapsed_seconds"]})

    def test_comparison_is_pinned_to_the_current_baseline_files(self):
        for rel, rec in self.cmp["files"].items():
            p = A2 / rel
            self.assertTrue(p.exists(), rel)
            self.assertEqual(_sha256(p.read_bytes()), rec["certified_sha256"],
                             f"{rel} changed since the reproduction was certified")


if __name__ == "__main__":
    unittest.main()
