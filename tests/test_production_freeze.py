"""The post-adoption production freeze must certify the current production state.

It must also be reconstructible from committed content, and it must NOT be confused
with the historical research freezes, which certify a different thing and are
preserved byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMO = REPO_ROOT / "diagnostics/election_noise_v2/production_promotion"
FREEZE = PROMO / "production_freeze.json"

INTENTIONALLY_CHANGED = {
    "scripts/vote_share_calibration/national_engine.py",
    "scripts/simulator/engine.py",
    "scripts/simulator/config.py",
    "scripts/simulator/reproducibility.py",
}

#: Part 7B1 metadata/schema namespace fix. These files changed deliberately after the
#: Part-6B production freeze was taken, so that freeze is now HISTORICAL: it certifies
#: the production state at its own commit and is preserved byte-for-byte. The current
#: publication-ready state is certified by publication_freeze.json instead.
PART7B1_METADATA_CHANGED = {
    "scripts/static_exporter/exporter.py",
    "scripts/prospective_archive/archive.py",
    "scripts/simulator/config.py",
    "scripts/vote_share_calibration/election_noise_b.py",
    "scripts/presentation_reexport/reexport.py",
    "diagnostics/election_noise_v2/production_promotion/production_freeze.py",
    "tests/test_production_default_is_b.py",
    "tests/test_prospective_archive.py",
    "tests/test_static_exporter.py",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_available() -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--git-dir"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class ProductionFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("production freeze absent")
        cls.f = json.loads(FREEZE.read_text())
        cls.tables = {f"production:{g}": t
                      for g, t in cls.f["production_file_hashes"].items()}
        cls.tables["import_closure"] = cls.f["production_import_closure_hashes"]

    def test_tables_are_not_empty(self):
        self.assertGreaterEqual(len(self.f["production_import_closure_hashes"]), 60)
        self.assertGreaterEqual(len(self.f["production_file_hashes"]), 4)

    def test_every_recorded_entry_was_committed_when_frozen(self):
        for g, t in self.tables.items():
            for rel, rec in t.items():
                self.assertIsNotNone(rec["head_sha256"], f"{g}:{rel} has no committed blob")
                self.assertEqual(rec["working_tree_sha256"], rec["head_sha256"], f"{g}:{rel}")
                self.assertFalse(rec["uncommitted_local_edit"], f"{g}:{rel}")

    def test_drift_is_confined_to_the_part7b1_metadata_fix(self):
        """This freeze is historical after Part 7B1; drift must stay in the known set."""
        drifted = set()
        for g, t in self.tables.items():
            for rel, rec in t.items():
                if _sha((REPO_ROOT / rel).read_bytes()) != rec["working_tree_sha256"]:
                    drifted.add(rel)
        self.assertTrue(
            drifted <= PART7B1_METADATA_CHANGED,
            f"unexpected drift outside the Part-7B1 metadata fix: "
            f"{sorted(drifted - PART7B1_METADATA_CHANGED)}")

    def test_unchanged_entries_still_match_their_committed_blobs(self):
        if not _git_available():
            self.skipTest("git unavailable")
        for g, t in self.tables.items():
            for rel, rec in t.items():
                if rel in PART7B1_METADATA_CHANGED:
                    continue
                blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT)
                self.assertEqual(_sha(blob), rec["head_sha256"], f"{g}:{rel}")

    def test_verifier_drift_is_confined_and_reported(self):
        """verify() semantics are unchanged; the drift it reports is the intended fix."""
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.production_promotion import production_freeze as pf
        res = pf.verify()
        drifted = {d["file"] for d in res["drift"]}
        self.assertTrue(
            drifted <= PART7B1_METADATA_CHANGED,
            f"unexpected drift: {sorted(drifted - PART7B1_METADATA_CHANGED)}")

    def test_records_the_adopted_model_and_version(self):
        a = self.f["adopted_model"]
        self.assertEqual(a["election_noise_law"], "pp_lw_gaussian")
        self.assertEqual(a["candidate"], "B")
        self.assertEqual(a["superseded_law"], "pp_centered_noise")
        self.assertTrue(a["superseded_law_still_selectable"])
        self.assertEqual(a["tunable_hyperparameters"], 0)
        v = self.f["model_version"]
        self.assertEqual(v["model_version"], "1.1.0-rc1")
        self.assertEqual(v["release_tag"], "election-simulator-v1.1-rc1")
        self.assertIn("not declared stable", v["release_status"])

    def test_records_every_authoritative_reference(self):
        r = self.f["references"]
        self.assertEqual(r["adopt_b_decision_commit"],
                         "ff89621848c95ac9320804ffc4f148454f522284")
        self.assertEqual(r["evaluator_refreeze_commit"],
                         "a5b8c7a234acf60cac71ef1ab1439343fae88639")
        self.assertEqual(r["challenger_freeze_commit"],
                         "1450e6f301a98d5d6e4af1357113435534b0e7a9")
        self.assertEqual(r["part6a_production_implementation_commit"],
                         "8c8eaed20292961c8c262d1568b73a9ff1ebd679")
        self.assertEqual(r["part6a_same_input_diagnostic_commit"],
                         "b8705e33ba469be29962164edf96a7f558d127ba")

    def test_same_input_certification_is_recorded_and_passing(self):
        c = self.f["same_input_certification"]
        self.assertTrue(c["default_reproduces_part6a_b"])
        self.assertTrue(c["control_reproduces_archived"])
        self.assertEqual(c["configuration"]["as_of"], "2026-08-24")
        self.assertEqual(c["configuration"]["samples"], 100000)
        self.assertEqual(c["configuration"]["seed"], 12345)
        self.assertFalse(c["configuration"]["noise_model_argument_passed"])
        self.assertFalse(c["configuration"]["polling_inputs_refreshed"])
        self.assertEqual(_sha((PROMO / "default_path_certification.json").read_bytes()),
                         c["sha256"])

    def test_targeted_tests_passed_at_freeze_time(self):
        t = self.f["targeted_test_results"]
        self.assertTrue(t["all_passed"])
        self.assertEqual(t["total_tests"], 38)


class HistoricalScopeIsPreserved(unittest.TestCase):
    """The production freeze must not have disturbed the historical record."""

    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("production freeze absent")
        cls.f = json.loads(FREEZE.read_text())

    def test_historical_artifacts_preserved_byte_for_byte(self):
        for rel, rec in self.f["preserved_historical_artifacts"].items():
            self.assertTrue(rec["preserved"], f"{rel} was not preserved")
            self.assertEqual(_sha((REPO_ROOT / rel).read_bytes()), rec["expected"], rel)

    def test_historical_drift_is_confined_to_the_intentional_flip(self):
        for name, d in self.f["historical_freeze_drift_against_head"].items():
            self.assertTrue(d["all_drift_is_intentional"], name)
            self.assertTrue(set(d["drifted_files"]) <= INTENTIONALLY_CHANGED,
                            f"{name}: {d['drifted_files']}")

    def test_drift_is_reported_not_hidden(self):
        """The freeze must state the drift explicitly rather than claim a clean pass."""
        for name, d in self.f["historical_freeze_drift_against_head"].items():
            self.assertGreater(len(d["drifted_files"]), 0,
                               f"{name}: the flip changed files; that must be recorded")
            self.assertIn("historical", d["interpretation"].lower())

    def test_scope_note_distinguishes_the_two_freezes(self):
        self.assertIn("does NOT supersede", self.f["scope_note"])

    def test_mathematical_law_implementations_untouched(self):
        cf = json.loads((REPO_ROOT / "diagnostics/election_noise_v2/challengers"
                         / "challenger_implementation_freeze.json").read_text())
        rec = {**cf["frozen_dependency_hashes"], **cf["import_closure_hashes"]}
        for g in cf["implementation_hashes"].values():
            rec.update(g)
        for rel in ("diagnostics/election_noise_v2/challengers/challenger_b.py",
                    "scripts/vote_share_calibration/models.py",
                    "scripts/election_layer_v2/transfer.py",
                    "scripts/election_layer_v2/residuals_pool.py",
                    "scripts/geography/projection.py",
                    "scripts/mandates/allocator.py"):
            self.assertEqual(_sha((REPO_ROOT / rel).read_bytes()),
                             rec[rel]["working_tree_sha256"], rel)


if __name__ == "__main__":
    unittest.main()
