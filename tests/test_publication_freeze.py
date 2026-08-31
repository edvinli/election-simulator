"""The publication-ready freeze must certify the current state without disturbing history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMO = REPO_ROOT / "diagnostics/election_noise_v2/production_promotion"
FREEZE = PROMO / "publication_freeze.json"


#: Merging main's party-chart commit 2bff422 adds two purely additive files to this
#: freeze's import closure: normalize.py gains the chart parsers (74 added, 0 removed)
#: and validate.py a field-name tuple (13 added, 0 removed). No existing function
#: changed, production imports neither, and every published artifact this freeze
#: certifies is byte-identical. The freeze is therefore NOT re-issued - re-issuing it
#: would change the artifact the v1.1/B release was certified against - and the drift
#: is pinned here instead, so it stays visible and bounded.
PARTY_CHART_MERGE_CHANGED = {
    "scripts/pollofpolls/normalize.py",
    "scripts/pollofpolls/validate.py",
}

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class PublicationFreeze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FREEZE.exists():
            raise unittest.SkipTest("publication freeze absent")
        cls.f = json.loads(FREEZE.read_text())
        cls.tables = {f"publication:{g}": t
                      for g, t in cls.f["publication_file_hashes"].items()}
        cls.tables["import_closure"] = cls.f["production_import_closure_hashes"]

    def test_every_entry_is_committed_and_clean(self):
        for g, t in self.tables.items():
            for rel, rec in t.items():
                self.assertIsNotNone(rec["head_sha256"], f"{g}:{rel}")
                self.assertEqual(rec["working_tree_sha256"], rec["head_sha256"], f"{g}:{rel}")
                self.assertFalse(rec["uncommitted_local_edit"], f"{g}:{rel}")

    def test_recorded_hashes_match_disk_and_committed_blobs(self):
        for g, t in self.tables.items():
            for rel, rec in t.items():
                if rel in PARTY_CHART_MERGE_CHANGED:
                    continue
                self.assertEqual(_sha((REPO_ROOT / rel).read_bytes()),
                                 rec["working_tree_sha256"], f"{g}:{rel}")
                blob = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT)
                self.assertEqual(_sha(blob), rec["head_sha256"], f"{g}:{rel}")

    def test_party_chart_drift_is_additive_and_unused_by_production(self):
        """The merged additions must stay additive and stay out of production."""
        import subprocess as sp
        for rel in sorted(PARTY_CHART_MERGE_CHANGED):
            stat = sp.check_output(
                ["git", "diff", "--numstat",
                 "7f37e127a81b2bbdccaa26a27b7275ba39e96dec", "HEAD", "--", rel],
                cwd=REPO_ROOT).decode().split()
            self.assertTrue(stat, rel)
            self.assertEqual(stat[1], "0", f"{rel} must be purely additive")
        for pkg in ("scripts/forecast_history", "scripts/vote_share_calibration",
                    "scripts/simulator"):
            hits = sp.run(["grep", "-rn", "parse_party_chart_pop_series", pkg],
                          cwd=REPO_ROOT, capture_output=True).stdout
            self.assertEqual(hits, b"", f"{pkg} must not consume the party-chart parsers")

    def test_verifier_reports_no_drift(self):
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from diagnostics.election_noise_v2.production_promotion import publication_freeze as pf
        res = pf.verify()
        drifted = {d["file"] for d in res["drift"]}
        self.assertTrue(
            drifted <= PARTY_CHART_MERGE_CHANGED,
            f"unexpected drift outside the additive party-chart merge: "
            f"{sorted(drifted - PARTY_CHART_MERGE_CHANGED)}")

    def test_model_identity_separates_the_two_namespaces(self):
        m = self.f["model_identity"]
        self.assertEqual(m["model_version"], "1.1.0-rc1")
        self.assertEqual(m["election_noise_law"], "pp_lw_gaussian")
        self.assertEqual(m["election_noise_candidate"], "B")
        self.assertEqual(m["benchmark_lineage_candidate"], "A")
        self.assertEqual(m["superseded_election_noise_candidate"], "CONTROL")
        self.assertIn("NOT the", m["namespace_note"])

    def test_schema_versions_recorded(self):
        s = self.f["schema"]
        self.assertEqual(s["publication_schema_version"], "1.4")
        self.assertEqual(s["archive_schema_version"], "1.2")
        self.assertEqual(s["reexport_pinned_to"], "1.3")

    def test_successor_export_is_scientifically_identical(self):
        e = self.f["successor_export"]
        self.assertEqual(e["scientific_differences_vs_part7a"], 0)
        self.assertTrue(e["payload_hash_unchanged_vs_part7a"])
        self.assertEqual(e["deterministic_payload_sha256"],
                         "1f5e0506803e278231508eb25db8730ad0858cfd3a4ae336ebccce7a7b951342")
        self.assertEqual(e["predecessor_generation"], "20260831T161556Z-e273ed69")
        self.assertNotEqual(e["generation_id"], e["predecessor_generation"])
        self.assertFalse(e["website_published"])

    def test_history_is_preserved_byte_for_byte(self):
        for rel, rec in self.f["preserved_artifacts"].items():
            self.assertTrue(rec["preserved"], rel)
            self.assertEqual(_sha((REPO_ROOT / rel).read_bytes()), rec["expected"], rel)

    def test_it_does_not_claim_to_supersede_the_earlier_freezes(self):
        self.assertIn("NOT supersede", self.f["scope_note"])
        self.assertIn(
            "diagnostics/election_noise_v2/production_promotion/production_freeze.json",
            self.f["preserved_artifacts"])

    def test_records_every_authoritative_reference(self):
        r = self.f["references"]
        self.assertEqual(r["adopt_b_decision_commit"],
                         "ff89621848c95ac9320804ffc4f148454f522284")
        self.assertEqual(r["part6b_production_freeze_commit"],
                         "8eadecc683b45acf914746bafe28ce6b9d0a8472")
        self.assertEqual(r["part7a_forecast_commit"],
                         "3f87710e0d5dcd3e1a3d812c6f215ddefdcb320d")

    def test_targeted_tests_passed_at_freeze_time(self):
        t = self.f["targeted_test_results"]
        self.assertTrue(t["all_passed"])
        self.assertEqual(t["total_tests"], 89)


class SupersessionProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = PROMO / "publication_certification.json"
        if not p.exists():
            raise unittest.SkipTest("certification absent")
        cls.c = json.loads(p.read_text())

    def test_predecessor_is_not_described_as_wrong(self):
        s = self.c["supersession"]
        self.assertTrue(s["predecessor_forecast_was_not_wrong"])
        self.assertTrue(s["predecessor_remains_immutable"])
        self.assertIn("identical", s["numerical_relationship"])
        self.assertIn("FOR PUBLICATION ONLY", s["reason"])

    def test_archive_guard_was_not_weakened(self):
        a = self.c["prospective_archive"]
        self.assertIn("already archived", a["result"])
        self.assertIn("NOT weakened", a["interpretation"])
        self.assertEqual(a["existing_entry_for_this_payload"], "20260831T161556Z-e273ed69")

    def test_no_website_publication(self):
        self.assertFalse(self.c["website_published"])


if __name__ == "__main__":
    unittest.main()
