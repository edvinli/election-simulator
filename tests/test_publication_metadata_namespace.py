"""The published artifact must state its ElectionNoise identity, without namespace collision.

Two distinct namespaces share the letter "B" and must never be merged:

* ``model.candidate`` - the botten-ada benchmark / model-lineage label. "A" is this
  simulator; "B" would be a rival external model. Unrelated to ElectionNoise.
* ``election_noise_candidate`` - the challenger selected by the preregistered
  ElectionNoise v2 competition. "B" is adopted; "CONTROL" is the superseded
  empirical bootstrap (which is NOT Challenger A).
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts.prospective_archive.archive import (
    ARCHIVE_SCHEMA_VERSION,
    SUPPORTED_ARCHIVE_SCHEMA_VERSIONS,
    build_snapshot,
)
from scripts.simulator.config import (
    ADOPTED_ELECTION_NOISE_CANDIDATE,
    BENCHMARK_LINEAGE_CANDIDATE,
)
from scripts.simulator.engine import simulate_election
from scripts.static_exporter.exporter import (
    PUBLICATION_SCHEMA_VERSION,
    SUPPORTED_PUBLICATION_SCHEMA_VERSIONS,
    _build_contracts,
    validate_publication_contract,
)
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.vote_share_calibration.election_noise_b import (
    ELECTION_NOISE_CANDIDATE_BY_LAW,
    LEGACY_MODEL_ID,
    MODEL_ID,
    election_noise_candidate_for_law,
)


class Namespaces(unittest.TestCase):
    def test_two_constants_are_distinct_and_named_unambiguously(self):
        self.assertEqual(BENCHMARK_LINEAGE_CANDIDATE, "A")
        self.assertEqual(ADOPTED_ELECTION_NOISE_CANDIDATE, "B")

    def test_no_ambiguous_constant_remains(self):
        import scripts.simulator.config as cfg
        import scripts.vote_share_calibration.election_noise_b as enb
        for mod in (cfg, enb):
            self.assertFalse(hasattr(mod, "ADOPTED_CANDIDATE"),
                             f"{mod.__name__} still exposes the ambiguous ADOPTED_CANDIDATE")

    def test_legacy_law_is_control_not_challenger_a(self):
        """ElectionNoise Challenger A was the smoothed bootstrap, never adopted."""
        self.assertEqual(election_noise_candidate_for_law(LEGACY_MODEL_ID), "CONTROL")
        self.assertNotEqual(election_noise_candidate_for_law(LEGACY_MODEL_ID), "A")
        self.assertEqual(election_noise_candidate_for_law(MODEL_ID), "B")
        self.assertIsNone(election_noise_candidate_for_law("nonexistent_law"))
        self.assertEqual(set(ELECTION_NOISE_CANDIDATE_BY_LAW), {MODEL_ID, LEGACY_MODEL_ID})

    def test_no_source_file_hard_codes_the_candidate_letter_in_a_model_block(self):
        for rel in ("scripts/static_exporter/exporter.py",
                    "scripts/prospective_archive/archive.py"):
            text = (REPO_ROOT / rel).read_text()
            self.assertNotIn('"candidate": "A"', text,
                             f"{rel} should use BENCHMARK_LINEAGE_CANDIDATE, not a literal")


class SchemaVersions(unittest.TestCase):
    def test_publication_schema_bumped_additively(self):
        self.assertEqual(PUBLICATION_SCHEMA_VERSION, "1.4")
        for old in ("1.0", "1.1", "1.2", "1.3"):
            self.assertIn(old, SUPPORTED_PUBLICATION_SCHEMA_VERSIONS,
                          "historical publications must stay valid")

    def test_archive_schema_bumped_additively(self):
        self.assertEqual(ARCHIVE_SCHEMA_VERSION, "1.2")
        for old in ("1.0", "1.1"):
            self.assertIn(old, SUPPORTED_ARCHIVE_SCHEMA_VERSIONS)

    def test_reexport_stays_pinned_to_its_documented_target(self):
        from scripts.presentation_reexport.reexport import REEXPORT_TARGET_SCHEMA_VERSION
        self.assertEqual(REEXPORT_TARGET_SCHEMA_VERSION, "1.3")


class ExportedArtifact(unittest.TestCase):
    """Build a small real export and assert both namespaces are correct."""

    @classmethod
    def setUpClass(cls):
        cls.result = simulate_election(as_of="2026-08-24", election_date="2026-09-13",
                                       samples=200, seed=12345)
        cls.summary = build_canonical_summary_dict(cls.result)
        # These tests assert metadata CONTENT, not release certification. The exporter
        # refuses to build contracts unless the worktree is clean, which is a
        # production guard rather than anything about the fields under test, so the
        # flag is set on this in-memory copy only. Nothing here writes an artifact.
        cls.result.manifest["source_worktree_clean"] = True
        cls.contracts = _build_contracts(
            cls.result,
            generated_at_utc="2026-08-31T00:00:00+00:00",
            calibration_dir=None,
            prior_snapshot=None,
        )

    def test_metadata_and_forecast_state_the_election_noise_identity(self):
        for name in ("metadata.json", "forecast.json"):
            c = self.contracts[name]
            self.assertEqual(c["election_noise_law"], "pp_lw_gaussian", name)
            self.assertEqual(c["election_noise_candidate"], "B", name)

    def test_candidate_field_remains_the_benchmark_lineage_label(self):
        for name in ("metadata.json", "forecast.json"):
            self.assertEqual(self.contracts[name]["model"]["candidate"], "A", name)
            self.assertIn("benchmark", self.contracts[name]["model"]["candidate_namespace"])

    def test_law_is_sourced_from_the_run_manifest_not_hard_coded(self):
        self.assertEqual(self.contracts["metadata.json"]["election_noise_law"],
                         self.result.manifest["model_config"]["noise_model"])

    def test_files_do_not_disagree(self):
        m, f = self.contracts["metadata.json"], self.contracts["forecast.json"]
        self.assertEqual(m["election_noise_law"], f["election_noise_law"])
        self.assertEqual(m["election_noise_candidate"], f["election_noise_candidate"])
        self.assertEqual(m["model"]["candidate"], f["model"]["candidate"])

    def test_schema_validation_passes(self):
        validate_publication_contract(self.contracts)

    def test_validator_rejects_a_missing_election_noise_law(self):
        import copy
        bad = copy.deepcopy(self.contracts)
        bad["metadata.json"].pop("election_noise_law")
        with self.assertRaises(ValueError):
            validate_publication_contract(bad)

    def test_validator_rejects_disagreeing_namespaces(self):
        import copy
        bad = copy.deepcopy(self.contracts)
        bad["forecast.json"]["election_noise_candidate"] = "CONTROL"
        with self.assertRaises(ValueError):
            validate_publication_contract(bad)

    def test_validator_rejects_repurposing_the_benchmark_candidate(self):
        import copy
        bad = copy.deepcopy(self.contracts)
        bad["metadata.json"]["model"]["candidate"] = "B"
        with self.assertRaises(ValueError):
            validate_publication_contract(bad)

    @staticmethod
    def _snapshot(result):
        """Build a snapshot against a temporary canonical sidecar for this run.

        The production sidecar pins one canonical forecast; these tests use small
        sample counts, so they supply their own matching pair rather than weakening
        the production cross-check.
        """
        import json as _json
        import tempfile
        from scripts.simulator.pipeline import build_canonical_summary_dict as _b
        summary = _b(result)
        d = Path(tempfile.mkdtemp())
        art = d / "canonical.json"
        art.write_text(_json.dumps(summary, sort_keys=True))
        side = d / "payload.sha256"
        side.write_text(summary["deterministic_payload_sha256"] + "\n")
        return build_snapshot(result, generated_at_utc="2026-08-31T00:00:00+00:00",
                              canonical_artifact_path=art, canonical_payload_hash_path=side)

    def test_archive_snapshot_agrees_with_the_exporter(self):
        snap = self._snapshot(self.result)
        self.assertEqual(snap["model"]["candidate"], "A")
        self.assertEqual(snap["model"]["election_noise_law"], "pp_lw_gaussian")
        self.assertEqual(snap["model"]["election_noise_candidate"], "B")
        self.assertEqual(snap["model"]["election_noise_law"],
                         self.contracts["metadata.json"]["election_noise_law"])
        self.assertEqual(snap["schema_version"], ARCHIVE_SCHEMA_VERSION)

    def test_legacy_law_is_labelled_control_end_to_end(self):
        r = simulate_election(as_of="2026-08-24", election_date="2026-09-13",
                              samples=100, seed=12345, noise_model=LEGACY_MODEL_ID)
        r.manifest["source_worktree_clean"] = True
        snap = self._snapshot(r)
        self.assertEqual(snap["model"]["election_noise_law"], LEGACY_MODEL_ID)
        self.assertEqual(snap["model"]["election_noise_candidate"], "CONTROL")
        self.assertEqual(snap["model"]["candidate"], "A")


class PredecessorSnapshotUntouched(unittest.TestCase):
    PRED = "data/processed/prospective_forecasts/20260831T161556Z-e273ed69/snapshot.json"

    def test_predecessor_is_byte_for_byte_unchanged(self):
        import hashlib
        import subprocess
        ref = "3f87710e0d5dcd3e1a3d812c6f215ddefdcb320d"
        if subprocess.run(["git", "cat-file", "-e", ref + "^{commit}"],
                          cwd=REPO_ROOT, capture_output=True).returncode != 0:
            # Shallow CI clone: the Part-7A commit is not fetched. The immutability
            # guarantee is about content, not clone depth.
            self.skipTest("shallow clone: Part-7A commit unavailable")
        p = REPO_ROOT / self.PRED
        self.assertTrue(p.exists())
        blob = subprocess.check_output(
            ["git", "show", f"3f87710e0d5dcd3e1a3d812c6f215ddefdcb320d:{self.PRED}"],
            cwd=REPO_ROOT)
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
                         hashlib.sha256(blob).hexdigest(),
                         "the Part-7A snapshot is immutable and must not be edited")

    def test_predecessor_identity_is_intact(self):
        d = json.loads((REPO_ROOT / self.PRED).read_text())
        self.assertEqual(d["generation_id"], "20260831T161556Z-e273ed69")
        self.assertEqual(
            d["deterministic_payload_sha256"],
            "1f5e0506803e278231508eb25db8730ad0858cfd3a4ae336ebccce7a7b951342")


if __name__ == "__main__":
    unittest.main()
