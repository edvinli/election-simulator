"""Tests for immutable prospective forecast snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.prospective_archive.archive import SnapshotCollisionError, write_snapshot
from scripts.simulator.engine import simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict


class TestProspectiveArchive(unittest.TestCase):
    def _result(self):
        return simulate_election(
            as_of="2026-08-23",
            election_date="2026-09-13",
            samples=8,
            seed=12345,
        )

    def test_snapshot_schema_hash_linkage_and_collision_refusal(self) -> None:
        result = self._result()
        summary = build_canonical_summary_dict(result)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical.json"
            canonical.write_text(json.dumps(summary), encoding="utf-8")
            sidecar = root / "payload.sha256"
            sidecar.write_text(summary["deterministic_payload_sha256"] + "\n", encoding="utf-8")
            archive = root / "archive"
            snapshot_path, index_path, snapshot = write_snapshot(
                result,
                archive_dir=archive,
                generated_at_utc="2026-08-27T12:00:00+00:00",
                canonical_artifact_path=canonical,
                canonical_payload_hash_path=sidecar,
            )
            self.assertTrue(snapshot_path.exists())
            self.assertTrue(index_path.exists())
            self.assertEqual(snapshot["schema_version"], "1.0")
            self.assertTrue(snapshot["deterministic_payload_sha256"])
            self.assertEqual(snapshot["hashes"]["canonical_artifact_sha256"], snapshot["canonical_artifact_sha256"])
            self.assertEqual(set(snapshot["threshold_probabilities_4pct"]), {"M", "L", "C", "KD", "S", "V", "MP", "SD"})
            self.assertEqual(snapshot["seat_summary"]["M"]["mean"], snapshot["national_vote_summary"]["M"]["seats_mean"])
            self.assertIn("REST", snapshot["national_vote_summary"])
            with self.assertRaises(SnapshotCollisionError):
                write_snapshot(
                    result,
                    archive_dir=archive,
                    generated_at_utc="2026-08-27T12:01:00+00:00",
                    canonical_artifact_path=canonical,
                    canonical_payload_hash_path=sidecar,
                )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(index["snapshots"]), 1)
            self.assertEqual(index["snapshots"][0]["snapshot_id"], snapshot["snapshot_id"])

    def test_snapshot_identity_is_deterministic_for_fixed_inputs(self) -> None:
        result = self._result()
        summary = build_canonical_summary_dict(result)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "canonical.json"
            canonical.write_text(json.dumps(summary), encoding="utf-8")
            sidecar = root / "payload.sha256"
            sidecar.write_text(summary["deterministic_payload_sha256"] + "\n", encoding="utf-8")
            # Separate archive roots permit comparing the complete JSON identity.
            _, _, first = write_snapshot(result, archive_dir=root / "a", generated_at_utc="2026-08-27T12:00:00+00:00", canonical_artifact_path=canonical, canonical_payload_hash_path=sidecar)
            _, _, second = write_snapshot(result, archive_dir=root / "b", generated_at_utc="2026-08-27T13:00:00+00:00", canonical_artifact_path=canonical, canonical_payload_hash_path=sidecar)
            self.assertEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertEqual(first["deterministic_payload_sha256"], second["deterministic_payload_sha256"])
            self.assertNotEqual(first["generated_at_utc"], second["generated_at_utc"])

    def test_rest_is_aggregate_and_never_in_seat_surfaces(self) -> None:
        result = self._result()
        self.assertTrue(np.all(result.seats_matrix.sum(axis=1) == 349))
        self.assertEqual(result.summary.parties["REST"].seats_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
