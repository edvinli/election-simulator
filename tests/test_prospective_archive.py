"""Tests for immutable prospective forecast snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prospective_archive.archive import SnapshotCollisionError, write_snapshot
from scripts.prospective_archive.archive import ARCHIVE_SCHEMA_VERSION
from scripts.simulator.engine import simulate_election
from scripts.simulator.pipeline import build_canonical_summary_dict
from scripts.simulator.reproducibility import UNRESOLVED_GIT_COMMIT


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
            self.assertEqual(snapshot["schema_version"], ARCHIVE_SCHEMA_VERSION)
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

    def _canonical_pair(self, root: Path, result) -> tuple[Path, Path]:
        summary = build_canonical_summary_dict(result)
        canonical = root / f"canonical-{summary['deterministic_payload_sha256'][:8]}.json"
        canonical.write_text(json.dumps(summary), encoding="utf-8")
        sidecar = root / f"payload-{summary['deterministic_payload_sha256'][:8]}.sha256"
        sidecar.write_text(summary["deterministic_payload_sha256"] + "\n", encoding="utf-8")
        return canonical, sidecar

    def test_multiple_immutable_snapshots_are_allowed_on_one_calendar_day(self) -> None:
        morning = self._result()
        evening = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=54321)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive"
            first_canonical, first_sidecar = self._canonical_pair(root, morning)
            second_canonical, second_sidecar = self._canonical_pair(root, evening)
            first_path, index_path, first = write_snapshot(
                morning,
                archive_dir=archive,
                generated_at_utc="2026-08-23T09:00:00+00:00",
                canonical_artifact_path=first_canonical,
                canonical_payload_hash_path=first_sidecar,
            )
            second_path, _, second = write_snapshot(
                evening,
                archive_dir=archive,
                generated_at_utc="2026-08-23T18:30:00+00:00",
                canonical_artifact_path=second_canonical,
                canonical_payload_hash_path=second_sidecar,
            )
            # Same as-of date, two immutable generations, neither overwritten.
            self.assertEqual(first["snapshot_date"], second["snapshot_date"])
            self.assertNotEqual(first["generation_id"], second["generation_id"])
            self.assertEqual(first["generation_id"], "20260823T090000Z-" + first["snapshot_id"][:8])
            self.assertEqual(second["generation_id"], "20260823T183000Z-" + second["snapshot_id"][:8])
            # The identity is sortable, so lexical order is chronological.
            self.assertLess(first["generation_id"], second["generation_id"])
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.is_file() and second_path.is_file())
            self.assertEqual(first_path.parent.name, first["generation_id"])
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(index["snapshots"]), 2)
            self.assertEqual(
                [row["generation_id"] for row in index["snapshots"]],
                [first["generation_id"], second["generation_id"]],
            )

    def test_generation_id_collision_is_refused(self) -> None:
        result = self._result()
        other = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=54321)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive"
            canonical, sidecar = self._canonical_pair(root, result)
            other_canonical, other_sidecar = self._canonical_pair(root, other)
            _, _, first = write_snapshot(
                result,
                archive_dir=archive,
                generated_at_utc="2026-08-23T09:00:00+00:00",
                canonical_artifact_path=canonical,
                canonical_payload_hash_path=sidecar,
            )
            # A second, genuinely different forecast that somehow derives the
            # same generation id must fail closed rather than overwrite an
            # immutable snapshot.
            with patch(
                "scripts.prospective_archive.archive.build_generation_id",
                return_value=first["generation_id"],
            ):
                with self.assertRaises(SnapshotCollisionError):
                    write_snapshot(
                        other,
                        archive_dir=archive,
                        generated_at_utc="2026-08-23T09:00:00+00:00",
                        canonical_artifact_path=other_canonical,
                        canonical_payload_hash_path=other_sidecar,
                    )
            # The original snapshot is intact and still the only entry.
            index = json.loads((archive / "index.json").read_text(encoding="utf-8"))
            self.assertEqual([row["generation_id"] for row in index["snapshots"]], [first["generation_id"]])

    def test_appending_to_a_1_0_archive_preserves_every_existing_entry(self) -> None:
        """A pre-extraction index gains an entry without any entry changing."""

        result = self._result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archive"
            archive.mkdir(parents=True)
            historical_entry = {
                "snapshot_id": "06e4debd7150bc845650510c344a222bd7546951c2e3575d55026a93300bdbbc",
                "snapshot_date": "2026-08-23",
                "as_of": "2026-08-23",
                "election_date": "2026-09-13",
                "generated_at_utc": "2026-08-27T12:44:00+00:00",
                "source_git_commit": "bd834a9e069881220c147d3b72d87683f57d69df",
                "model_version": "1.0.0-rc1",
                "seed": 12345,
                "deterministic_payload_sha256": "33e818990cf10994f652ad2d9ea32f1ff762cbd1ed4669e28197a1c8dc892ffd",
                "canonical_artifact_sha256": "84df9e7ef385f8d0c2adc5e961d071c9bba37c8683fab3fe3008ca04f90e46d8",
                "snapshot_file_sha256": "77dffe0c76002170b0bdee2fa9ee49d34a9ae253a38c280debcac7711bbce01a",
                "path": "2026-08-23/snapshot.json",
            }
            index_path = archive / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "archive": "ElectionSimulator prospective forecasts",
                        "snapshots": [dict(historical_entry)],
                        "updated_at_utc": "2026-08-27T12:44:00+00:00",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            canonical, sidecar = self._canonical_pair(root, result)
            write_snapshot(
                result,
                archive_dir=archive,
                generated_at_utc="2026-08-28T09:00:00+00:00",
                canonical_artifact_path=canonical,
                canonical_payload_hash_path=sidecar,
            )
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(index["snapshots"]), 2)
            # The historical entry is carried over unchanged, generation id and
            # all: a 1.0 entry is never retrofitted.
            self.assertEqual(index["snapshots"][0], historical_entry)
            self.assertNotIn("generation_id", index["snapshots"][0])
            # The header declares the newest schema the index now contains.
            self.assertEqual(index["schema_version"], ARCHIVE_SCHEMA_VERSION)
            self.assertIn("generation_id", index["snapshots"][1])

    def test_unresolvable_source_commit_is_a_hard_archive_failure(self) -> None:
        result = self._result()
        result.manifest["source_git_commit"] = UNRESOLVED_GIT_COMMIT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical, sidecar = self._canonical_pair(root, result)
            with self.assertRaisesRegex(ValueError, "resolvable source Git commit"):
                write_snapshot(
                    result,
                    archive_dir=root / "archive",
                    generated_at_utc="2026-08-23T09:00:00+00:00",
                    canonical_artifact_path=canonical,
                    canonical_payload_hash_path=sidecar,
                )

    def test_rest_is_aggregate_and_never_in_seat_surfaces(self) -> None:
        result = self._result()
        self.assertTrue(np.all(result.seats_matrix.sum(axis=1) == 349))
        self.assertEqual(result.summary.parties["REST"].seats_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
