"""Determinism and fail-safe tests for the static publication contract."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.simulator.engine import simulate_election
from scripts.simulator.reproducibility import is_git_worktree_clean
from scripts.static_exporter import export_static_data, validate_published_directory
from scripts.static_exporter.exporter import validate_publication_contract


class StaticExporterTests(unittest.TestCase):
    @staticmethod
    def _clean_result(*, seed: int = 12345):
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=seed)
        # The test process has unrelated uncommitted research artifacts.  The
        # production exporter requires this provenance bit to be an actual
        # boolean true, so use a synthetic clean-source result for positive
        # contract tests.
        result.manifest["source_worktree_clean"] = True
        return result

    def test_export_contract_is_complete_and_deterministic_excluding_timestamp(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = export_static_data(
                result,
                output_dir=root / "first",
                generated_at_utc="2026-08-27T00:00:00+00:00",
                calibration_dir=Path("data/processed"),
            )
            second = export_static_data(
                result,
                output_dir=root / "second",
                generated_at_utc="2026-08-27T01:00:00+00:00",
                calibration_dir=Path("data/processed"),
            )
            self.assertEqual(first["deterministic_content_sha256"], second["deterministic_content_sha256"])
            self.assertNotEqual(first["generated_at_utc"], second["generated_at_utc"])
            publication_root = root / "first"
            pointer = json.loads((publication_root / "current.json").read_text())
            version = publication_root / pointer["path"]
            self.assertEqual(
                {path.name for path in version.iterdir()},
                {"forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json", "manifest.json"},
            )
            self.assertEqual(
                first["deterministic_content_sha256"],
                validate_published_directory(publication_root)["deterministic_content_sha256"],
            )
            self.assertEqual(pointer["publication_generation"], version.name)
            self.assertEqual(
                sum(json.loads((version / "seats.json").read_text())["representative_allocation"]["seats"].values()),
                349,
            )

    def test_dirty_source_is_rejected_for_certified_publication(self) -> None:
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=12345)
        with tempfile.TemporaryDirectory() as tmp:
            for value in (False, 1, "true", None):
                result.manifest["source_worktree_clean"] = value
                with self.assertRaisesRegex(ValueError, "source_worktree_clean"):
                    export_static_data(result, output_dir=Path(tmp) / f"publication-{value}")

    def test_failed_validation_does_not_replace_existing_publication(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            old_manifest = (output / "manifest.json").read_bytes()
            # A malformed result fails while contracts are built, before any
            # staging directory is swapped into the live output.
            with self.assertRaises((AttributeError, KeyError, ValueError)):
                export_static_data(object(), output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00")
            self.assertEqual((output / "manifest.json").read_bytes(), old_manifest)

    def test_change_since_prior_is_explicit_and_uses_median_deltas(self) -> None:
        result = self._clean_result()
        prior = {
            "as_of": "2026-08-22",
            "snapshot_id": "prior-id",
            "deterministic_payload_sha256": "prior-payload",
            "national_vote_summary": {
                party: {"vote_share_median": 1.0} for party in result.summary.parties
            },
            "seat_summary": {
                party: {"median": 1} for party in result.summary.parties if party != "REST"
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            contracts = export_static_data(
                result,
                output_dir=Path(tmp) / "publication",
                generated_at_utc="2026-08-27T00:00:00+00:00",
                prior_snapshot=prior,
            )
            self.assertEqual(contracts["deterministic_content_sha256"].__class__, str)
            forecast = json.loads((Path(tmp) / "publication" / "forecast.json").read_text())
            self.assertEqual(forecast["change_since_prior"]["status"], "AVAILABLE")
            self.assertEqual(forecast["change_since_prior"]["prior_as_of"], "2026-08-22")
            self.assertEqual(
                forecast["change_since_prior"]["seat_median_change"]["M"],
                result.summary.parties["M"].seats_median - 1,
            )
            validate_publication_contract({
                name: json.loads((Path(tmp) / "publication" / name).read_text())
                for name in ("forecast.json", "parties.json", "seats.json", "groups.json", "calibration.json", "metadata.json")
            })

    def test_pointer_failure_keeps_previous_version_loadable(self) -> None:
        first_result = self._clean_result(seed=12345)
        second_result = self._clean_result(seed=54321)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            export_static_data(first_result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
            old_pointer = (output / "current.json").read_bytes()
            old_pointer_data = json.loads(old_pointer)
            old_target = output / old_pointer_data["path"]
            old_manifest = (old_target / "manifest.json").read_bytes()
            with patch("scripts.static_exporter.exporter._atomic_replace_pointer", side_effect=OSError("injected pointer failure")):
                with self.assertRaisesRegex(OSError, "injected pointer failure"):
                    export_static_data(second_result, output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00")
            self.assertEqual((output / "current.json").read_bytes(), old_pointer)
            self.assertEqual((output / old_pointer_data["path"] / "manifest.json").read_bytes(), old_manifest)
            self.assertEqual(validate_published_directory(output)["publication_generation"], old_target.name)
            self.assertEqual(validate_published_directory(old_target)["publication_generation"], old_target.name)
            self.assertEqual(len(list((output / "versions").iterdir())), 2)

    def test_generation_collision_does_not_overwrite_immutable_version(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            fixed_uuid = SimpleNamespace(hex="fixed-generation")
            with patch("scripts.static_exporter.exporter.uuid.uuid4", return_value=fixed_uuid):
                export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T00:00:00+00:00")
                old_pointer = (output / "current.json").read_bytes()
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    export_static_data(result, output_dir=output, generated_at_utc="2026-08-27T01:00:00+00:00")
            self.assertEqual((output / "current.json").read_bytes(), old_pointer)
            self.assertEqual(validate_published_directory(output)["publication_generation"], "fixed-generation")

    def test_existing_real_directory_is_supported_without_directory_swap(self) -> None:
        result = self._clean_result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "publication"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("legacy", encoding="utf-8")
            export_static_data(result, output_dir=output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "legacy")
            self.assertEqual(validate_published_directory(output)["publication_state"], "COMPLETE")

    def test_generated_static_output_does_not_dirty_source_provenance(self) -> None:
        generated_status = SimpleNamespace(stdout="?? files/election-simulator\n?? files/election-simulator/versions/abc/forecast.json\n?? files/.election-simulator.versions/abc/seats.json\n")
        with patch("scripts.simulator.reproducibility.subprocess.run", return_value=generated_status):
            self.assertTrue(is_git_worktree_clean(Path(".")))
        for changed_path in ("scripts/changed.py", "tests/changed.py", "data/processed/input.csv"):
            with patch(
                "scripts.simulator.reproducibility.subprocess.run",
                return_value=SimpleNamespace(stdout=f"?? {changed_path}\n"),
            ):
                self.assertFalse(is_git_worktree_clean(Path(".")))


if __name__ == "__main__":
    unittest.main()
