"""Offline publication orchestration and fail-closed archive tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.simulator.engine import simulate_election
from scripts.publication_pipeline import (
    run_publication_pipeline,
    validate_existing_inputs,
)
from scripts.publication_pipeline.pipeline import PipelineInputError


class PublicationPipelineTests(unittest.TestCase):
    @staticmethod
    def _clean_runner(**kwargs):
        result = simulate_election(**kwargs)
        result.manifest["source_worktree_clean"] = True
        return result

    @staticmethod
    def _dirty_runner(**kwargs):
        result = simulate_election(**kwargs)
        result.manifest["source_worktree_clean"] = False
        return result

    def test_existing_inputs_are_validated_without_network(self) -> None:
        report = validate_existing_inputs(Path("data/processed"))
        self.assertEqual(report["status"], "OFFLINE_VALIDATED")
        self.assertEqual(report["network_access"], "none")
        self.assertEqual(report["poll_validation"]["error_count"], 0)
        self.assertTrue(report["input_sha256"]["poll_timeseries"])

    def test_missing_processed_input_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PipelineInputError):
                validate_existing_inputs(tmp)

    def test_archive_collision_does_not_replace_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = {
                "as_of": "2026-08-23",
                "election_date": "2026-09-13",
                "samples": 2,
                "seed": 12345,
                "processed_root": Path("data/processed"),
                "archive_dir": root / "archive",
                "publication_dir": root / "publication",
                "generated_at_utc": "2026-08-27T12:00:00+00:00",
                "simulation_runner": self._clean_runner,
            }
            first = run_publication_pipeline(**kwargs)
            self.assertEqual(first.status, "PUBLISHED")
            old_manifest = (root / "publication" / "manifest.json").read_bytes()
            second = run_publication_pipeline(**kwargs)
            self.assertEqual(second.status, "COLLISION")
            self.assertEqual((root / "publication" / "manifest.json").read_bytes(), old_manifest)
            self.assertEqual(len(list((root / "archive").glob("*/snapshot.json"))), 1)

    def test_custom_processed_root_fails_before_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            called = False

            def unexpected_runner(**kwargs):
                nonlocal called
                called = True
                raise AssertionError("custom processed root must be rejected before simulation")

            run = run_publication_pipeline(
                processed_root=Path(tmp),
                append_archive=False,
                export_publication=False,
                simulation_runner=unexpected_runner,
            )
            self.assertEqual(run.status, "FAILED")
            self.assertEqual(run.error["type"], "PipelineInputError")
            self.assertIn("Custom processed_root is not supported", run.error["message"])
            self.assertFalse(called)

    def test_dirty_source_fails_before_archive_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = run_publication_pipeline(
                as_of="2026-08-23",
                samples=2,
                processed_root=Path("data/processed"),
                archive_dir=Path(tmp) / "archive",
                publication_dir=Path(tmp) / "publication",
                append_archive=True,
                export_publication=True,
            )
            self.assertEqual(run.status, "FAILED")
            self.assertEqual(run.error["type"], "PipelineInputError")
            self.assertIn("source_worktree_clean", run.error["message"])
            self.assertFalse((Path(tmp) / "archive").exists())

    def test_dirty_source_archive_only_fails_before_archive_append(self) -> None:
        """An archive is certified evidence even when no static export is requested."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = run_publication_pipeline(
                as_of="2026-08-23",
                samples=2,
                processed_root=Path("data/processed"),
                archive_dir=root / "archive",
                publication_dir=root / "publication",
                append_archive=True,
                export_publication=False,
                simulation_runner=self._dirty_runner,
            )
            self.assertEqual(run.status, "FAILED")
            self.assertEqual(run.error["type"], "PipelineInputError")
            self.assertIn("source_worktree_clean", run.error["message"])
            self.assertEqual(run.stages[-1]["name"], "source_certification")
            self.assertFalse((root / "archive").exists())


if __name__ == "__main__":
    unittest.main()
