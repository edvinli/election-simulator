"""Offline publication orchestration and fail-closed archive tests."""

from __future__ import annotations

import json
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
            # The canonical contract publishes no flat aliases, so read the
            # published version through the pointer as the browser does.
            publication = root / "publication"
            pointer = json.loads((publication / "current.json").read_text())
            version = publication / pointer["path"]
            old_pointer = (publication / "current.json").read_bytes()
            old_manifest = (version / "manifest.json").read_bytes()

            second = run_publication_pipeline(**kwargs)
            self.assertEqual(second.status, "COLLISION")
            self.assertEqual((version / "manifest.json").read_bytes(), old_manifest)
            self.assertEqual((publication / "current.json").read_bytes(), old_pointer)
            self.assertEqual(len(list((root / "archive").glob("*/snapshot.json"))), 1)
            self.assertEqual(len(list((publication / "versions").iterdir())), 1)
            # The archive snapshot and the published version share one id.
            self.assertEqual(first.snapshot["generation_id"], pointer["publication_generation"])
            self.assertEqual(version.name, first.snapshot["generation_id"])

    def test_two_publications_on_one_day_produce_two_immutable_generations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                "as_of": "2026-08-23",
                "election_date": "2026-09-13",
                "samples": 2,
                "processed_root": Path("data/processed"),
                "archive_dir": root / "archive",
                "publication_dir": root / "publication",
            }
            morning = run_publication_pipeline(
                **common,
                seed=12345,
                generated_at_utc="2026-08-23T09:00:00+00:00",
                simulation_runner=self._clean_runner,
            )
            evening = run_publication_pipeline(
                **common,
                seed=54321,
                generated_at_utc="2026-08-23T18:30:00+00:00",
                simulation_runner=self._clean_runner,
            )
            self.assertEqual(morning.status, "PUBLISHED")
            self.assertEqual(evening.status, "PUBLISHED")

            publication = root / "publication"
            first_version = publication / "versions" / morning.snapshot["generation_id"]
            first_bytes = {path.name: path.read_bytes() for path in first_version.iterdir()}

            # Same calendar day, two archived snapshots and two published
            # versions, each addressable and neither rewritten.
            self.assertNotEqual(morning.snapshot["generation_id"], evening.snapshot["generation_id"])
            self.assertLess(morning.snapshot["generation_id"], evening.snapshot["generation_id"])
            self.assertEqual(len(list((root / "archive").glob("*/snapshot.json"))), 2)
            self.assertEqual(
                {path.name for path in (publication / "versions").iterdir()},
                {morning.snapshot["generation_id"], evening.snapshot["generation_id"]},
            )
            self.assertEqual({path.name: path.read_bytes() for path in first_version.iterdir()}, first_bytes)

            pointer = json.loads((publication / "current.json").read_text())
            self.assertEqual(pointer["publication_generation"], evening.snapshot["generation_id"])
            # The later publication compares against the earlier calendar day,
            # not against the earlier snapshot from the same day.
            forecast = json.loads((publication / pointer["path"] / "forecast.json").read_text())
            self.assertEqual(forecast["change_since_prior"]["status"], "NOT_AVAILABLE_NO_PRIOR_SNAPSHOT")

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
                simulation_runner=self._dirty_runner,
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
