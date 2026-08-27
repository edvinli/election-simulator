"""Tests for cross-repository mirroring of certified publication generations."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.simulator.engine import simulate_election
from scripts.site_publisher import GENERATION_FILES, SitePublishError, publish_generation_to_site
from scripts.site_publisher.publisher import SITE_PUBLICATION_RELATIVE
from scripts.static_exporter import export_static_data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SitePublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.site = self.tmp / "website"
        (self.site / "files").mkdir(parents=True)
        self.source = self.tmp / "simulator-publication"

    def _publish(self, *, seed: int = 12345, generated_at_utc: str = "2026-08-27T00:00:00+00:00") -> dict:
        result = simulate_election(as_of="2026-08-23", election_date="2026-09-13", samples=8, seed=seed)
        result.manifest["source_worktree_clean"] = True
        return export_static_data(
            result,
            output_dir=self.source,
            generated_at_utc=generated_at_utc,
            calibration_dir=REPOSITORY_ROOT / "data" / "processed",
        )

    @property
    def _site_publication(self) -> Path:
        return self.site / SITE_PUBLICATION_RELATIVE

    def test_mirrors_seven_real_files_and_writes_the_pointer_last(self) -> None:
        manifest = self._publish()
        generation = manifest["publication_generation"]
        report = publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)

        self.assertEqual(report["status"], "MIRRORED")
        self.assertEqual(report["generation"], generation)
        self.assertFalse(report["committed"])
        self.assertFalse(report["pushed"])

        destination = self._site_publication / "versions" / generation
        self.assertEqual({path.name for path in destination.iterdir()}, set(GENERATION_FILES))
        for path in destination.iterdir():
            self.assertFalse(path.is_symlink(), f"{path} must be a real file")
            self.assertTrue(path.is_file())
            self.assertEqual(
                path.read_bytes(),
                (self.source / "versions" / generation / path.name).read_bytes(),
            )

        pointer_path = self._site_publication / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        self.assertEqual(pointer["publication_generation"], generation)
        self.assertEqual(pointer["path"], f"versions/{generation}")
        self.assertEqual(pointer["publication_state"], "COMPLETE")
        # The pointer is the last write of the mirror operation.
        for path in destination.iterdir():
            self.assertLessEqual(path.stat().st_mtime_ns, pointer_path.stat().st_mtime_ns)

    def test_refuses_to_overwrite_an_existing_generation(self) -> None:
        manifest = self._publish()
        generation = manifest["publication_generation"]
        publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        destination = self._site_publication / "versions" / generation
        before = {path.name: path.read_bytes() for path in destination.iterdir()}
        pointer_before = (self._site_publication / "current.json").read_bytes()

        with self.assertRaisesRegex(SitePublishError, "existing published generation"):
            publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)

        self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()}, before)
        self.assertEqual((self._site_publication / "current.json").read_bytes(), pointer_before)

    def test_previous_generation_stays_byte_identical_after_a_new_publish(self) -> None:
        first = self._publish(seed=12345, generated_at_utc="2026-08-27T00:00:00+00:00")
        publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        first_generation = first["publication_generation"]
        first_destination = self._site_publication / "versions" / first_generation
        first_bytes = {path.name: path.read_bytes() for path in first_destination.iterdir()}

        second = self._publish(seed=54321, generated_at_utc="2026-08-27T01:00:00+00:00")
        publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        second_generation = second["publication_generation"]
        self.assertNotEqual(first_generation, second_generation)

        # The earlier immutable generation is untouched; only the pointer moved.
        self.assertEqual({path.name: path.read_bytes() for path in first_destination.iterdir()}, first_bytes)
        pointer = json.loads((self._site_publication / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(pointer["publication_generation"], second_generation)
        self.assertEqual(
            {path.name for path in (self._site_publication / "versions").iterdir()},
            {first_generation, second_generation},
        )

    def test_uncertified_source_generation_is_refused(self) -> None:
        manifest = self._publish()
        generation = manifest["publication_generation"]
        metadata_path = self.source / "versions" / generation / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_worktree_clean"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        self.assertFalse((self._site_publication / "versions" / generation).exists())
        self.assertFalse((self._site_publication / "current.json").exists())

    def test_pointer_is_not_written_when_destination_validation_fails(self) -> None:
        self._publish()
        with patch(
            "scripts.site_publisher.publisher.validate_publication_version",
            side_effect=[{"deterministic_content_sha256": "a"}, ValueError("destination is corrupt")],
        ):
            with self.assertRaisesRegex(ValueError, "destination is corrupt"):
                publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        self.assertFalse((self._site_publication / "current.json").exists())

    def test_no_pointer_mode_installs_the_version_only(self) -> None:
        manifest = self._publish()
        report = publish_generation_to_site(
            site_repo=self.site, source_publication_dir=self.source, update_pointer=False
        )
        self.assertFalse(report["pointer_written"])
        self.assertTrue((self._site_publication / "versions" / manifest["publication_generation"]).is_dir())
        self.assertFalse((self._site_publication / "current.json").exists())

    def test_missing_site_repo_is_refused(self) -> None:
        self._publish()
        with self.assertRaisesRegex(SitePublishError, "existing directory"):
            publish_generation_to_site(
                site_repo=self.tmp / "does-not-exist", source_publication_dir=self.source
            )

    def test_publishing_a_repository_into_itself_is_refused(self) -> None:
        self._publish()
        with self.assertRaisesRegex(SitePublishError, "into itself"):
            publish_generation_to_site(site_repo=self.source, source_publication_dir=self.source)

    def test_legacy_flat_files_on_the_site_are_left_untouched(self) -> None:
        """Mirroring never migrates or rewrites the pre-extraction payload."""

        legacy_source = REPOSITORY_ROOT / "tests" / "fixtures" / "legacy_flat_publication_2026_08_27"
        self._site_publication.mkdir(parents=True)
        legacy_bytes = {}
        for path in sorted(legacy_source.glob("*.json")):
            shutil.copyfile(path, self._site_publication / path.name)
            legacy_bytes[path.name] = (self._site_publication / path.name).read_bytes()

        self._publish()
        publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)

        for name, expected in legacy_bytes.items():
            self.assertEqual((self._site_publication / name).read_bytes(), expected, f"{name} was modified")
        # The legacy payload is never copied into the immutable version store.
        for version in (self._site_publication / "versions").iterdir():
            metadata = json.loads((version / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["source_worktree_clean"])

    def test_publisher_never_invokes_git(self) -> None:
        self._publish()
        with patch("subprocess.run", side_effect=AssertionError("site publisher must not run git")):
            report = publish_generation_to_site(site_repo=self.site, source_publication_dir=self.source)
        self.assertEqual(report["status"], "MIRRORED")


if __name__ == "__main__":
    unittest.main()
