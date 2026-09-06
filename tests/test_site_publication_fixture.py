"""The frozen site publication, and its independence from production.

PR #18 froze the history and polling inputs the publication tests read. The
site was the last live input: ``_production_fixture`` copied the committed
``files/election-simulator`` tree, so the prior generation the tests started
from was whatever production had last published. Generation ids sort
lexicographically by their UTC timestamp, so a synthetic generation could sort
*behind* the live one and be refused by a generation-agreement guard -- a
failure with nothing to do with the behaviour under test, and one that a later
synthetic timestamp would only postpone.

These tests hold the replacement in place: the fixture is valid by the real
validators, it sorts before anything the tests publish, and no publication
production makes can move either fact.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.static_exporter import validate_published_directory
from scripts.static_exporter.exporter import PUBLICATION_FILES, validate_publication_version
from tests.history_fixtures import FROZEN_AS_OF
from tests.site_publication_fixture import (
    FIXTURE_ROOT,
    FROZEN_GENERATIONS,
    FROZEN_SOURCE_GIT_COMMIT,
    frozen_site_generation,
    install_frozen_site_publication,
)
# Imported as a module, not by name: importing the TestCase class would make
# unittest collect all of its tests a second time in this module.
from tests import test_election_automation as automation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIVE_PUBLICATION = REPOSITORY_ROOT / "files" / "election-simulator"

# The publication tests stamp their runs at FROZEN_AS_OF, so the earliest
# generation id any of them can produce begins at that date's midnight.
EARLIEST_SYNTHETIC_GENERATION = FROZEN_AS_OF.replace("-", "") + "T000000Z"


class FrozenFixtureIsValid(unittest.TestCase):
    """Validated by the real publication validators, not by re-derived hashes."""

    def test_the_active_version_passes_the_real_published_directory_validator(self) -> None:
        manifest = validate_published_directory(FIXTURE_ROOT)
        self.assertEqual(manifest["publication_state"], "COMPLETE")
        self.assertEqual(manifest["publication_generation"], frozen_site_generation())
        self.assertIs(manifest["source_worktree_clean"], True)
        self.assertEqual(manifest["source_git_commit"], FROZEN_SOURCE_GIT_COMMIT)

    def test_every_committed_version_validates_in_isolation(self) -> None:
        versions = sorted(path for path in (FIXTURE_ROOT / "versions").iterdir())
        self.assertEqual(len(versions), len(FROZEN_GENERATIONS))
        for version in versions:
            with self.subTest(version.name):
                validate_publication_version(version, expected_generation=version.name)
                for filename in (*PUBLICATION_FILES, "manifest.json"):
                    self.assertTrue((version / filename).is_file(), filename)

    def test_the_pointer_addresses_the_newest_committed_version(self) -> None:
        pointer = json.loads((FIXTURE_ROOT / "current.json").read_text())
        newest = max(path.name for path in (FIXTURE_ROOT / "versions").iterdir())
        self.assertEqual(pointer["publication_generation"], newest)
        self.assertEqual(pointer["path"], f"versions/{newest}")

    def test_the_fixture_carries_a_prior_generation_to_lag_behind(self) -> None:
        versions = sorted(path.name for path in (FIXTURE_ROOT / "versions").iterdir())
        self.assertGreater(len(versions), 1, "a stale pointer needs an older generation")
        self.assertLess(versions[0], frozen_site_generation())


class FrozenFixtureSortsBeforeTheTests(unittest.TestCase):
    def test_it_sorts_before_every_generation_the_tests_can_publish(self) -> None:
        self.assertLess(frozen_site_generation(), EARLIEST_SYNTHETIC_GENERATION)
        for spec in FROZEN_GENERATIONS:
            stamp = str(spec["generated_at_utc"])[:10].replace("-", "")
            self.assertLess(stamp, FROZEN_AS_OF.replace("-", ""))


class ProductionCannotBreakThesePublicationTests(unittest.TestCase):
    """The regression this whole fixture exists for."""

    def test_the_publication_fixture_installs_the_frozen_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, site = automation.ElectionAutomationTests._production_fixture(Path(tmp))
            for repository in (source, site):
                pointer = json.loads(
                    (repository / "files/election-simulator/current.json").read_text()
                )
                self.assertEqual(pointer["publication_generation"], frozen_site_generation())
                validate_published_directory(repository / "files/election-simulator")

    def test_advancing_the_live_pointer_cannot_reach_the_fixture(self) -> None:
        """A publication tomorrow changes current.json; the fixture is unmoved."""

        live_before = json.loads((LIVE_PUBLICATION / "current.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            # A copy of the live tree, advanced to a generation far beyond any
            # the tests use. Nothing the publication tests read may follow it.
            advanced = Path(tmp) / "live"
            advanced.mkdir()
            (advanced / "current.json").write_text(
                json.dumps({**live_before, "publication_generation": "20991231T235959Z-ffffffff"})
            )
            installed = install_frozen_site_publication(Path(tmp) / "fixture-site")
            self.assertEqual(installed, frozen_site_generation())
            self.assertLess(installed, "20991231T235959Z-ffffffff")
            self.assertLess(installed, EARLIEST_SYNTHETIC_GENERATION)

        # The live pointer is untouched by any of this, and is not the fixture.
        self.assertEqual(json.loads((LIVE_PUBLICATION / "current.json").read_text()), live_before)
        self.assertNotEqual(live_before["publication_generation"], frozen_site_generation())

    def test_only_the_named_history_artifact_still_reads_the_live_tree(self) -> None:
        """The integration lane is two history tests, and nothing else."""

        source = (REPOSITORY_ROOT / "tests" / "test_election_automation.py").read_text()
        live_reads = [
            line.strip()
            for line in source.splitlines()
            if "REPOSITORY_ROOT" in line and "files" in line and "election-simulator" in line
        ]
        self.assertEqual(
            live_reads,
            ['REPOSITORY_ROOT / "files" / "election-simulator" / "history" / "coalition-timeseries.json"'],
        )
        self.assertTrue(automation.LIVE_HISTORY_ARTIFACT.is_file())


if __name__ == "__main__":
    unittest.main()
