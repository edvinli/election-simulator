"""The fixture hardening that keeps temporary repositories out of their own way."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.support.git_fixtures import (
    NO_BACKGROUND_MAINTENANCE,
    disable_background_maintenance,
)


def _config(repository: Path, key: str) -> str:
    return subprocess.run(
        ["git", "config", "--get", key],
        cwd=str(repository), check=True, capture_output=True, text=True,
    ).stdout.strip()


class DisableBackgroundMaintenanceTests(unittest.TestCase):
    def test_a_worktree_repository_is_left_quiescent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            disable_background_maintenance(root)
            for key, value in NO_BACKGROUND_MAINTENANCE:
                self.assertEqual(_config(root, key), value, key)

    def test_a_bare_repository_is_left_quiescent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            remote = Path(tmp) / "remote.git"
            subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
            disable_background_maintenance(remote)
            for key, value in NO_BACKGROUND_MAINTENANCE:
                self.assertEqual(_config(remote, key), value, key)

    def test_the_settings_cover_both_halves_of_the_race(self) -> None:
        """Naming them, so neither can be dropped as redundant.

        ``gc.auto``/``maintenance.auto`` stop a run being scheduled at all;
        ``gc.autoDetach`` makes any run that is somehow still scheduled finish
        inside the synchronous git call rather than outliving the fixture and
        racing ``rmtree``.
        """

        keys = dict(NO_BACKGROUND_MAINTENANCE)
        self.assertEqual(keys["gc.auto"], "0")
        self.assertEqual(keys["maintenance.auto"], "false")
        self.assertEqual(keys["gc.autoDetach"], "false")

    def test_the_automation_fixture_applies_it_before_committing(self) -> None:
        """A commit is what triggers the first maintenance run.

        Hardening the repository after the fixture's own commit would leave
        exactly the window that broke the 07:30 publication, so the real
        fixture is checked, not just the helper.
        """

        from tests.test_election_automation import ElectionAutomationTests

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README").write_text("fixture\n", encoding="utf-8")
            ElectionAutomationTests._init_git(root)
            for key, value in NO_BACKGROUND_MAINTENANCE:
                self.assertEqual(_config(root, key), value, key)


if __name__ == "__main__":
    unittest.main()
