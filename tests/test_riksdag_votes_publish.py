"""Riksdag workflow git steps against real temporary bare remotes and shallow clones."""

import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.riksdag_votes.publish import (
    PublishError,
    changed_paths,
    commit_and_push,
    in_protected_interval,
    status,
    wait_until_quiet,
)
from tests.support.git_fixtures import disable_background_maintenance

PATH = "data/processed/riksdag_votes"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def identity(repo: Path) -> None:
    disable_background_maintenance(repo)
    git(repo, "config", "user.name", "github-actions[bot]")
    git(repo, "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")


def write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class PublishGitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.remote = root / "remote.git"
        subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(self.remote)], check=True)
        disable_background_maintenance(self.remote)
        seed = root / "seed"
        subprocess.run(["git", "init", "-q", "-b", "main", str(seed)], check=True)
        identity(seed)
        for i in range(3):
            write(seed, "files/election-simulator/current.json", f'{{"generation": {i}}}\n')
            write(seed, f"{PATH}/2026-27/votes.json", f'{{"v": {i}}}\n')
            git(seed, "add", "-A")
            git(seed, "commit", "-q", "-m", f"seed {i}")
        git(seed, "push", "-q", str(self.remote), "main")
        self.root = root

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def clone(self, name: str, depth: int | None = 1) -> Path:
        destination = self.root / name
        args = ["git", "clone", "-q", "--branch", "main"]
        if depth:
            args += ["--depth", str(depth)]
        subprocess.run(args + [self.remote.as_uri(), str(destination)], check=True)
        identity(destination)
        return destination

    def test_status_rejects_changes_outside_the_allowed_path(self) -> None:
        repo = self.clone("work")
        self.assertFalse(status(repo, PATH))
        write(repo, f"{PATH}/2026-27/summary.json", "{}\n")
        self.assertTrue(status(repo, PATH))
        write(repo, "files/election-simulator/current.json", "{}\n")
        with self.assertRaisesRegex(PublishError, "outside"):
            status(repo, PATH)
        with self.assertRaisesRegex(PublishError, "allowed"):
            status(repo, "files/election-simulator")
        self.assertIn("files/election-simulator/current.json", changed_paths(repo))

    def test_nothing_changed_commits_nothing(self) -> None:
        repo = self.clone("work")
        before = git(repo, "rev-parse", "HEAD")
        self.assertIsNone(commit_and_push(repo, PATH, "msg", branch="main"))
        self.assertEqual(git(repo, "rev-parse", "HEAD"), before)

    def test_push_lands_and_is_verified_in_a_fresh_clone(self) -> None:
        repo = self.clone("work")
        write(repo, f"{PATH}/2026-27/votes.json", '{"v": "new"}\n')
        commit = commit_and_push(repo, PATH, "data(riksdag_votes): update", branch="main")
        fresh = self.clone("fresh", depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD"), commit)
        self.assertEqual((fresh / PATH / "2026-27/votes.json").read_text(encoding="utf-8"), '{"v": "new"}\n')
        self.assertEqual(git(fresh, "show", "--name-only", "--format=", "HEAD"), f"{PATH}/2026-27/votes.json")

    def test_rejected_push_is_rebased_onto_a_concurrent_publication_commit(self) -> None:
        repo = self.clone("work")  # shallow, like the workflow's checkout
        publication = self.clone("publication")
        write(publication, "files/election-simulator/current.json", '{"generation": "concurrent"}\n')
        git(publication, "commit", "-q", "-am", "chore: publish election forecast")
        git(publication, "push", "-q", "origin", "HEAD:refs/heads/main")
        concurrent = git(publication, "rev-parse", "HEAD")

        write(repo, f"{PATH}/2026-27/votes.json", '{"v": "new"}\n')
        commit = commit_and_push(repo, PATH, "data(riksdag_votes): update", branch="main")
        fresh = self.clone("fresh", depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD"), commit)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~1"), concurrent)  # theirs kept, ours on top, no force
        self.assertEqual(
            (fresh / "files/election-simulator/current.json").read_text(encoding="utf-8"), '{"generation": "concurrent"}\n'
        )
        self.assertEqual((fresh / PATH / "2026-27/votes.json").read_text(encoding="utf-8"), '{"v": "new"}\n')

    def test_conflicting_rebase_fails_without_pushing(self) -> None:
        repo = self.clone("work")
        other = self.clone("other")
        write(other, f"{PATH}/2026-27/votes.json", '{"v": "theirs"}\n')
        git(other, "commit", "-q", "-am", "someone else wrote riksdag data")
        git(other, "push", "-q", "origin", "HEAD:refs/heads/main")
        theirs = git(other, "rev-parse", "HEAD")
        write(repo, f"{PATH}/2026-27/votes.json", '{"v": "ours"}\n')
        with self.assertRaisesRegex(PublishError, "rebase"):
            commit_and_push(repo, PATH, "update", branch="main")
        self.assertEqual(git(self.remote, "rev-parse", "main"), theirs)


class WaitTests(unittest.TestCase):
    def run_wait(self, counts, *, timeout=600.0, now=datetime(2026, 11, 20, 6, 41, tzinfo=timezone.utc)):
        state = {"t": 0.0}
        sequence = list(counts)

        def active():
            return sequence.pop(0) if sequence else 0

        def sleep(seconds):
            state["t"] += seconds

        quiet = wait_until_quiet(active, timeout=timeout, interval=60, sleep=sleep, monotonic=lambda: state["t"], now=lambda: now, log=lambda _m: None)
        return quiet, state["t"]

    def test_two_quiet_checks_are_required(self) -> None:
        self.assertEqual(self.run_wait([0, 0]), (True, 60.0))
        self.assertEqual(self.run_wait([1, 0, 1, 0, 0]), (True, 240.0))

    def test_unknown_counts_as_busy_and_deadline_defers(self) -> None:
        quiet, _ = self.run_wait([None] * 100, timeout=300)
        self.assertFalse(quiet)

    def test_protected_interval(self) -> None:
        self.assertTrue(in_protected_interval(datetime(2026, 11, 20, 18, 30, tzinfo=timezone.utc)))
        self.assertTrue(in_protected_interval(datetime(2026, 11, 20, 21, 59, tzinfo=timezone.utc)))
        self.assertFalse(in_protected_interval(datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)))
        self.assertFalse(in_protected_interval(datetime(2026, 11, 20, 6, 41, tzinfo=timezone.utc)))
        quiet, _ = self.run_wait([0] * 100, timeout=300, now=datetime(2026, 11, 20, 19, 0, tzinfo=timezone.utc))
        self.assertFalse(quiet)


if __name__ == "__main__":
    unittest.main()
