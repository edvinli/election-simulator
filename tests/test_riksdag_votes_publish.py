"""Riksdag workflow git steps against real temporary bare remotes and shallow clones.

The workflow never moves simulator main: its outputs go to the append-only
branch data/riksdag-votes and the website copy to website master.
"""

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
    prepare_data_branch,
    status,
    wait_until_quiet,
)
from tests.support.git_fixtures import disable_background_maintenance

PATH = "data/processed/riksdag_votes"
SITE = "files/riksdag"
DATA_BRANCH = "data/riksdag-votes"


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


def read(repo: Path, relative: str) -> str:
    return (repo / relative).read_text(encoding="utf-8")


class GitFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.remote = self.bare("remote.git", "main", {
            "files/election-simulator/current.json": '{"generation": 0}\n',
            f"{PATH}/2026-27/votes.json": '{"v": "main"}\n',
            f"{PATH}/manifest.json": '{"files": {}}\n',
        })

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def bare(self, name: str, branch: str, files: dict[str, str]) -> Path:
        remote = self.root / name
        subprocess.run(["git", "init", "--bare", "-q", "-b", branch, str(remote)], check=True)
        disable_background_maintenance(remote)
        seed = self.root / f"seed-{name}"
        subprocess.run(["git", "init", "-q", "-b", branch, str(seed)], check=True)
        identity(seed)
        for i in range(3):
            for relative, text in files.items():
                write(seed, relative, text.replace("}", f', "seed": {i}}}', 1) if i else text)
            git(seed, "add", "-A")
            git(seed, "commit", "-q", "-m", f"seed {i}")
        git(seed, "push", "-q", str(remote), branch)
        return remote

    def clone(self, name: str, *, remote: Path | None = None, branch: str = "main", depth: int | None = 1) -> Path:
        destination = self.root / name
        args = ["git", "clone", "-q", "--branch", branch]
        if depth:
            args += ["--depth", str(depth)]
        subprocess.run(args + [(remote or self.remote).as_uri(), str(destination)], check=True)
        identity(destination)
        return destination

    def ref(self, remote: Path, branch: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(remote), "rev-parse", "--verify", "-q", f"refs/heads/{branch}"], capture_output=True, text=True
        )
        return result.stdout.strip() or None


class StatusTests(GitFixture):
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

    def test_targets_are_fixed_and_main_is_refused(self) -> None:
        repo = self.clone("work")
        write(repo, f"{PATH}/2026-27/votes.json", '{"v": "new"}\n')
        main_before = self.ref(self.remote, "main")
        for path, branch in ((PATH, "main"), (PATH, "master"), (SITE, "main"), (SITE, DATA_BRANCH)):
            with self.assertRaisesRegex(PublishError, "may only be pushed"):
                commit_and_push(repo, path, "msg", branch=branch)
        with self.assertRaisesRegex(PublishError, "refusing"):
            prepare_data_branch(repo, self.root / "wt", branch="main")
        self.assertEqual(self.ref(self.remote, "main"), main_before)
        self.assertEqual(git(repo, "rev-list", "--count", "HEAD"), "1")  # nothing committed


class DataBranchTests(GitFixture):
    def run_workflow(self, name: str, votes: str) -> tuple[bool, str | None, Path]:
        """What one scheduled run does on the simulator side."""
        checkout = self.clone(name)  # shallow main, like actions/checkout
        worktree = self.root / f"{name}-data"
        existed = prepare_data_branch(checkout, worktree)
        write(worktree, f"{PATH}/2026-27/votes.json", votes)
        commit = commit_and_push(worktree, PATH, f"data(riksdag_votes): {name}", branch=DATA_BRANCH)
        self.assertEqual(git(checkout, "status", "--porcelain", "--untracked-files=all"), "")
        return existed, commit, worktree

    def test_first_run_creates_the_branch_from_main_and_later_runs_append(self) -> None:
        main_before = self.ref(self.remote, "main")
        self.assertIsNone(self.ref(self.remote, DATA_BRANCH))

        existed, first, _ = self.run_workflow("run1", '{"v": "day 1"}\n')
        self.assertFalse(existed)
        self.assertEqual(self.ref(self.remote, DATA_BRANCH), first)
        self.assertEqual(self.ref(self.remote, "main"), main_before)
        fresh = self.clone("fresh1", branch=DATA_BRANCH, depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~1"), main_before)  # based on main, not an orphan
        self.assertEqual(git(fresh, "show", "--name-only", "--format=", "HEAD"), f"{PATH}/2026-27/votes.json")

        existed, second, worktree = self.run_workflow("run2", '{"v": "day 2"}\n')
        self.assertTrue(existed)
        self.assertEqual(self.ref(self.remote, "main"), main_before)
        fresh = self.clone("fresh2", branch=DATA_BRANCH, depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD"), second)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~1"), first)  # appended, never rewritten
        self.assertEqual(read(fresh, f"{PATH}/2026-27/votes.json"), '{"v": "day 2"}\n')

        # The next run reads the previous outputs from the branch, not from main.
        checkout = self.clone("run3")
        worktree = self.root / "run3-data"
        self.assertTrue(prepare_data_branch(checkout, worktree))
        self.assertEqual(read(worktree, f"{PATH}/2026-27/votes.json"), '{"v": "day 2"}\n')
        self.assertEqual(read(checkout, f"{PATH}/2026-27/votes.json"), '{"v": "main", "seed": 2}\n')
        self.assertIsNone(commit_and_push(worktree, PATH, "nothing", branch=DATA_BRANCH))
        self.assertEqual(self.ref(self.remote, DATA_BRANCH), second)
        self.assertEqual(self.ref(self.remote, "main"), main_before)

    def test_main_moving_on_does_not_matter(self) -> None:
        self.run_workflow("run1", '{"v": "day 1"}\n')
        publication = self.clone("publication")
        write(publication, "files/election-simulator/current.json", '{"generation": "new"}\n')
        git(publication, "commit", "-q", "-am", "chore: publish election forecast")
        git(publication, "push", "-q", "origin", "HEAD:refs/heads/main")
        main_now = self.ref(self.remote, "main")
        _, commit, _ = self.run_workflow("run2", '{"v": "day 2"}\n')
        self.assertEqual(self.ref(self.remote, DATA_BRANCH), commit)
        self.assertEqual(self.ref(self.remote, "main"), main_now)

    def test_a_manifest_only_change_is_not_committed_but_rides_along(self) -> None:
        _, first, worktree = self.run_workflow("run1", '{"v": "day 1"}\n')
        write(worktree, f"{PATH}/manifest.json", '{"files": {"rezipped": 1}}\n')
        self.assertFalse(status(worktree, PATH))
        self.assertIsNone(commit_and_push(worktree, PATH, "msg", branch=DATA_BRANCH))
        self.assertEqual(self.ref(self.remote, DATA_BRANCH), first)
        write(worktree, f"{PATH}/2026-27/votes.json", '{"v": "new vote"}\n')
        commit = commit_and_push(worktree, PATH, "msg", branch=DATA_BRANCH)
        self.assertEqual(
            sorted(git(worktree, "show", "--name-only", "--format=", commit).splitlines()),
            [f"{PATH}/2026-27/votes.json", f"{PATH}/manifest.json"],
        )

    def test_rejected_push_is_rebased_once_without_force(self) -> None:
        _, first, _ = self.run_workflow("run1", '{"v": "day 1"}\n')
        checkout = self.clone("run2")
        worktree = self.root / "run2-data"
        prepare_data_branch(checkout, worktree)
        # Someone appends to the branch in between (e.g. a manual dispatch).
        other = self.clone("other", branch=DATA_BRANCH)
        write(other, f"{PATH}/2025-26/votes.json", '{"v": "rebuilt"}\n')
        git(other, "add", "-A")
        git(other, "commit", "-q", "-m", "data(riksdag_votes): rebuild 2025/26")
        git(other, "push", "-q", "origin", f"HEAD:refs/heads/{DATA_BRANCH}")
        theirs = git(other, "rev-parse", "HEAD")

        write(worktree, f"{PATH}/2026-27/votes.json", '{"v": "day 2"}\n')
        commit = commit_and_push(worktree, PATH, "data(riksdag_votes): day 2", branch=DATA_BRANCH)
        fresh = self.clone("fresh", branch=DATA_BRANCH, depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD"), commit)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~1"), theirs)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~2"), first)
        self.assertEqual(read(fresh, f"{PATH}/2025-26/votes.json"), '{"v": "rebuilt"}\n')

    def test_conflicting_rebase_fails_without_pushing(self) -> None:
        self.run_workflow("run1", '{"v": "day 1"}\n')
        checkout = self.clone("run2")
        worktree = self.root / "run2-data"
        prepare_data_branch(checkout, worktree)
        other = self.clone("other", branch=DATA_BRANCH)
        write(other, f"{PATH}/2026-27/votes.json", '{"v": "theirs"}\n')
        git(other, "commit", "-q", "-am", "someone else")
        git(other, "push", "-q", "origin", f"HEAD:refs/heads/{DATA_BRANCH}")
        theirs = git(other, "rev-parse", "HEAD")
        write(worktree, f"{PATH}/2026-27/votes.json", '{"v": "ours"}\n')
        with self.assertRaisesRegex(PublishError, "rebase"):
            commit_and_push(worktree, PATH, "update", branch=DATA_BRANCH)
        self.assertEqual(self.ref(self.remote, DATA_BRANCH), theirs)


class WebsiteTests(GitFixture):
    def setUp(self) -> None:
        super().setUp()
        self.website = self.bare("website.git", "master", {
            "files/election-simulator/current.json": '{"generation": 0}\n',
            "index.md": "home\n",
        })

    def test_website_copy_is_rebased_onto_a_concurrent_render_commit(self) -> None:
        site = self.clone("website", remote=self.website, branch="master")
        render = self.clone("render", remote=self.website, branch="master")
        write(render, "files/election-simulator/current.json", '{"generation": "rendered"}\n')
        git(render, "commit", "-q", "-am", "chore: render")
        git(render, "push", "-q", "origin", "HEAD:refs/heads/master")
        rendered = git(render, "rev-parse", "HEAD")

        write(site, f"{SITE}/index.json", "{}\n")
        write(site, f"{SITE}/2026-27/votes.json", "{}\n")
        commit = commit_and_push(site, SITE, "data(riksdag): update", branch="master")
        fresh = self.clone("fresh", remote=self.website, branch="master", depth=None)
        self.assertEqual(git(fresh, "rev-parse", "HEAD"), commit)
        self.assertEqual(git(fresh, "rev-parse", "HEAD~1"), rendered)
        self.assertEqual(
            sorted(git(fresh, "show", "--name-only", "--format=", "HEAD").splitlines()),
            [f"{SITE}/2026-27/votes.json", f"{SITE}/index.json"],
        )
        self.assertEqual(read(fresh, "files/election-simulator/current.json"), '{"generation": "rendered"}\n')


class WaitTests(unittest.TestCase):
    def run_wait(self, counts, *, timeout=600.0, now=datetime(2026, 11, 20, 6, 41, tzinfo=timezone.utc)):
        state = {"t": 0.0}
        sequence = list(counts)

        def active():
            return sequence.pop(0) if sequence else 0

        def sleep(seconds):
            state["t"] += seconds

        quiet = wait_until_quiet(
            active, timeout=timeout, interval=60, sleep=sleep, monotonic=lambda: state["t"], now=lambda: now, log=lambda _m: None
        )
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
