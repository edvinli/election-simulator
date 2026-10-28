"""The publication skips its contract suite only when Full CI provably ran it.

Each case builds a real repository, because the rule is about history -- which
commits separate HEAD from a green run, who made them, what they touched --
and a mocked git would only restate the implementation.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.ci.verified_code import BOT_EMAIL, is_bot_owned, main, verify

HUMAN_EMAIL = "someone@example.com"
WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/election-simulator-publication.yml"


class VerifiedCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "repo"
        self.repo.mkdir()
        self._git("init", "-q", "-b", "main")
        self.base = self._commit({"scripts/model.py": "x = 1\n"}, HUMAN_EMAIL)

    def _git(self, *args: str, email: str = HUMAN_EMAIL) -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": email,
            "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
        }
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True, capture_output=True, text=True, env=env,
        ).stdout.strip()

    def _commit(self, files: dict[str, str], email: str) -> str:
        for rel, text in files.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            self._git("add", rel)
        self._git("commit", "-q", "-m", "c", email=email)
        return self._git("rev-parse", "HEAD")

    def _bot_publication(self, n: int = 1) -> str:
        return self._commit({
            f"data/processed/prospective_forecasts/g{n}/snapshot.json": "{}\n",
            "files/election-simulator/current.json": f"{{\"n\": {n}}}\n",
            "data/raw/pollofpolls/swedishpolls_polls.csv": f"{n}\n",
        }, BOT_EMAIL)

    @staticmethod
    def _run(sha: str, *, run_id: int = 1, created: str = "2026-10-24T12:00:00Z",
             **overrides) -> dict:
        run = {
            "id": run_id, "head_sha": sha, "head_branch": "main",
            "path": ".github/workflows/full.yml", "status": "completed",
            "conclusion": "success", "created_at": created,
        }
        run.update(overrides)
        return run

    # --- the skip

    def test_bot_publications_since_a_green_run_are_verified(self) -> None:
        self._bot_publication(1)
        self._bot_publication(2)
        verdict = verify(self.repo, [self._run(self.base, run_id=7)])
        self.assertTrue(verdict.verified, verdict.reason)
        self.assertEqual((verdict.commit, verdict.run_id), (self.base, 7))

    def test_head_itself_green_is_verified(self) -> None:
        self.assertTrue(verify(self.repo, [self._run(self.base)]).verified)

    def test_an_older_green_ancestor_is_used_when_the_newest_is_elsewhere(self) -> None:
        self._bot_publication()
        runs = [
            self._run("f" * 40, run_id=2, created="2026-10-25T00:00:00Z"),
            self._run(self.base, run_id=1),
        ]
        verdict = verify(self.repo, runs)
        self.assertTrue(verdict.verified, verdict.reason)
        self.assertEqual(verdict.run_id, 1)

    # --- every reason to run the suite

    def test_a_human_commit_since_the_green_run_requires_the_suite(self) -> None:
        self._bot_publication()
        self._commit({"scripts/model.py": "x = 2\n"}, HUMAN_EMAIL)
        verdict = verify(self.repo, [self._run(self.base)])
        self.assertFalse(verdict.verified)
        self.assertIn("is not a bot commit", verdict.reason)

    def test_a_human_commit_to_bot_paths_still_requires_the_suite(self) -> None:
        """Ownership is the author's as well as the path's."""

        self._commit({"data/raw/pollofpolls/swedishpolls_polls.csv": "edited\n"},
                     HUMAN_EMAIL)
        self.assertFalse(verify(self.repo, [self._run(self.base)]).verified)

    def test_a_bot_commit_outside_its_paths_requires_the_suite(self) -> None:
        self._commit({"scripts/model.py": "x = 3\n"}, BOT_EMAIL)
        verdict = verify(self.repo, [self._run(self.base)])
        self.assertFalse(verdict.verified)
        self.assertIn("touches scripts/model.py", verdict.reason)

    def test_human_owned_data_is_not_bot_owned(self) -> None:
        """The calibration inputs live under data/ too, and the suite reads them."""

        self._commit({"data/processed/calibration/params.json": "{}\n"}, BOT_EMAIL)
        self.assertFalse(verify(self.repo, [self._run(self.base)]).verified)
        self.assertFalse(is_bot_owned("data/processed/calibration/params.json"))
        self.assertFalse(is_bot_owned("data/README.md.bak"))
        self.assertTrue(is_bot_owned("data/README.md"))

    def test_runs_that_did_not_pass_on_main_do_not_count(self) -> None:
        self._bot_publication()
        for override in (
            {"conclusion": "failure"},
            {"conclusion": "cancelled"},
            {"status": "in_progress", "conclusion": None},
            {"head_branch": "feature"},
            {"path": ".github/workflows/pr.yml"},
        ):
            with self.subTest(override=override):
                verdict = verify(self.repo, [self._run(self.base, **override)])
                self.assertFalse(verdict.verified)
                self.assertEqual(verdict.reason, "NO_SUCCESSFUL_FULL_CI_ANCESTOR")

    def test_no_runs_requires_the_suite(self) -> None:
        self.assertFalse(verify(self.repo, []).verified)

    def test_a_broken_repository_requires_the_suite(self) -> None:
        verdict = verify(self.repo / "missing", [self._run(self.base)])
        self.assertFalse(verdict.verified)
        self.assertTrue(verdict.reason.startswith("CHECK_FAILED"), verdict.reason)

    # --- the CLI the workflow calls

    def _cli(self, runs_text: str) -> tuple[str, str]:
        runs = Path(self._tmp.name) / "runs.json"
        output = Path(self._tmp.name) / "output"
        summary = Path(self._tmp.name) / "summary"
        runs.write_text(runs_text)
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["--repo-root", str(self.repo), "--runs-json", str(runs),
                         "--output", str(output), "--summary-path", str(summary)])
        self.assertEqual(code, 0)
        return output.read_text(), summary.read_text()

    def test_the_cli_reports_and_summarises_a_skip(self) -> None:
        self._bot_publication()
        output, summary = self._cli(json.dumps({"workflow_runs": [self._run(self.base, run_id=9)]}))
        self.assertEqual(output, "verified=true\nreason=VERIFIED_BY_FULL_CI\n")
        self.assertIn(f"SKIPPED_VERIFIED_BY_FULL_CI (commit {self.base}, run 9)", summary)

    def test_an_unreadable_run_list_runs_the_suite_and_never_fails(self) -> None:
        for text in ("", "{}", "not json", '{"workflow_runs": 3}'):
            with self.subTest(text=text):
                output, summary = self._cli(text)
                self.assertTrue(output.startswith("verified=false\nreason=RUNS_UNREADABLE"), output)
                self.assertIn("Backend contract tests: REQUIRED", summary)


class WorkflowGateTests(unittest.TestCase):
    """The skip removes the suite, and nothing else."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.publish = WORKFLOW.read_text(encoding="utf-8").split("\n  publish:\n")[1]

    def _step(self, name: str) -> str:
        return self.publish.split(f"- name: {name}")[1].split("\n      - ")[0]

    def test_the_check_runs_after_the_precheck_and_before_the_suite(self) -> None:
        order = [self.publish.index(f"- name: {name}") for name in (
            "Check whether there is anything to publish",
            "Check whether Full CI already verified this code",
            "Run backend and history contract tests",
        )]
        self.assertEqual(order, sorted(order))

    def test_only_the_suite_is_gated_on_it(self) -> None:
        gate = "steps.ci_verified.outputs.verified != 'true'"
        self.assertIn(gate, self._step("Run backend and history contract tests").split("\n")[1])
        for step in ("Set up Jekyll and Chromium", "Acquire, validate, publish, test, and sync"):
            with self.subTest(step=step):
                self.assertNotIn("ci_verified", self._step(step))

    def test_a_failed_check_is_never_a_skip(self) -> None:
        check = self._step("Check whether Full CI already verified this code")
        self.assertIn("id: ci_verified", check)
        self.assertIn("scripts.ci.verified_code", check)
        self.assertIn("echo '{}' > \"$runs\"", check)
        self.assertIn('echo "verified=false" >> "$GITHUB_OUTPUT"', check)
        self.assertIn("git status --porcelain", check)

    def test_the_job_may_read_the_run_list(self) -> None:
        permissions = self.publish.split("    permissions:\n")[1].split("\n    # The only job")[0]
        self.assertIn("actions: read", permissions)
        self.assertIn("contents: write", permissions)


if __name__ == "__main__":
    unittest.main()
