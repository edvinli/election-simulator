"""The scheduled riksdag workflow's contract: its lock, its clock, its paths, its token.

Asserted against the workflow text, as the other workflow tests in this suite
are (PyYAML is not a dependency). Comments are stripped before asserting what
the YAML does, so an explanation cannot satisfy an assertion.
"""

from __future__ import annotations

import re
import shlex
import unittest
from pathlib import Path

from scripts.riksdag_votes.constants import (
    PROCESSED_DIR_RELATIVE,
    WEBSITE_DIR_RELATIVE,
    WORKFLOW_CONCURRENCY_GROUP,
)
from scripts.riksdag_votes.publish import ALLOWED_PATHS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "riksdag-votes.yml"

FORBIDDEN_GROUPS = (
    "election-simulator-production",
    "election-simulator-render",
    "election-simulator-fallback-preflight",
    "election-simulator-reconstruction-canary",
    "calibration-2030-t5-shadow",
    "nightly",
)


def without_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def minutes(hour: int, minute: int) -> int:
    return hour * 60 + minute


class RiksdagWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = WORKFLOW.read_text(encoding="utf-8")
        cls.yaml = without_comments(cls.raw)
        # Shell continuation lines joined, so one command is one line.
        cls.commands = cls.yaml.replace("\\\n", " ")

    def test_own_concurrency_group_only(self) -> None:
        groups = re.findall(r"^\s*group:\s*(\S+)\s*$", self.yaml, re.MULTILINE)
        self.assertEqual(groups, [WORKFLOW_CONCURRENCY_GROUP])
        self.assertRegex(self.yaml, r"(?m)^concurrency:\n  group: riksdag-votes\n  cancel-in-progress: false$")
        for group in FORBIDDEN_GROUPS:
            self.assertNotIn(group, self.yaml)

    def test_group_is_not_used_by_any_other_workflow(self) -> None:
        for path in WORKFLOWS.glob("*.yml"):
            if path == WORKFLOW:
                continue
            groups = re.findall(r"^\s*group:\s*(\S+)\s*$", without_comments(path.read_text(encoding="utf-8")), re.MULTILINE)
            self.assertNotIn(WORKFLOW_CONCURRENCY_GROUP, groups, path.name)

    def test_triggers(self) -> None:
        on = self.yaml.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertIn("  schedule:", on)
        self.assertIn("  workflow_dispatch:", on)
        self.assertNotRegex(on, r"(?m)^  (push|pull_request|workflow_run):")
        dry_run = on.split("dry_run:", 1)[1].split("sessions:", 1)[0]
        self.assertIn("default: true", dry_run)
        self.assertIn("type: boolean", dry_run)

    def test_schedule_avoids_the_protected_interval_and_the_0400_run(self) -> None:
        crons = re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', self.yaml, re.MULTILINE)
        self.assertTrue(crons)
        for cron in crons:
            minute, hour, day, month, weekday = cron.split()
            self.assertEqual((day, month, weekday), ("*", "*", "*"), cron)
            for h in (int(x) for x in hour.split(",")):
                for m in (int(x) for x in minute.split(",")):
                    t = minutes(h, m)
                    self.assertFalse(minutes(18, 30) <= t < minutes(22, 0), cron)
                    self.assertGreaterEqual(abs(t - minutes(4, 0)), 120, f"{cron} is too close to the 04:00 publication")
                    # Before the hourly publication poll checks start at 07:00.
                    self.assertLess(t, minutes(7, 0), cron)

    def test_dry_run_gates_every_push(self) -> None:
        self.assertIn("DRY_RUN: ${{ github.event_name == 'workflow_dispatch' && inputs.dry_run && 'true' || 'false' }}", self.yaml)
        steps = self.yaml.split("\n      - ")
        pushing = [step for step in steps if "publish commit" in step]
        self.assertEqual(len(pushing), 2)
        for step in pushing:
            condition = re.search(r"^\s*if: (.*)$", step, re.MULTILINE).group(1)
            self.assertIn("env.DRY_RUN != 'true'", condition)
            self.assertIn("steps.quiet.outputs.quiet == 'yes'", condition)

    def test_only_the_riksdag_paths_are_staged(self) -> None:
        self.assertEqual(set(ALLOWED_PATHS), {PROCESSED_DIR_RELATIVE, WEBSITE_DIR_RELATIVE})
        self.assertEqual((PROCESSED_DIR_RELATIVE, WEBSITE_DIR_RELATIVE), ("data/processed/riksdag_votes", "files/riksdag"))
        # No raw staging, committing or pushing: all of it goes through publish.py.
        for forbidden in ("git add", "git commit", "git push", "--force", "push -f", "+refs/"):
            self.assertNotIn(forbidden, self.commands, forbidden)
        commits = [line for line in self.commands.splitlines() if "scripts.riksdag_votes.publish commit" in line]
        self.assertEqual(len(commits), 2)
        targets = set()
        for line in commits:
            words = shlex.split(line.split("uv run ", 1)[1])
            targets.add((words[words.index("--repo") + 1], words[words.index("--path") + 1], words[words.index("--branch") + 1]))
        self.assertEqual(targets, {(".", PROCESSED_DIR_RELATIVE, "main"), ("../website", WEBSITE_DIR_RELATIVE, "master")})
        # The website copy touches only files/riksdag.
        copies = [line.strip() for line in self.commands.splitlines() if re.match(r"\s*(cp|rsync|mv|rm)\b", line)]
        self.assertEqual(copies, ['cp -R "$SITE_DIR/." website/files/riksdag/'])

    def test_website_token(self) -> None:
        self.assertEqual(sorted(set(re.findall(r"secrets\.(\w+)", self.yaml))), ["WEBSITE_REPO_TOKEN"])
        website = self.yaml.split("- name: Check out website", 1)[1].split("\n      - ", 1)[0]
        self.assertIn("repository: edvinli/edvinli.github.io", website)
        self.assertIn("ref: master", website)
        self.assertIn("token: ${{ secrets.WEBSITE_REPO_TOKEN }}", website)

    def test_permissions_setup_and_failure_issue(self) -> None:
        permissions = self.yaml.split("\npermissions:\n", 1)[1].split("\njobs:", 1)[0]
        self.assertIn("contents: write", permissions)
        self.assertIn("issues: write", permissions)
        self.assertIn("uses: astral-sh/setup-uv@v5", self.yaml)
        self.assertIn("uv python install 3.12", self.yaml)
        self.assertIn("uv sync --frozen --python 3.12", self.yaml)
        self.assertIn("uses: actions/cache@v4", self.yaml)
        self.assertIn('title="Riksdag votes update failing"', self.yaml)
        self.assertRegex(self.yaml, r"(?m)^\s*if: failure\(\)$")

    def test_runs_every_riksdag_test_module(self) -> None:
        modules = sorted(p.stem for p in (REPOSITORY_ROOT / "tests").glob("test_riksdag_votes_*.py"))
        for module in modules:
            self.assertIn(f"tests.{module}", self.yaml, module)

    def test_header_documents_the_trigger_facts(self) -> None:
        self.assertIn("GITHUB_TOKEN push", self.raw)
        self.assertIn("starts no workflow", self.raw)
        self.assertIn("full.yml", self.raw)


if __name__ == "__main__":
    unittest.main()
